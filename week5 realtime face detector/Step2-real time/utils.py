import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf
import numpy as np
from PIL import Image, ImageDraw
import math
import random
import copy
import cv2


def L2_distance(x1,x2):
    assert x1.size() == x2.size()
    eps = 1e-4 / x1.size(1)
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff, 2).sum(dim=1)
    return torch.pow(out + eps, 1./2).item()


def run_first_stage(image, net, scale, threshold):
    
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = transforms.ToTensor()(img).unsqueeze(0)
    img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    output = net(img)
    probs = output[0].data.cpu().numpy()[0, 0, :, :]
    offsets = output[1].data.cpu().numpy()
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):

    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])

    return bounding_boxes.T

def nms(boxes, overlap_threshold=0.5, mode='union'):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode is 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep


def convert_to_square(bboxes):

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    
    num_boxes = len(bounding_boxes)
    width, height = img.size
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        try:
            img_box = np.zeros((h[i], w[i], 3), 'uint8')
        except:
            continue
        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        
        if img_box.shape[0]*img_box.shape[1]!=0:
            img_box = Image.fromarray(img_box)
            img_box = img_box.resize((size, size), Image.BILINEAR)
            img_box = np.asarray(img_box, 'float32')
            img_boxes[i, :, :, :] = img_normalization(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    x, y, ex, ey = x1, y1, x2, y2
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0
    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list


def img_normalization(img):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def run_first_stage(image, net, scale, threshold):
    
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = transforms.ToTensor()(img).unsqueeze(0)
    img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    output = net(img)
    probs = output[0].data.cpu().numpy()[0, 0, :, :]
    offsets = output[1].data.cpu().numpy()
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def pnet_boxes(img, pnet, min_face_size=15.0, thresholds=[0.85, 0.03, 0.5], nms_thresholds=[0.5, 0.9, 0.3]):
    pnet.eval()
    width, height = img.size
    min_length = min(height, width)
    min_detection_size = 12
    factor = 0.707
    scales = []
    m = min_detection_size / min_face_size
    min_length *= m
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    bounding_boxes = []
    for s in scales: 
        boxes = run_first_stage(img, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
        
    bounding_boxes = [i for i in bounding_boxes if i is not None]

    try:
        _ = bounding_boxes[0]

    except Exception:
        pass
    if len(bounding_boxes) == 0:
        return []
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    
    
    return bounding_boxes


def rnet_boxes(img, rnet, bounding_boxes, thresholds=[0.85, 0.03, 0.5], nms_thresholds=[0.5, 0.9, 0.3]):
    rnet.eval()
    img_boxes = get_image_boxes(bounding_boxes, img, size=24)
    img_boxes = torch.FloatTensor(img_boxes)
    img_boxes = img_boxes.cuda()
    if img_boxes.size(0)==0:
        return []
    output = rnet(img_boxes)
    probs = output[0].data.cpu().numpy()
    offsets = output[1].data.cpu().numpy()

    keep = np.where(probs[:, 0] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    return bounding_boxes


def onet_boxes(img, onet, bounding_boxes, thresholds=[0.85, 0.03, 0.5], nms_thresholds=[0.5, 0.9, 0.3]):
    onet.eval()
    img_boxes = get_image_boxes(bounding_boxes, img, size=48)
    if len(img_boxes) == 0:
        return [],[]
    img_boxes = torch.FloatTensor(img_boxes)
    img_boxes = img_boxes.cuda()
    
    output = onet(img_boxes)
    probs = output[0].data.cpu().numpy()
    offsets = output[1].data.cpu().numpy()
    landmarks = output[2].data.cpu().numpy()

    keep = np.where(probs[:, 0] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]

    bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

    landmarks[:, 0::2] = (np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0::2]).copy()
    landmarks[:, 1::2] = (np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 1::2]).copy()

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='union')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]
    return bounding_boxes, landmarks


def face_alignment(img, box, ldmk):

    w, h = img.size
    box = box[:4]
    
    tan_angle = (ldmk[3]-ldmk[1])/(ldmk[2]-ldmk[0])
    eye_angle = np.degrees(np.arctan(tan_angle))
    M = cv2.getRotationMatrix2D((ldmk[4]/2, ldmk[5]/2), eye_angle, 1)
    aligned = cv2.warpAffine(np.array(img), M, (w, h))
    
    box_p = (np.append(box,[box[0],box[3],box[2],box[1]])).reshape(-1,2).T
    new_box_p = np.dot(M[:,:2],box_p) + M[:,2:3]

    x1 = min(new_box_p[0])
    y1 = min(new_box_p[1])
    x2 = max(new_box_p[0])
    y2 = max(new_box_p[1])

    dx = x2-x1
    dy = y2-y1
    x1 = max(0, x1-0.2*dx)
    y1 = max(0, y1-0.2*dy)
    x2 = min(w-1, x2+0.2*dx)
    y2 = min(h-1, y2+0.2*dy)

    crop_img = aligned[int(y1):int(y2),int(x1):int(x2),:]
    crop_img = tf.to_pil_image(crop_img)

    return crop_img