import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
import math
from utils import pnet_boxes, rnet_boxes, onet_boxes, face_alignment, L2_distance


def mtcnn(pnet, rnet, onet, image):
    img = tf.to_pil_image(image)
    p_bounding_boxes = pnet_boxes(img, pnet)
    if len(p_bounding_boxes)==0:
        return None, None, None
    r_bounding_boxes = rnet_boxes(img, rnet, p_bounding_boxes)
    if len(r_bounding_boxes)==0:
        return None, None, None
    o_bounding_boxes, pred_ldmk = onet_boxes(img, onet, r_bounding_boxes)
    if len(o_bounding_boxes)==0:
        return None, None, None

    aligned_faces = []
    for i in range(len(o_bounding_boxes)):
        box = o_bounding_boxes[i]
        ldmk = pred_ldmk[i]
        aligned = face_alignment(img, box, ldmk)
        aligned_faces.append(aligned)
        
    return o_bounding_boxes, pred_ldmk, aligned_faces


def fn(facenet, image, embeddings, threshold):
    image = tf.resize(image, (224,224))
    image = tf.to_tensor(image)
    image = tf.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = image.expand(1,3,224,224)

    facenet.eval()
    image = image.cuda()
    decode = facenet(image)

    min_dist = threshold
    most_likely = 'unknown'
    for name, eb in embeddings.items():
        dist = L2_distance(decode,eb)
        if (dist < threshold) and (dist < min_dist):
            min_dist = dist
            most_likely = name

    return most_likely, min_dist


def decode_face(facenet, image, threshold):
    image = tf.resize(image, (224,224)).convert('RGB')
    image = tf.to_tensor(image)   
    image = tf.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = image.expand(1,3,224,224)

    facenet.eval()
    image = image.cuda()
    decode = facenet(image)

    return decode