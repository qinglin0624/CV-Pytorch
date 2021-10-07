## Process of  training MTCNN on WIDERFACE dataset
Step | Task | File  | Input | Output |
|:--------:|:-------:|:--------------:|:-------:|:--------:|
0 | generate customized txt file for data loading | Make_my_txt.ipynb | txt file in widerface dataset | train.txt, val.txt
1 | train P-Net | Train_PNet_1.ipynb<br>Train_PNet_2.ipynb<br>Train_PNet_3.ipynb | train.txt, val.txt | boxes predicted by P-Net
2 | generate training data for R-Net | Make_RNet_data.ipynb | boxes predicted by P-Net | rnet_train.txt, rnet_eval.txt
3 | train R-Net  | Train_RNet_bigdata.ipynb | rnet_train.txt, rnet_eval.txt | boxes refined by R-Net
4 | generate training data for O-Net | Make_ONet_data.ipynb | boxes predicted by P-Net (and refined by R-Net) | onet_train.txt, onet_eval.txt
5 | train O-Net | Train_ONet_1.ipynb | onet_train.txt, onet_eval.txt | boxes outputted by O-Net
6 | test | Test_MTCNN.ipynb | P-Net, R-Net, O-Net | predictions


### Summary of the models finalized

#### P-Net
- trained on all the boxes of 15k random images in widerface dataset
- 120 epochs in total
- loss decreases to 0.26 when the coefficients are 1 for class and 0.5 for box

#### R-Net
- trained on all the boxes of all the images in widerface dataset
- 220 epochs in total
- loss decreases to 0.27 when the coefficients are **2** for class and 0.5 for box

#### O-Net
- trained on all the boxes of all the images in widerface dataset
- 60 epochs in total
- loss decreases to 0.10 when the coefficients are 1 for class and 0.5 for box


### References
https://github.com/TropComplique/mtcnn-pytorch<br>
https://github.com/GitHberChen/MTCNN_Pytorch
