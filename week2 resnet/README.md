## Compare transform
File | # Batch Size | LR | Data Augmentation  | Initialization
:--------:|:--------------:|:--------:|:-----------------:|:-----------------:|
Resnet50 scratch - try different transform.ipynb | 64  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | <font color=#AOO>default: </font>RandomHorizontalFlip<br>ColorJitter(0.25 brightness, contrast, hue) | kaiming normal 
Resnet50 scratch - try different transform.ipynb | 64  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | <font color=#AOO>CenterCrop in eval</font> | kaiming normal 
Resnet50 scratch - try different transform.ipynb | 64  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | <font color=#AOO>RandomResizedCrop in train<br>scale=(0.08,1)</font> | kaiming normal 
Resnet50 scratch - try different transform.ipynb | 64  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | <font color=#AOO>RandomResizedCrop in train<br>scale=(0.5,1)</font> | kaiming normal 
Resnet50 scratch - try different transform.ipynb | 64  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | <font color=#AOO>RandomResizedCrop in train<br>scale=(0.75,1)</font> | kaiming normal 

## Compare batchsize
File | # Batch Size | LR | Data Augmentation  | Initialization
:--------:|:--------------:|:--------:|:-----------------:|:-----------------:|
Resnet50 scratch - try different batchsize.ipynb | <font color=#AOO>8</font>  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | default| kaiming normal 
Resnet50 scratch - try different batchsize.ipynb | <font color=#AOO>16</font>  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | default| kaiming normal
Resnet50 scratch - try different batchsize 3.ipynb | <font color=#AOO>32</font>  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | default| kaiming normal
Resnet50 scratch - try different batchsize 3.ipynb | <font color=#AOO>64</font>  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | default| kaiming normal
Resnet50 scratch - try different batchsize.ipynb | <font color=#AOO>128</font>  | `Adam(lr=0.001,weight_decay=1e-5)` with **no** lr_schedular | default| kaiming normal

## Compare batchsize
File | # Batch Size | LR | Data Augmentation  | Initialization
:--------:|:--------------:|:--------:|:-----------------:|:-----------------:|
resnet50 scratch - try no lr decay.ipynb | 64 |<font color=#AOO> `SGD(lr=0.025, momentum=0.9)`</font> | default| kaiming normal 
resnet50 scratch - try no lr decay.ipynb | 64 | <font color=#AOO>`SGD(lr=0.01, momentum=0.9)`</font> | default| kaiming normal 
resnet50 scratch - try step lr decay 2.ipynb | 64 | <font color=#AOO>`SGD(lr=0.025, momentum=0.9)`<br>`StepLR(step_size=30, gamma=0.1)`</font> | default| kaiming normal 
resnet50 scratch - try no lr decay 2.ipynb | 64 | <font color=#AOO>`SGD(lr=0.025, momentum=0.9)`<br>`StepLR(step_size=20, gamma=0.5)`</font> | default| kaiming normal 

## Compare initialization
File | # Batch Size | LR | Data Augmentation  | Initialization
:--------:|:--------------:|:--------:|:-----------------:|:-----------------:|
Resnet18 scratch - kaiming initialization.ipynb | 64 | `SGD(lr=0.01, momentum=0.9)` | default| <font color=#AOO>kaiming normal</font>
Resnet18 scratch - default initialization.ipynb | 64 | `SGD(lr=0.01, momentum=0.9)` | default| <font color=#AOO>defult normal</font>

## Compare "bag of tricks"
File | # Batch Size | LR | Data Augmentation  | Initialization | criterion |
:--------:|:--------------:|:--------:|:-----------------:|:-----------------:|:-----------------:|
Resnet<font color=#AOO>50</font> scratch - no tricks.ipynb | 128 | <font color=#AOO>`SGD(lr=0.01, momentum=0.9)`</font> | default| kaiming normal | <font color=#AOO>CrossEntropyLoss</font>
Resnet<font color=#AOO>50D</font> scratch - with tricks.ipynb | 128 | <font color=#AOO>`SGD(lr=0.05, momentum=0.9)`<br>`CosineAnnealingWarmRestarts(T_0=120, eta_min=0, last_epoch=-1)`</font> | default| kaiming normal | <font color=#AOO>LabelSmoothing</font>







