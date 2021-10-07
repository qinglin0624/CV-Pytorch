## Train U-Net
File | # Train Images | # Epochs | Data Augmentation | Continue Trained On 
|:--------:|:--------------:|:--------:|:-----------------:|:---------------:|
UN_600_50e.ipynb | 600 | 50 | HFlip | × 
UN_1000_50e.ipynb | 1000 | 50 | HFlip | ×
UN_2000_30e.ipynb  | 2000 | 30 | HFlip | UN_600_50e.ipynb
UN_8000_30e.ipynb  | 8000 | 30 | HFlip | UN_2000_30e.ipynb 
UN_10000_20e_aug.ipynb  |  10000(I)   |   20   |     HFlip, VFlip, ColorJitter      |   UN_1000_50e.ipynb
UN_10000_20e_aug2.ipynb |  10000(I)   |   20   |     HFlip, VFlip, ColorJitter      |   UN_10000_20e_aug.ipynb
UN_10000III_20e_Dice.ipynb |  10000(III)   |   20   |     HFlip, VFlip      |   UN_10000_20e_aug.ipynb
UN_10000II_20e_Dice_etaug.ipynb |  10000(II)   |   20   |     HFlip, VFlip, elastic deformation, ColorJitter   |   UN_10000III_20e_Dice.ipynb
UN_FineTune_UNm.ipynb |  10000(III)  | 30  | HFlip, VFlip, Rotation | UNm_10000II_30e.ipynb
UN_FineTune2_UNm.ipynb |  10000(III)  | 30  | HFlip, VFlip, Rotation | UNm_10000III_20e_FewerClasses.ipynb


`UN_600_50e ——> UN_2000_30e ——> UN_8000_30e`<br>
`UN_1000_50e ——> UN_10000_20e_aug ——> UN_10000_20e_aug2`<br>
`UN_1000_50e ——> UN_10000_20e_aug ——> UN_10000III_20e_Dice ——> UN_10000II_20e_Dice_etaug`<br>
`UNm_10000II_30e ——> UN_finetune_UNm`<br>
`UNm_10000III_20e_FewerClasses.ipynb ——> UN_FineTune2_UNm`


## Compare MSE, BCE, Dice
File | # Train Images | # Epochs | Criterion | Data Augmentation | Continue Trained On 
|:--------:|:--------:|:-------:|:---------:|:---------:|:--------:|
UN_1500_60e_LossExperimentMSE.ipynb | 1500 | 60 | MSE | HFlip, VFlip | × 
UN_1500_60e_LossExperimentBCE.ipynb | 1500 | 60 | BCE | HFlip, VFlip | ×
UN_1500_60e_LossExperimentDice.ipynb  | 1500 | 60 | Dice | HFlip, VFlip | ×


## Compare tanh and sigmoid
File | # Train Images | # Epochs | Criterion | Data Augmentation | last activation | Continue Trained On 
|:--------:|:--------:|:-------:|:---------:|:---------:|:--------:|:--------:|
Compare tanh and sigmoid.ipynb | 1500 | 70 | MSE | HFlip, VFlip, contrast, brightness | tanh | × 
Compare tanh and sigmoid.ipynb | 1500 | 70 | MSE | HFlip, VFlip, contrast, brightness | sigmoid | ×