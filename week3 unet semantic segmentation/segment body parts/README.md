## Segment different body parts
File | # Classes | # Train Images | # Epochs | Data Augmentation | Continue Trained On  | Eval Accuracy
|:-----------------:|:----:|:----:|:----:|:----:|:-----------------:|:-----------------:|
UNm_10000_30e.ipynb | 20 |10000(I) | 30 | HFlip | ×  | ×
UNm_30000_10e.ipynb | 20 |30000 | 10 | HFlip | UNm_10000_30e.ipynb | 0.23
UNm_10000II_30e_ed.ipynb | 20 | 10000(II) | 30 | HFlip, elastic deformation | UNm_10000_30e.ipynb | 0.26
UNm_10000_30e_MoreAug.ipynb | 20 | 10000(I) | 30 | HFlip, VFlip, Rotation, ColorJitter, elastic deformation| UNm_10000II_30e_ed.ipynb | 0.27
FewerClasses_10000III_20e.ipynb | 10 | 10000(III) | 20 | HFlip, VFlip, Rotation, elastic deformation | UNm_10000II_30e_ed.ipynb | 0.46
FewerClasses_10000_20e_448.ipynb | 10 | 10000(I) | 30 | HFlip, VFlip, Rotation, ColorJitter | FewerClasses_10000III_20e.ipynb  | 0.44
Apply depthwise separable conv.ipynb | 20 | × | × | × | ×  | ×

`UNm_10000_30e ——> UNm_30000_10e`<br>
`UNm_10000_30e ——> UNm_10000II_30e_ed ——> UNm_10000_30e_MoreAug`<br>
`UNm_10000_30e ——> UNm_10000II_30e_ed ——> FewerClasses_10000III_20e ——> FewerClasses_10000_20e_448`


