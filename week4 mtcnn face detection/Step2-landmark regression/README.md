## Process of training MTCNN
train P-Net ——> generate data for R-Net ——> train R-Net ——> generate data for O-Net ——> train O-Net 

## Trained on two datasets
<table>
    <tr>
        <td>Dataset</td>
        <td>Phase</td>
        <td>P-Net</td>
        <td>R-Net</td>
        <td>O-Net</td>
    </tr>
    <tr>
        <td rowspan="2">FPD</td>
        <td>train</td>
        <td>10,000</td>
        <td>84,430</td>
        <td>58,380</td>
    </tr>
    <tr>
        <td>val</td>
        <td>3,466</td>
        <td>29,365</td>
        <td>20,504</td>
    </tr>
    <tr>
        <td rowspan="2">CelebA</td>
        <td>train</td>
        <td>162,770</td>
        <td>2,985,821</td>
        <td>1,351,104</td>
    </tr>
    <tr>
        <td>val</td>
        <td>19,867</td>
        <td>359,155</td>
        <td>165,438</td>
    </tr>
</table>


### References
https://github.com/TropComplique/mtcnn-pytorch<br>
https://github.com/GitHberChen/MTCNN_Pytorch
