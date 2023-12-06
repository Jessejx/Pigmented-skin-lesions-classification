# Pigmented-skin-lesions-classification

Pigmented skin lesions are almost common around the world, since human skin is generally exposed and highly vulnerable to strong ultraviolet rays 
and external pathogens, with almost all people having several pigmented lesions. Pigmented skin lesions are caused by melanocyte cells in the body.
Malignant skin cancer such as melanoma, basal cell carcinoma, or squamous-cell carcinoma is one of the pigmented skin lesions, with a high degree of malignancy, 
rapid metastasis, difficulty to find, and other characteristics, early diagnosis of skin cancer can effectively improve the cure rate.  
Pigmented skin lesion:
<div align=center>
<img src="https://github.com/Jessejx/Pigmented-skin-lesions-classification/blob/0da837f86c085a44f1d1a1b4d54c438b5f2853b1/Downstream-pigmented-skin-lesion-classification/figs/7-skin-lesion.png" width="1050px">
</div>


## Getting Started  
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on your local machine.

### Configurations

Please install the following libraries

1. python 3.7.0
2. pytorch 1.7.0 + cu110
3. albumentations 0.5.2
4. tqdm
5. opencv-python
6. numpy

## Pretrained weights
pretrained weights are in: [Pretraining weights](https://pan.baidu.com/s/1_eAXc1Xg9r_e-I-so4y4Cw) Password is: t5nq

## Datasets
The pretraining dataset can be downloaded from the following URLs:

1. [Pretraining dataset](https://challenge.isic-archive.com/data/)
2. [segmentation dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Downstream classification 
In this repositories, we present downstream pigmented skin lesions classification task (KNN and Transfer learning):
### For transfer learning
1. cd to "downstream-task"
2. Download the pretrained weights:
3. Load the pretrained weitghts from backbone network in the '5-fold-cross.py':
4. Afterward, please run the '5-fold-cross.py' to classifier skin lesion
### For KNN
1. cd to "downstream task"
2. Run the "Knn_train.py" directly.

## Self-supervised-learning
For skin images unsupervised training, we present proposed method in the paper:
1. cd to self-supervised-learning
2. run the 'train.py' to unsupervised training.

## Visualization

TSEN Results:

<div align=center>
<img src="https://github.com/Jessejx/Pigmented-skin-lesions-classification/blob/0da837f86c085a44f1d1a1b4d54c438b5f2853b1/Downstream-pigmented-skin-lesion-classification/figs/tsne.png" width="1050px">
</div>

CAM Results:

<div align=center>
<img src="https://github.com/Jessejx/Pigmented-skin-lesions-classification/blob/0da837f86c085a44f1d1a1b4d54c438b5f2853b1/Downstream-pigmented-skin-lesion-classification/figs/cam.png" width="450px">
</div>

## Citation
If you use the proposed framework (or any part of this code in your research), please cite the following paper:

## Contact
If you have any query, please feel free to contact us




