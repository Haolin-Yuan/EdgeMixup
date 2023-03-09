# EdgeMixup


 
There are two datasets: one for segmentation model and one for classification model. 

For the segmentation dataset, we annotate skin images into three classes: background (black), skin (yellow), and lesion (blue). 
The lesion area contains three types of disease/lesions: Tinea Corporis (TC), Herpes Zoster (HZ), and Erythema Migrans (EM).

<!-- <img width="722" alt="image" src="https://user-images.githubusercontent.com/73618869/158436013-da5dc2ae-be16-4cad-a497-e0b08ff70b1a.png"> -->

The classification dataset has 2,712 samples, and we annotate those skin images into four classes: No Disease (NO), TC, HZ, and EM.

<!-- We use Individual Topology Angle (ITA) for both datastes as a proxy for skin tone labels, and the distribution of skin tones are shown below.

<img width="670" alt="image" src="https://user-images.githubusercontent.com/73618869/158436296-c9dc6e2b-e1f0-4fdd-bdcf-ffc7699c271a.png">

<img width="668" alt="image" src="https://user-images.githubusercontent.com/73618869/158437478-ff5c19fa-c8fe-4dee-bc42-70e6067a0a8c.png"> -->

<h2>Download</h2>

1. [Segmentation dataset](https://anonymfile.com/7P2Xb/03-032023.zip)

2. [Classification dataset](https://anonymfile.com/bVzrd/lyme-data.zip)


# How to run segmentation:
1. Regenerating Train, Validation, and Test Splits
    ```python train_segmentation_model.py --regenerate-splits```
2. Train segmentation model
    ```python train_segmentation_model.py --train```
3. Evaluate segmentation model
    ```python train_segmentation_model.py --test```
4. Generate segmentation masks for all splits
    ```python train_segmentation_model.py --plot-all```

![Segmentation Results](segmentation_results.png)


# How to run disease classification
1. Generate segmentation masks for the classification images calculate the image's ITA, and clean the data splits.
    ```python classify_disease.py --generate-masks```
2. Train a model:
    1. Image only model: ```python classify_disease.py --train --model-type baseline```
    2. Image with adversarial debias model: ```python classify_disease.py --train --model-type AD```
    3. Masked images model: ```python classify_disease.py --train --model-type masked```
    4. Masked images with adversarial debias model: ```python classify_disease.py --train --model-type masked+AD```
3. Evaluate a model:
    1. Image only model: ```python classify_disease.py --test --model-type baseline```
    2. Image with adversarial debias model: ```python classify_disease.py --test --model-type AD```
    3. Masked images model: ```python classify_disease.py --test --model-type masked```
    4. Masked images with adversarial debias model: ```python classify_disease.py --test --model-type masked+AD``

![Cls Results](classification_results.png)

Test