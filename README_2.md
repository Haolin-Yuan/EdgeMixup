## Skin disease segmentation and classification
# Experiments and Goals
The experiments and goals are described in `2022_lyme_seg_bias_paper.pdf` (this paper is unpublished).

# Setting up the environment
```
conda env create -f environment.yml
conda activate lyme
cd lyme_segmentation
export SM_FRAMEWORK=tf.keras
```


# Segmentation Data
1. To point code to segmentation data, modify ```ROOT_DATA_DIR``` in ```config.py```
2. Modify paths in ```config.py``` to fit your machine's user folder structure.

# Classification Data
1. In ```disease_classification/clf_config.py``` modify ```ROOT_CLF_IMG_DIR``` with the path to the classification data
    Skin disease classification data is located at ```im_harvest_2019/images_July2019```
2. Modify paths in ```disease_classification/clf_config.py``` to fit your machine's folder structure.


# How to run segmentation:
1. Regenerating Train, Validation, and Test Splits
    ```python train_segmentation_model.py --regenerate-splits```
2. Train segmentation model
    ```python train_segmentation_model.py --train```
3. Evaluate segmentation model
    ```python train_segmentation_model.py --test```
4. Generate segmentation masks for all splits
    ```python train_segmentation_model.py --plot-all```

# ITA
The segmentation testing process generates the ITA content automatically, but should you want to only generate the ITA data, the following options are available:

1. Calculate the ITA for the ground truth and predicted masks
    ```python ita.py --calculate-ita```
2. Plot the ITA distribution
    ```python ita.py --plot```
3. Calculate the segmentation prediction's bias metrics
    ```python ita.py --bias-metrics```

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


# How to run disease classification

1. On Segmentation tasks: ```python train_segmentation_model.py --iterative```