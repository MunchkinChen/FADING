# FADING

## Dataset
FFHQ-Aging Dataset is downloaded from https://github.com/royorel/FFHQ-Aging-Dataset

## Specialization
```shell
accelerate launch specialize.py \
--instance_data_dir 'specialization_data/training_images' \
--instance_age_path 'specialization_data/training_ages.npy' \
--output_dir <PATH_TO_SAVE_MODEL> \
--max_train_steps 150
```
Training images should be saved at `specialization_data/training_images`. The training set is described through `training_ages.npy` that contains the age of the training images.
```angular2html
array([['00007.jpg', '1'],
       ['00004.jpg', '35'],
        ...
       ['00009.jpg', '35']], dtype='<U21')
```

## Age Editing

```shell
python age_editing.py \
--FFHQ_path <PATH_TO_FFHQ_DATASET> \
--FFHQ_label_path <PATH_TO_ffhq_aging_labels.csv> \
--FFHQ_id  1 \
--save_aged_dir <OUTPUT_DIR> \
--specialized_path  <PATH_TO_SPECIALIZED_MODEL> \
--target_ages 10 20 40 60 80
```