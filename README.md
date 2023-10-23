# FADING

## Dataset
The FFHQ-Aging Dataset used for training FADING could be downloaded from https://github.com/royorel/FFHQ-Aging-Dataset

## Training (Specialization)

### Available pretrained weights
We release weights of our specialized model at https://drive.google.com/file/d/1galwrcHq1HoZNfOI4jdJJqVs5ehB_dvO/view?usp=share_link

### Train a new model

```shell
accelerate launch specialize_general.py \
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

## Inference (Age Editing)

```shell
python age_editing.py \
--image_path <PATH_TO_INPUT_IMAGE> \
--age_init <INITIAL_AGE> \
--gender <female|male> \
--save_aged_dir <OUTPUT_DIR> \
--specialized_path  <PATH_TO_SPECIALIZED_MODEL> \
--target_ages 10 20 40 60 80
```
