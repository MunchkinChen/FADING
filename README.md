# FADING

## Face Aging via Diffusion-based Editing
[![arXiv](https://img.shields.io/badge/arXiv-2309.11321-b31b1b)](https://arxiv.org/abs/2309.11321)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://proceedings.bmvc2023.org/595/)
[![Static Badge](https://img.shields.io/badge/supplementary-blue)](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0595_supp.pdf)

Official repo for BMVC 2023 paper: [Face Aging via Diffusion-based Editing](https://proceedings.bmvc2023.org/595/).

For more visualization results, please check [supplementary materiels](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0595_supp.pdf).

<div align="center">
    <a><img src="FADING_demo.png"  width="100%" ></a>
</div>

> In this paper, we address the problem of face aging—generating past or future facial images by incorporating age-related changes to the given face. Previous aging methods rely solely on human facial image datasets and are thus constrained by their inherent scale and bias. This restricts their application to a limited generatable age range and the inability to handle large age gaps. We propose FADING, a novel approach to address Face Aging via DIffusion-based editiNG. We go beyond existing methods by leveraging the rich prior of large-scale language-image diffusion models. First, we specialize a pre-trained diffusion model for the task of face age editing by using an age-aware fine-tuning scheme. Next, we invert the input image to latent noise and obtain optimized null text embeddings. Finally, we perform text-guided local age editing via attention control. The quantitative and qualitative analyses demonstrate that our method outperforms existing approaches with respect to aging accuracy, attribute preservation, and aging quality.



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
