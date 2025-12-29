# Your ViT is Secretly an Image Segmentation Model  

This code is adapted from the [📄 Paper](https://arxiv.org/abs/2503.19108) to use it as a common framework for comparing different variants of Vision Transformers for the task of image segmentation.

Please refer to the original in case you were mislead -> [Original Repo](https://github.com/tue-mps/eomt)



## Overview

Similar to the Benchmarking project at [Which Transformer to favor](https://github.com/tobna/WhatTransformerToFavor). This project aims to create a common framework that can compare variants of Vision Transformers for the task of image segmentation.

## Adaptations

- Decoupled Queries and Image Features processing to allow backbone flexibility: `models\eomt.py`
- One way Cross-attention between queries derived from images vs. key/values from image features: `models\cross_attention.py`
- Change Learning scheduler to cosine instead of poly: ` training\two_stage_warmup_cosine_schedule.py`

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```
## Supported Models
| Architecture | Versions                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Linformer    | linformer_vit_tiny_patch16, linformer_vit_small_patch16, linformer_vit_base_patch16, linformer_vit_large_patch16  |
| ViT          | ViT-{Ti,S,B,L}/<patch_size>                                                                                                  |
| Swin         | swin_tiny_patch4_window7, swin_small_patch4_window7, swin_base_patch4_window7, swin_large_patch4_window7          |
## Usage

### Training

```

```