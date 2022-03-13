# README

This repo is for a research project about prediction of fashion compatibility, supervised by professor June Shi, Department of Marketing, HKUST SBM.

## Data source
The data was collected from https://lookbook.nu/. Datasets 2 and 3 can be linked by user id. 

1. data/look: contains the images in jpg format. Paths follow user_id/look_id.jpg
2. look.csv: each entry corresponds to a post
3. lb_user.csv: each entry is a user

## Step 1: Crop single human body from images

As the raw image sometimes contains multiple human figures, we copped the largest one for further analysis.

The raw data is saved in `data/look/user_id/look_id.jpg` format, the mapping is saved in `data/look_mapping.csv`. Use `human_detection_frcnn.py` to crop the largest single human from the image. It will save the cropped image in `data/human_images/userID_lookID.jpg` 

```
python human_detection_frcnn.py --batch_size=32 --base_path="data/look" --csv_path="data/look_mapping.csv" --save_path="data/human_images"
```

Since there are 2,868 corrupted images, we filter the corrupted images (`data/empty_image.csv`) and save a new mapping csv in `data/look_mapping_1.csv`, which has 383,556 images left.


## Step 2: Segment cloth items

- Reference

The code is forked from [cloth-segmentation](https://github.com/levindabhi/cloth-segmentation).

- Procedure

The U2NET model classifies each pixel into four categories: {'background': 0, 'upper body': 1, 'lower body': 2, 'full body': 3} and save each detected item as a separate image in path `data/cloth_images/userID_lookID_itemID.jpg`. Due to cuda memory constraint, the max batch_size is up to 4. It also outputs a csv in `data/segmentation_output.csv` that gives the number of detected cloth items and their categories for each look_id image.

Download the pretrain model [here](https://drive.google.com/file/d/1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ/view) and put it under `cloth-segmentation/assets/model.pth`.

```
python cloth-segmentation/main.py --batch_size=4 --base_path="data/human_images" --mapping_path="data/look_mapping_1.csv" --model_path="cloth-segmentation/assets/model.pth" --save_path="data/cloth_images"
```

## Step 3: Embed cloth segment images into vectors and compute compatibility scores

- Reference

The code is forked from [fashion-compatability](https://github.com/mvasil/fashion-compatibility).

- Procedure

First, need to download the pretrain model [here](https://drive.google.com/file/d/1JrRgM_EaLQqLw1CNjM65XnTm9rZyLRgj/view), and put it under `fashion-compatibility/model/model_best.pth.tar`

The embeded vectors are saved in `data/embed_vector.h5` in group format of `str(user_id)/str(look_id)`. Since the pretrained model uses 66 type spaces to embed the item, plus a general embedding space. The embeded dimension is （67, 64) for each detected item.  To track the hierarchy in hdf5 file, a mapping csv is saved in `data/embed_item_id_mapping.csv`, which gives the mapping from look_id to item_id in `data/cloth_images` to idx in embeded vectors in hdf5. 

The compatibility score is saved in `data/compatibility_score.csv`

Note that the batch_size must be set to 1 for calculating the compatibility score for all items under a single look_id image.

```
python fashion-compatibility/predict.py --resume="fashion-compatibility/model/model_best.pth.tar" --batch_size=1 --dim_embed=64 --base_path="data/cloth_images" --segmentation_csv="data/segmentation_output.csv" --output_score="data/compatability_score.csv" --output_mapping="data/embed_item_id_mapping.csv" --output_hdf5="data/embed_vector.h5"
```

## Step 4: Compute cloth item distinctiveness score

Definition of distinctiveness: how different this cloth item is from other items within the same category and posted in the latest 3 months (fashion trend varies across time). The difference is measured by summation of pairwise distance from other items within the cluster, and further normalized by the cluster compactness.
