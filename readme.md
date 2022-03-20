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

There are 176,505 images with 2 detected items, and 83,596 images with 3 detected items. We only embed and predict the compatibility scores for images containing multiple detected items.

## Step 3: Embed cloth segment images into vectors and compute compatibility scores

- Reference

The code is forked from [fashion-compatability](https://github.com/mvasil/fashion-compatibility).

- Procedure

First, need to download the pretrain model [here](https://drive.google.com/file/d/1JrRgM_EaLQqLw1CNjM65XnTm9rZyLRgj/view), and put it under `fashion-compatibility/model/model_best.pth.tar`

The embeded vectors are saved in hdf5 format with hierarchy being `str(user_id)/str(look_id)`. The embeded dimension is 64 for each detected item. To track the hierarchy in hdf5 file, a mapping csv is saved in `data/embed_item_id_mapping.csv`, which gives the mapping from look_id to item_id in `data/cloth_images` to idx in embeded vectors in hdf5. 

Note that the batch_size must be set to 1 for calculating the compatibility score for multiple items in a single look_id image.

```
python fashion-compatibility/predict.py --resume="fashion-compatibility/model/model_best.pth.tar" --batch_size=1 --dim_embed=64 --base_path="data/cloth_images" --segmentation_csv="data/segmentation_output.csv" --output_score="data/compatability_score.csv" --output_mapping="data/embed_item_id_mapping.csv" --output_hdf5="data/embed_vector.h5"
```


## Step 4: Compute cloth item distinctiveness score

- Definition of a cluster

Items within the same cloth category (upper body, lower body or full body) and are posted in the latest three months. 

For example, when computing the distinctiveness of upper body cloth items posted in Mar 2018, we choose the upper body cloth items posted from Jan to Mar 2018 to form the cluster.


- Measure of cluster center

Median of embedding vectors of all items within the cluster.

- Definition of cluster compactness 

Summation of Euclidean distance between embedding vector of each item in the cluster and the cluster center.

- Definition of item distinctiveness

How different this cloth item is from other items within the cluster.

The difference is measured by summation of pairwise Euclidean distance between this item from other items within the cluster, and further normalized by the cluster compactness.


- How to run the program

The code is written in `cloth_distinctiveness.py`. We need to load the embedding vector h5 file - `data/embed_vector.h5`, the mapping csv from h5 file to the item_id - `data/embed_item_id_mapping.csv` and the look csv - `data/look.csv`.

The program will first loop each of the item category, and then a second loop of the latest month. For each latest month, a cluster is formed by including the items posted in the past 3 months, and distinctiveness of items posted in the latest month are computed. 

Run the following command, the output score is saved in `data/distinctiveness_score.csv`

```
python cloth_distinctiveness.py --embed_vector="data/embed_vector.h5" --embed_mapping="data/embed_item_id_mapping.csv"
--look="data/look.csv"
--save_path="data/distinctiveness_score.csv"
```
