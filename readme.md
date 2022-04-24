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


- Rationale of type-specific embedding

It first learns a single, general embedding space, and then projects items from the general embedding space to the subspaces identified by pair of types. For example, all bottoms that are compatible with a given top item, will be close in the top-bottom subspace. While these bottom items may vary differently in the general embedding space (depend on their similarity among the bottom items).

The rationale of the type-specific embedding is to consider type in describing similarity and compatability. If there is only one single embedding space:
1. All shoes that are close to a bottom must be close to the bottom, then these shoes are also close to each other, which cannot further distinguish their similarity within the bottom type.  
2. It does not allow items match in one context, not the other. But compatibility is not transitive. (A is compatible with B, B is compatible with C, but it does not imply that A is compatible with C.)


- How to match type spaces

The polyvore dataset have 11 semantic categories, including tops, bottoms, all-body, shoes, bags, outerwear, sunglasses, jewellery, hats, accessories, scarves. There are 66 type spaces derived from the 11 categories, including (shoes, bags), (tops, bottoms) (top, all-body) etc. 

As in the lookbook dataset, the cloth items fall into the upper-body, lower-body and full-body, which corresponding to the tops, bottoms and all-body categories in polyvore dataset. Thus, we mainly interact with the type spaces related to tops, bottoms and all-body in later image embedding and compatibility measure.


- Image embedding

The pretrained model uses an embedding dimension of 64. To get the embeddings of items within an outfit, we pass the segmented item's image (anchor) to the model and get an embedding matrix of shape (67, 64). The first row is the general embedding and the remaining rows are type-specific embeddings. Then we obtain the anchor and its paired image's types to get the corresponding type space and extract the type-specific embedding in the embedding matrix with the type space index. Below are several examples.

1. For an outfit that only detects one item, such as top. We extract the row of (top, top) embedding from the embedding matrix.

2. For an outfit that contains two itemes such as a top and a bottom. If the current anchor is the top item, we extract the row of (top, bottom) embedding from the embedding matrix. Same for the bottom item.

3. If the outfit contains three items: top, bottom and all-body. When the top item is the anchor, we extract the rows of (top, bottom) and (top, all-body) from the embedding matrix and take the average as the final image embedding for the top item.


- Compatibility measure

Compatibility is measured by the average of pairwise Euclidean distance (L2-norm) among the type-specific embedding of cloth items within an outfit. 

For example, for outfit containing 3 items and we have the embeddings v1, v2, and v3. We compute the pairwise Euclidean distance of (v1, v2), (v1, v3) and (v2, v3), take the average to get the final compatibility score.

Compatibility score is only applicable for outfits containing more than one items. For outfits that only one item is detected, the compatibility score is None in this case.

Notice that here compatibility score is a distance measure, so a larger score means that the items within an outfit are less compatible.

- Procedures

First, need to download the pretrain model [here](https://drive.google.com/file/d/1JrRgM_EaLQqLw1CNjM65XnTm9rZyLRgj/view), and put it under `fashion-compatibility/model/model_best.pth.tar`

The embeded vectors are saved in hdf5 format with hierarchy being `str(user_id)/str(look_id)`. The embeded dimension is 64 for each detected item. To track the hierarchy in hdf5 file, a mapping csv is saved in `data/embed_item_id_mapping.csv`, which gives the mapping from look_id to item_id in `data/cloth_images` to idx in embeded vectors in hdf5. 

Note that the batch_size must be set to 1 for calculating the compatibility score for multiple items in a single look_id image. The root directory is `lookbook` folder.

```
python fashion-compatibility/predict.py --resume="fashion-compatibility/model/model_best.pth.tar" --batch_size=1 --dim_embed=64 --base_path="data/cloth_images" --segmentation_csv="data/segmentation_output.csv" --output_score="data/compatibility_score.csv" --output_mapping="data/embed_item_id_mapping.csv" --output_hdf5="data/embed_vector.h5" --typespace_path="fashion-compatibility/data/polyvore_outfits/nondisjoint"
```

- Plot charts to evaluate pretrain model's performance on polyvore dataset

1. Dataset

To evaluate pretrained model's performance on the dataset used in the paper, we downloaded the polyvore outfit dataset [here](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view), unzip and put it under the `fashion_compatibility/data` folder. There are three datatsets: (1) maryland dataset, (2) disjoint polyvore outfit, (3) nondisjoint polyvore outfit. We use the last two datasets. The difference between the disjoint and nondisjoint dataset is the way of splitting the train, validation and test set. For nondisjoint one, the outfits in respective set do not overlap while the individual items may overlap. For disjint one, both the outfits and individual items don't overlap in all three sets.

2. Code

The root directory is `fashion_compatibility` folder. The code is written in `test.py` and helper functions to plot roc and precision-recall curves are added in `polyvore_outfits.py`. The save path arguments are passed in `test_compatibility` function, and `plot_roc_curve` and `plot_pr_curve` functions define the layout of charts. By alternating the polyvore_split argument ("nondisjoint" or "disjoint"), charts on different dataset are saved in `result` folder.

```
python test.py --test --l2_embed --polyvore_split="nondisjoint" --resume="model/model_best.pth.tar"
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

Each detected item will have a distinctivess score.


- How to run the program

The code is written in `cloth_distinctiveness.py`. We need to load the embedding vector h5 file - `data/embed_vector.h5`, the mapping csv from h5 file to the item_id - `data/embed_item_id_mapping.csv` and the look csv - `data/look.csv`.

The program will first loop each of the item category, and then a second loop of the latest month. For each latest month, a cluster is formed by including the items posted in the past 3 months, and distinctiveness of items posted in the latest month are computed. 

Run the following command, the output score is saved in `data/distinctiveness_score.csv`

```
python cloth_distinctiveness.py --embed_vector="data/embed_vector.h5" --embed_mapping="data/embed_item_id_mapping.csv" --look="data/look.csv" --save_path="data/distinctiveness_score.csv"
```