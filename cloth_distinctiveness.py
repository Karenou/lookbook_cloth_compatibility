"""
calculate the item distinctiveness from other items
within the same category and posted within 3 months
"""

from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
import argparse
import h5py
from dateutil.relativedelta import relativedelta

CATEGORY_MAPPING = {0: 'background', 1: 'upper body', 2: 'lower body', 3: 'full body'}

def read_embed_vector(h5_path, df):
    """
    read embed_vector, dim: (n, 67, 64)
    @param h5_path: path to load h5 file
    @param df: mapping dataframe
    """
    embed_hdf5 = h5py.File(h5_path, 'r')
    embed_vectors = []
    
    for i in range(len(df)):
        user_id = df.loc[i, "user_id"]
        look_id = df.loc[i, "look_id"]
        idx = df.loc[i, "idx"]
        embed_vectors.append(embed_hdf5[str(user_id)][str(look_id)][idx])

    embed_hdf5.close()
    embed_vectors = np.array(embed_vectors)
    return embed_vectors

def compute_compactness(matrix):
    """
    compute the cluster compactness using Frobenius norm
    cluster center: median
    @param matrix: a matrix of embedding vectors with shape (n, 67, 64), reshape to (n, 67*64)
    """
    cluster_center = np.median(matrix, axis=0, keepdims=True)
    # Frobenius norm
    fro_norm = np.linalg.norm(matrix - cluster_center, ord='fro')
    return fro_norm

def compute_distinctiveness(matrix):
    """
    calculate the item distinctiveness from other items
    @param matrix: a matrix of embedding vectors with shape (n, 67, 64), reshape to (n, 67 * 64)
    """
    # reshape to (n, 67 * 64)
    n = matrix.shape[0]
    matrix = matrix.reshape(n, -1)
    compactness = compute_compactness(matrix)

    # compute sum of pairwise distance to other items within the cluster
    pairwise_dist = squareform(pdist(matrix, metric='euclidean'))
    distinctiveness = np.sum(pairwise_dist, axis=1) 

    # normalize by cluster compactness
    distinctiveness /= compactness

    return distinctiveness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_vector", type=str, help="Path to load embed_vector.h5")
    parser.add_argument("--embed_mapping", type=str, help="Path to load embed_item_mapping.csv")
    parser.add_argument("--look", type=str, help="Path to load look.csv")
    parser.add_argument("--save_path", type=str, help="Path to save the output distinctiveness score")
    args = parser.parse_args()

    mapping_df = pd.read_csv(args.embed_mapping)
    look = pd.read_csv(args.look)
    look = look[["Look_ID", "DateCreated"]]
    look["DateCreated"] = look["DateCreated"].apply(lambda x: x[:7])

    scores = None

    # filter each item category: 1, 2, 3
    for item_id in range(1, 4):
        print("Item category: %s" % CATEGORY_MAPPING[item_id])
        data = mapping_df[mapping_df["item_id"] == item_id]
        data = data.merge(look, how="inner", left_on="look_id", right_on="Look_ID")

        month_list = sorted(data["DateCreated"].unique())
        # filter last three months posts to form a cluster
        for month in month_list:
            past_3m = (datetime.strptime(month, "%Y-%m") - relativedelta(months=3)).strftime("%Y-%m")
            cluster = data[(data["DateCreated"] > past_3m) & \
                            (data["DateCreated"] <= month)]\
                    .sort_values("DateCreated") \
                    .reset_index(drop=True)

            print("month: %s, cluster size: %d" % (month, len(cluster)))

            # only record the score for current dt
            idx = cluster[cluster["DateCreated"] == month].index[0]
            embed_vec = read_embed_vector(args.embed_vector, cluster)
            distinctiveness = compute_distinctiveness(embed_vec)

            score = cluster[cluster["DateCreated"] == month][["user_id", "look_id", "item_id"]]
            score["distinctiveness_score"] = distinctiveness[idx:]

            if scores is None:
                scores = score
            else:
                scores = pd.concat([scores, score], axis=0)


    # average the score by user_id and look_id
    print("save scores to csv")
    scores = scores.groupby(["user_id", "look_id"]).agg({'distinctiveness_score': 'mean'})
    scores.sort_values(["user_id", "look_id"]).to_csv(args.save_path)