import pandas as pd

import umap.umap_ as umap # pip install umap-learn

from w2v import file_2_vectors
from generate_clusters import generate_clusters, score_clusters
from create_all_data import file_2_df





if __name__ == "__main__":
    vectors, model = file_2_vectors("data/all/1036.json")
    data_json = file_2_df("data/all/1036.json")

    clusters_default = generate_clusters(
        vectors, n_neighbors=5, n_components=1, min_cluster_size=5, random_state=42
    )

    labels_def, cost_def = score_clusters(clusters_default)
    print(labels_def)
    print(cost_def)

    clustered_data = clusters_2_df(
        vectors, clusters_default, model, data_json, n_neighbors=5
    )
    print(clustered_data.head(5))
