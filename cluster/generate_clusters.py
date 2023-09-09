import numpy as np
import pandas as pd

# import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import umap.umap_ as umap  # pip install umap-learn


def generate_clusters(
    message_embeddings, n_neighbors, n_components, min_cluster_size, random_state=None
):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(message_embeddings)

    clusters = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(umap_embeddings)

    return clusters


def generate_k_means_clusters(embeddings):
    best_num_clusters = 2
    max_silhouette_score = 0
    for n_clusters in range(2, len(embeddings) // 2):
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        # print(
        #     "For n_clusters =",
        #     n_clusters,
        #     "The average silhouette_score is :",
        #     silhouette_avg,
        # )

        if max_silhouette_score < silhouette_avg:
            max_silhouette_score = silhouette_avg
            best_num_clusters = n_clusters

    kmeans = KMeans(n_clusters=best_num_clusters, n_init="auto", random_state=42)
    kmeans.fit(embeddings)
    return max_silhouette_score, kmeans


def clusters_2_df(
    embeddings, clusters, w2v_model, df_texts, n_neighbors=15, min_dist=0.1
) -> pd.DataFrame:
    """

    Arguments:
        embeddings: embeddings to use
        clusters: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    umap_data = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        # metric='cosine',
        random_state=42,
    ).fit_transform(embeddings)

    centroids = clusters.cluster_centers_
    centroid_labels = [centroids[i] for i in clusters.labels_]

    texts = []
    for i in range(len(embeddings)):
        text = str(w2v_model.wv.most_similar(positive=embeddings[i], topn=1)[0][0])
        label_text = str(
            w2v_model.wv.most_similar(positive=centroid_labels[i], topn=1)[0][0]
        )
        texts.append(
            [
                text[: len(text) - len(df_texts["question"][0])],
                label_text[: len(label_text) - len(df_texts["question"][0])],
                # umap_data[i][0],
                # umap_data[i][1],
            ]
        )

    result = pd.DataFrame(texts, columns=["answer", "text_labels"])
    # print(result)
    result["labels"] = clusters.labels_

    clustered = result[result.labels != -1]

    return clustered


def score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num

    return label_count, cost
