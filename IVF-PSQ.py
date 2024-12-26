import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import sqlite3

m = 8
D = 768
subvector_dim = D // m
M = 100
K = 150
embeddings_file = "embedding.csv"

data = pd.read_csv(embeddings_file)
vector_data = data.iloc[1:, :].values.astype(float)

kmeans_coarse = KMeans(n_clusters=M, random_state=42)
kmeans_coarse.fit(vector_data)
coarse_clusters = kmeans_coarse.labels_
coarse_centers = kmeans_coarse.cluster_centers_

coarse_clusters_sheets = {}
for i in range(M):
    cluster_data = vector_data[coarse_clusters == i]
    coarse_clusters_sheets[f"Cluster_{i}"] = pd.DataFrame(cluster_data)

residual_clusters_sheets = {}
residual_centers = []
integer_residual_sheets = {}

db_path = "inverted_index.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS coarse_to_residual (
        coarse_cluster_id INTEGER,
        residual_center BLOB
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS residual_to_integer (
        coarse_cluster_id INTEGER,
        residual_cluster_id INTEGER,
        integer_subvector BLOB
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS coarse_to_center (
        coarse_cluster_id INTEGER,
        coarse_center BLOB
    )
''')

for i in tqdm(range(M)):
    cluster_data = vector_data[coarse_clusters == i]
    cluster_center = coarse_centers[i]
    residuals = cluster_data - cluster_center
    residual_clusters_sheets[f"Residual_Cluster_{i}"] = pd.DataFrame(residuals)
    all_subvectors = residuals.reshape(-1, subvector_dim)
    effective_K = min(K, len(all_subvectors))
    kmeans_fine = KMeans(n_clusters=effective_K, random_state=42)
    kmeans_fine.fit(all_subvectors)
    residual_centers.append(kmeans_fine.cluster_centers_)

    for k in range(effective_K):
        subvector_cluster = all_subvectors[kmeans_fine.labels_ == k]
        integer_cluster = np.round(subvector_cluster * 255).astype(np.uint8)
        cursor.execute('''
            INSERT INTO residual_to_integer (coarse_cluster_id, residual_cluster_id, integer_subvector)
            VALUES (?, ?, ?)
        ''', (i, k, integer_cluster.tobytes()))

    cursor.execute('''
        INSERT INTO coarse_to_center (coarse_cluster_id, coarse_center)
        VALUES (?, ?)
    ''', (i, cluster_center.tobytes()))

    cursor.execute('''
        INSERT INTO coarse_to_residual (coarse_cluster_id, residual_center)
        VALUES (?, ?)
    ''', (i, kmeans_fine.cluster_centers_.tobytes()))

conn.commit()
conn.close()