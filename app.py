from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

app = Flask(__name__)

# Load dataset from UCI
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
COLUMN_NAMES = [
    'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]
df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
true_labels = df['Class'].values
X = df.drop('Class', axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ensure plots directory exists
os.makedirs("static/plots", exist_ok=True)

# Temporary file to store clustered CSV
clustered_csv_path = "static/clustered_data.csv"

def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set2')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    filepath = f'static/plots/{title}.png'
    plt.savefig(filepath)
    plt.close()
    return filepath

def save_clustered_data(labels):
    result_df = df.copy()
    result_df['Cluster'] = labels
    result_df.to_csv(clustered_csv_path, index=False)

def run_kmeans():
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    rand_index = adjusted_rand_score(true_labels, labels)
    img_path = plot_clusters(X_pca, labels, "KMeans")
    return {'name': 'KMeans', 'labels': labels, 'sil': sil_score, 'rand': rand_index, 'img': img_path}

def run_pam():
    distance_matrix = calculate_distance_matrix(X_scaled.tolist())
    initial_medoids = random.sample(range(len(X_scaled)), 3)
    pam_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
    pam_instance.process()
    clusters = pam_instance.get_clusters()
    labels = np.zeros(len(X_scaled))
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = i
    sil_score = silhouette_score(X_scaled, labels)
    rand_index = adjusted_rand_score(true_labels, labels)
    img_path = plot_clusters(X_pca, labels, "PAM")
    return {'name': 'PAM', 'labels': labels, 'sil': sil_score, 'rand': rand_index, 'img': img_path}

def run_clara():
    sample_size = int(0.2 * len(X_scaled))
    best_score = -1
    best_medoids = None

    for _ in range(5):
        sample_indices = random.sample(range(len(X_scaled)), sample_size)
        sample = X_scaled[sample_indices]
        distance_matrix = calculate_distance_matrix(sample.tolist())
        initial_medoids = random.sample(range(sample_size), 3)

        pam_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        pam_instance.process()
        clusters = pam_instance.get_clusters()

        sample_labels = np.zeros(sample_size)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                sample_labels[idx] = i

        score = silhouette_score(sample, sample_labels)
        if score > best_score:
            best_score = score
            best_medoids = pam_instance.get_medoids()

    distance_matrix_full = calculate_distance_matrix(X_scaled.tolist())
    clara_final = kmedoids(distance_matrix_full, best_medoids, data_type='distance_matrix')
    clara_final.process()
    final_clusters = clara_final.get_clusters()
    labels = np.zeros(len(X_scaled))
    for i, cluster in enumerate(final_clusters):
        for idx in cluster:
            labels[idx] = i

    sil_score = silhouette_score(X_scaled, labels)
    rand_index = adjusted_rand_score(true_labels, labels)
    img_path = plot_clusters(X_pca, labels, "CLARA")
    return {'name': 'CLARA', 'labels': labels, 'sil': sil_score, 'rand': rand_index, 'img': img_path}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def run():
    results = [run_kmeans(), run_pam(), run_clara()]
    best_algorithm = max(results, key=lambda x: x['sil'])
    return render_template("results.html", results=results, best=best_algorithm)

@app.route("/download")
def download():
    return send_file(clustered_csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
