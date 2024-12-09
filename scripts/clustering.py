import pandas as pd
from sklearn.cluster import KMeans

def cluster_data(file_path, n_clusters=3):
    """
    Clusters the data for balancing low-resource languages.
    """
    data = pd.read_csv(file_path)
    features = data[['premise', 'hypothesis']].applymap(lambda x: len(x))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    data.to_csv(file_path, index=False)
    print("Clustering completed and saved.")