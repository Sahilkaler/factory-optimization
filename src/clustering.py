from sklearn.cluster import KMeans

def cluster_routes(df):
    kmeans = KMeans(n_clusters=4)
    df['route_cluster'] = kmeans.fit_predict(df[['distance', 'lead_time']])
    return df