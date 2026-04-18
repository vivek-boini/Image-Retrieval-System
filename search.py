from pyspark.ml.linalg import Vectors
from feature_extraction import extract_features, compute_distance

def process_query_image(query_img_path, kmeans_model, scaler_model, spark):
    """
    Extracts features, scales them using our pre-fitted model, and predicts the cluster.
    """
    features = extract_features(query_img_path)
    if features is None:
        raise ValueError(f"Could not extract features from query image: {query_img_path}")
    
    # Convert query into a PySpark DataFrame to run the StandardScaler properly
    df_query = spark.createDataFrame([(Vectors.dense(features),)], ["raw_features"])
    
    # Apply StandardScaling to normalize
    df_scaled = scaler_model.transform(df_query)
    scaled_vec = df_scaled.collect()[0]["scaled_features"]
    
    # Predict the cluster mathematically via assigned spatial boundaries from K-means
    cluster_id = kmeans_model.predict(scaled_vec)
    return scaled_vec.toArray().tolist(), cluster_id

def perform_search(predictions_df, query_scaled_features, expected_cluster, top_k=5):
    """
    Executes Multi-Stage Search logic.
    Stage 1: Discard irrelevant clusters.
    Stage 2: Filter remainder using precise Euclidean distance.
    """
    
    # Explanation: Why clustering drastically reduces search time?
    # By assigning labels offline, we skip matching against every repository image (N operations). 
    # We instantly filter PySpark's DataFrame down only to images matching our `expected_cluster` (K operations), 
    # shrinking scan-space substantially.
    cluster_df = predictions_df.filter(predictions_df.cluster == expected_cluster).collect()

    results = []
    # Stage 2: Fine-Tuning Similarity ranking
    for row in cluster_df:
        path = row["image_path"]
        row_feats = row["scaled_features_array"]
        
        dist = compute_distance(query_scaled_features, row_feats)
        results.append({
            "image_path": path,
            "distance": dist,
            "cluster": expected_cluster
        })

    # Sort results to get the lowest numeric distance first (safest matches)
    results_sorted = sorted(results, key=lambda x: x["distance"])
    return results_sorted[:top_k]
