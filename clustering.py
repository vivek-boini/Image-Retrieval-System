from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import os
import glob
from feature_extraction import extract_features

def build_spark_session(app_name="ImageClusteringApp"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

def train_and_assign_clusters(dataset_path, num_clusters=5, model_save_path="kmeans_model", scaler_save_path="scaler_model"):
    spark = build_spark_session()
    
    # 1. Load image paths
    image_paths = glob.glob(os.path.join(dataset_path, "*.*"))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    # Safeguard: Limit over-parallel execution timeouts by bounding the target images
    if len(image_paths) > 100:
        print(f"Dataset limitation active: Trimmed down from {len(image_paths)} to 100 images to ensure local Windows stability.")
        image_paths = image_paths[:100]
    
    if not image_paths:
        print("No images found in the dataset folder.")
        return spark, None, None, None

    # 2. Extract features sequentially in pure Python BEFORE DataFrame creation.
    # Explanation: Why OpenCV should not run inside a Spark UDF?
    # Spawning multiple heavy graphical OpenCV C-bindings across PySpark worker processes asynchronously
    # frequently causes deadlocks, memory leaks, and socket.timeout EOF errors on local architectures.
    # Why this architecture improves stability:
    # A hybrid approach leverages a standard Python main-thread loop for complex image extraction,
    # thereby eliminating worker crashes. PySpark only accepts the clean data afterward for strict scaled ML tasks.
    
    extracted_data = []
    print(f"Extracting features across {len(image_paths)} images via native Python...")
    for i, path in enumerate(image_paths):
        if i > 0 and i % 10 == 0:
            print(f"Processed {i} images...")
            
        feats = extract_features(path)
        if feats is not None:
            clean_feats = [float(f) for f in feats]
            extracted_data.append((path, clean_feats))
    
    if len(extracted_data) == 0:
        print("Failed to extract valid features from image paths.")
        return spark, None, None, None

    # Load structured extraction tuples cleanly into PySpark framework
    df_features = spark.createDataFrame(extracted_data, ["image_path", "features_array"])

    # 3. Convert array to Spark MLlib Vector
    # Explaintion: Safely converting to vectors directly using raw Resilient Distributed Datasets (RDDs)
    # Eliminates any remaining unstable UDF calls internally for pure architectural resilience. 
    df_vectors = df_features.rdd.map(
        lambda row: (row["image_path"], Vectors.dense(row["features_array"]))
    ).toDF(["image_path", "raw_features"])
    
    # Performance Security: Cache structures strictly before deep mathematical functions
    # Prevents worker exhaustion attempting to recompute upstream Python extracts repeatedly.
    df_vectors = df_vectors.cache()
    
    print("Beginning Scaler Initialization and KMeans MLlib Training...")

    # 4. Feature Normalization (IMPORTANT)
    # Explanation: Features often have varying scales depending on extraction. 
    # StandardScaler transforms features to have mean=0 and variance=1. This is critical for K-Means 
    # to prevent larger numerical metrics from artificially skewing and dominating the Euclidean distance.
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_vectors)
    df_scaled = scaler_model.transform(df_vectors)

    # 🛑 Windows Hadoop Fix: Temporarily suppressing direct save attempts to avoid winutils.exe crashing. 
    # scaler_model.write().overwrite().save(scaler_save_path)

    # 5. Train KMeans Model
    # Explanation: KMeans is chosen because it efficiently partitions n observations into k clusters natively in distributed space.
    kmeans = KMeans().setK(num_clusters).setSeed(42).setFeaturesCol("scaled_features").setPredictionCol("cluster")
    model = kmeans.fit(df_scaled)

    # 🛑 Windows Hadoop Fix: Temporarily suppressing direct save attempts to avoid winutils.exe crashing. Everything runs dynamically in memory safely.
    # model.write().overwrite().save(model_save_path)

    # 6. Assign Clusters
    predictions = model.transform(df_scaled)

    return spark, predictions, model, scaler_model
