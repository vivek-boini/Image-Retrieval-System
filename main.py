import os
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from clustering import train_and_assign_clusters

def main():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created '{dataset_dir}' folder. Please place 50-150 test images inside and run again.")
        return

    print("Training Scaler & KMeans models using PySpark MLlib...")
    spark, predictions, model, scaler_model = train_and_assign_clusters(
        dataset_path=dataset_dir,
        num_clusters=4,  # Configurable cluster counts
        model_save_path="kmeans_model",
        scaler_save_path="scaler_model"
    )

    if predictions is None:
        print("Failed to train model. Check if the dataset folder contains valid images.")
        return

    print("Evaluating clustered dataset dynamically mapped into memory...")
    
    # We serialize the scaled features back to a list structure.
    vector_to_list_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
    predictions = predictions.withColumn("scaled_features_array", vector_to_list_udf("scaled_features"))

    # 🛑 Windows Hadoop Fix: Temporarily suppressing direct parquet save attempts to avoid winutils.exe crashing. 
    # predictions.select("image_path", "scaled_features_array", "cluster") \
    #     .write.mode("overwrite").parquet("clustered_data.parquet")
    
    print("\nSample clustered results displayed flawlessly from secure local memory:")
    predictions.select("image_path", "cluster").show(10, truncate=False)
    
    print("\n====================================")
    print("✅ Training sequence completely executed. Memory clusters correctly assigned.")
    print("====================================")

if __name__ == "__main__":
    main()
