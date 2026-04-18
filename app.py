import streamlit as st
import os
import tempfile
from PIL import Image

# Initialize PySpark requirements
from pyspark.sql import SparkSession

from search import process_query_image, perform_search
from clustering import train_and_assign_clusters

@st.cache_resource
def initialize_pipeline_in_memory():
    # 🛑 Windows Hadoop Fix: Bypassing Parquet IO entirely by engaging ML components directly into localized caching!
    dataset_dir = "dataset"
    spark, predictions, model_km, model_scaler = train_and_assign_clusters(
        dataset_path=dataset_dir,
        num_clusters=4,  
        model_save_path="memory_only_kmean",
        scaler_save_path="memory_only_scaler"
    )
    
    if predictions is not None:
        # Repackage the ML vector arrays to match Python's flat mathematical structure for search distance mappings
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, DoubleType
        vector_to_list_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
        predictions = predictions.withColumn("scaled_features_array", vector_to_list_udf("scaled_features"))
        
    return spark, predictions, model_km, model_scaler

st.set_page_config(page_title="Image Retrieval System", layout="wide", page_icon="🖼️")

st.title("✨ Optimized Spatial Image Retrieval")
st.markdown("Multi-stage retrieve scaling **PySpark MLlib (K-Means & Scaler)** natively in purely cached memory.")

try:
    spark, df, model_km, model_scaler = initialize_pipeline_in_memory()
except Exception as e:
    st.error(f"⚠️ Memory Mapping Failed: {e}")
    st.stop()

if df is None or model_km is None:
    st.error("⚠️ Target Dataset folder is completely missing or empty! Please assign images strictly into 'dataset/' before executing searches.")
    st.stop()

st.sidebar.header("Settings & Search Setup")
top_k = st.sidebar.slider("Maximum Retrieve Count (K)", min_value=1, max_value=20, value=5)
# Note: Standardization sets variance roughly to 1, changing expected distance thresholds 
distance_threshold = st.sidebar.slider("Maximum Scaled Distance Tolerance", min_value=0.0, max_value=50.0, value=25.0, step=1.0)

uploaded_file = st.sidebar.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Query Input", use_column_width=True)

    if st.sidebar.button("🔍 Execute Intelligent Search"):
        with st.spinner("Extracting multi-color histograms and scaling coordinates..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Process Query (Feature extraction -> Scaler -> KMeans)
                query_scaled_list, predicted_cluster = process_query_image(tmp_path, model_km, model_scaler, spark)
                
                st.success(f"**Stage 1 Optimization:** Request safely routed to **Cluster {predicted_cluster}**")

                # Perform Search (DataFrame Filtering -> Euclidean Distance Calculation)
                results = perform_search(df, query_scaled_list, predicted_cluster, top_k=top_k)
                
                filtered_results = [r for r in results if r['distance'] <= distance_threshold]
                
                st.markdown("### Processed Matches")
                if not filtered_results:
                    st.warning(f"No sufficiently similar images found within distance limit {distance_threshold}.")
                else:
                    cols = st.columns(min(3, len(filtered_results)))
                    for idx, res in enumerate(filtered_results):
                        img_path = res["image_path"]
                        dist = res["distance"]
                        c_id = res["cluster"]
                        
                        try:
                            res_img = Image.open(img_path)
                            with cols[idx % 3]:
                                # Display results cleanly using a Grid Structure with precision captions
                                st.image(res_img, use_column_width=True)
                                st.caption(f"🎯 Distance: {dist:.3f} | 📁 Cluster: {c_id}")
                        except BaseException as e:
                            st.error(f"Cannot load file reference: {img_path}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    st.info("Upload an image on the left side to observe optimized multi-stage retrieval.")
