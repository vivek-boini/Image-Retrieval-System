import cv2
import numpy as np

def extract_features(image_path, size=(64, 64)):
    """
    Reads an image, resizes it, and extracts a 3D color histogram.
    Returns a flattened list of histogram bins.
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize to standard size to ensure consistent baseline processing
        img = cv2.resize(img, size)
        
        # 1. Base Color Feature Extraction (8x8x8 = 512 dimensions)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        
        # 2. Structural Texture & Edge Feature Extraction (16x16 = 256 dimensions)
        # Explanation: Why color-only features were insufficient?
        # A banana and a yellow sports car generate nearly identical global color histograms. 
        # By blending a smoothing blur and isolating macro-structure boundaries using Edge mapping,
        # the MLlib pipelines evaluate physical shape boundaries preventing false object equivalents!
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray_blurred, threshold1=100, threshold2=200)
        
        # Resize the raw edge map down to a 16x16 macro-grid maintaining stable spatial shift-invariance.
        edges_macro = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA)
        
        # 3. Mathematical Dimensional Normalization (L2 Norm bounds)
        # True Equilibrium: Splitting weights 50/50 normally leaves dimensional bias when lengths differ heavily (512 vs 256 dims).
        # We actively enforce L2 vector norms, ripping apart raw magnitudes before securely applying symmetric weights!
        color = np.array(hist.flatten(), dtype=np.float32)
        color = color / (np.linalg.norm(color) + 1e-8)
        
        edges_flat = np.array(edges_macro.flatten(), dtype=np.float32) / 255.0
        edges_flat = edges_flat / (np.linalg.norm(edges_flat) + 1e-8)
        
        # Unify normalized dimensions enforcing an absolute 50/50 algorithmic relationship correctly
        color = color * 0.5
        edges_flat = edges_flat * 0.5
        
        # 4. Concatenate cleanly into one functional flat Python list (768 structured features)
        combined_features = np.concatenate([color, edges_flat]).tolist()
        return combined_features

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_distance(vec1, vec2):
    """Computes Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))
