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
        
        # Resize to standard size to ensure consistent processing
        img = cv2.resize(img, size)
        
        # Extract color histogram (8 bins per RGB channel)
        # Using a color histogram provides much better features than raw grayscale pixels
        # since it captures overall color distribution, making it robust to slight rotations or shifts.
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histogram mathematically via OpenCV
        cv2.normalize(hist, hist)
        
        # Flatten the 3D histogram into a 1D vector list (8x8x8 = 512 dimensions)
        features = hist.flatten().tolist()
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_distance(vec1, vec2):
    """Computes Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))
