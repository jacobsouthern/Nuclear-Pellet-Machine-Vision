import cv2
import numpy as np
import os
import glob
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def calculate_texture_metrics(img):
    """
    Calculates texture features (GLCM) from the center of the pellet.
    Ignores shape/geometry entirely.
    """
    # 1. Get the Center Patch
    # We assume the images are 300x300 (from your preprocessing).
    # We take a 100x100 patch from the center to avoid edges and background.
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    patch_size = 50 # Radius (so 100x100 total)
    
    # Extract the patch
    patch = img[center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size]
    
    # Safety Check: If the patch contains ANY pure white (255) background pixels,
    # the pellet might be off-center or too small. We should ignore this sample
    # to prevent skewing the data with "edge" contrast.
    if np.any(patch == 255):
        # Optional: Try a smaller patch? For now, returns None to be safe.
        return None

    # 2. Compute GLCM (Gray-Level Co-occurrence Matrix)
    # distances=[1]: Look at immediate neighbor pixels
    # angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]: Look in all 4 directions (horizontal, vertical, diagonal)
    # levels=256: For 8-bit grayscale
    glcm = graycomatrix(patch, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)

    # 3. Extract Features (Average across all 4 directions)
    
    # CONTRAST: Measures local variations. 
    # High Contrast = Rough texture, heavy layer lines, noise.
    # Low Contrast = Smooth, polished surface.
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    
    # DISSIMILARITY: Similar to contrast but scales linearly.
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    
    # HOMOGENEITY: Measures closeness of distribution to diagonal.
    # High Homogeneity = Very smooth, uniform surface.
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    
    # ENERGY: Measures orderliness.
    # High Energy = Constant image (very little variation).
    energy = np.mean(graycoprops(glcm, 'energy'))
    
    # CORRELATION: Measures linear dependency.
    # High Correlation = Predictable patterns (like consistent layer lines).
    # Low Correlation = Random noise/static.
    correlation = np.mean(graycoprops(glcm, 'correlation'))

    return {
        'Contrast': contrast,
        'Dissimilarity': dissimilarity,
        'Homogeneity': homogeneity,
        'Energy': energy,
        'Correlation': correlation
    }

def analyze_folder(folder_path):
    metrics_list = []
    filenames = []
    
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    print(f"Analyzing {len(files)} images in {folder_path}...")
        
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            m = calculate_texture_metrics(img)
            if m:
                metrics_list.append(m)
                filenames.append(os.path.basename(f))
            else:
                print(f"Skipping {os.path.basename(f)}: Background detected in texture patch.")
    
    return pd.DataFrame(metrics_list, index=filenames)

def main():
    TRAIN_FOLDER = 'processed_images/train_good'
    TEST_FOLDER = 'processed_images/test_mixed'
    
    # --- 1. TRAIN (Learn Normal Texture) ---
    df_train = analyze_folder(TRAIN_FOLDER)
    
    if df_train.empty:
        print("Error: No valid training data found.")
        return

    # Calculate Mean and Std Dev
    stats = df_train.describe().loc[['mean', 'std']]
    print("\n--- TRAINING STATS (Normal Texture) ---")
    print(stats)

    # Define Tolerance
    SIGMA = 3.0 
    
    limits = {}
    for col in df_train.columns:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]
        limits[col] = (mean - (SIGMA * std), mean + (SIGMA * std))
        
    print("\n--- TEXTURE THRESHOLDS ---")
    for k, v in limits.items():
        print(f"{k}: {v[0]:.4f} to {v[1]:.4f}")

    # --- 2. TEST (Detect Texture Defects) ---
    df_test = analyze_folder(TEST_FOLDER)
    
    print("\n--- TEST RESULTS ---")
    for filename, row in df_test.iterrows():
        reasons = []
        
        for metric, (low, high) in limits.items():
            val = row[metric]
            
            # Logic:
            # High Contrast/Dissimilarity = Too Rough (Defect)
            # Low Homogeneity/Energy = Too Messy (Defect)
            
            if metric in ['Contrast', 'Dissimilarity']:
                if val > high:
                    reasons.append(f"Too Rough ({metric} {val:.2f} > {high:.2f})")
            
            elif metric in ['Homogeneity', 'Energy']:
                if val < low:
                    reasons.append(f"Not Smooth ({metric} {val:.4f} < {low:.4f})")
            
            # Correlation can go either way, so we check both bounds
            elif metric == 'Correlation':
                if val < low or val > high:
                    reasons.append(f"Abnormal Pattern ({val:.4f})")
        
        if not reasons:
            print(f"{filename}: PASS")
        else:
            print(f"{filename}: FAIL -> {', '.join(reasons)}")

if __name__ == "__main__":
    main()