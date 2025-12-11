import cv2
import numpy as np
import os
import glob
import pandas as pd

def calculate_metrics(img):
    """
    Calculates geometric features including roughness and defect depth.
    """
    # Ensure binary (0 or 255)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find External Contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Assume the largest contour is the pellet
    c = max(contours, key=cv2.contourArea)
    
    # 1. Area & Perimeter
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0: return None
    
    # 2. Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # 3. Convex Hull & Solidity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return None
    solidity = area / hull_area
    
    # --- NEW METRICS FOR NOISE/TEXTURE ---
    
    # 4. Edge Roughness (Hull Perimeter vs Actual Perimeter)
    # A smooth circle has a ratio near 1.0. 
    # A jagged/noisy shape has a lower ratio because the actual perimeter is longer.
    hull_perimeter = cv2.arcLength(hull, True)
    roughness = hull_perimeter / perimeter
    
    # 5. Max Defect Depth (How deep are the dips?)
    # We need hull indices (returnPoints=False) to calculate defects
    hull_indices = cv2.convexHull(c, returnPoints=False)
    
    max_defect_depth = 0
    if len(hull_indices) > 3: # Need at least 3 points to define defects
        try:
            defects = cv2.convexityDefects(c, hull_indices)
            if defects is not None:
                # defects structure: [start_index, end_index, farthest_pt_index, fixpt_depth]
                # Depth is the last value (index 3). It is scaled by 256.
                depths = defects[:, 0, 3] / 256.0 # Normalize to pixels
                max_defect_depth = np.max(depths)
        except Exception:
            pass # Fallback if geometry is degenerate

    return {
        'Area': area,
        'Circularity': circularity,
        'Solidity': solidity,
        'Roughness': roughness,         # Sensitive to jagged edges
        'Max_Defect': max_defect_depth  # Sensitive to chips or deep texture grooves
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
            m = calculate_metrics(img)
            if m:
                metrics_list.append(m)
                filenames.append(os.path.basename(f))
    
    return pd.DataFrame(metrics_list, index=filenames)

def main():
    TRAIN_FOLDER = 'processed_images/train_good'
    TEST_FOLDER = 'processed_images/test_mixed'
    
    # --- 1. TRAIN (Learn Statistics) ---
    df_train = analyze_folder(TRAIN_FOLDER)
    
    if df_train.empty:
        print("Error: No valid training data found.")
        return

    # Calculate Mean and Std Dev for each metric
    stats = df_train.describe().loc[['mean', 'std']]
    print("\n--- TRAINING STATS (Normal Ranges) ---")
    print(stats)

    # Define Tolerance (Sigma Rule)
    # 3.0 Sigma = Loose (Passes 99.7% of normals)
    SIGMA = 3.0 
    
    limits = {}
    for col in df_train.columns:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]
        
        # Calculate upper and lower bounds
        limits[col] = (mean - (SIGMA * std), mean + (SIGMA * std))
        
    print("\n--- DEFINED THRESHOLDS ---")
    for k, v in limits.items():
        print(f"{k}: {v[0]:.4f} to {v[1]:.4f}")

    # --- 2. TEST (Detect Defects) ---
    df_test = analyze_folder(TEST_FOLDER)
    
    print("\n--- TEST RESULTS ---")
    for filename, row in df_test.iterrows():
        reasons = []
        
        # Check each metric against limits
        for metric, (low, high) in limits.items():
            val = row[metric]
            
            # Special logic for defects: We only care if they are too BIG (High)
            # A defect of size 0 is perfect, so we don't check the lower bound.
            if metric == 'Max_Defect':
                if val > high:
                    reasons.append(f"Deep Defect/Chip ({val:.2f} > {high:.2f})")
                continue

            # Standard checks for other metrics
            if val < low:
                reasons.append(f"Low {metric} ({val:.4f} < {low:.4f})")
            elif val > high:
                reasons.append(f"High {metric} ({val:.4f} > {high:.4f})")
        
        if not reasons:
            print(f"{filename}: PASS")
        else:
            print(f"{filename}: FAIL -> {', '.join(reasons)}")

if __name__ == "__main__":
    main()