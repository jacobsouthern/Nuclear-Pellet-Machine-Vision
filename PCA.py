import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def load_images_from_folder(folder):
    """
    Loads all images from a folder and flattens them into 1D arrays.
    Returns: (images_matrix, filenames)
    """
    images = []
    filenames = []
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    
    print(f"Loading {len(files)} images from {folder}...")
    
    for file_path in files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Flatten the 300x300 image into a single row of 90,000 numbers
            flat_img = img.flatten() 
            images.append(flat_img)
            filenames.append(os.path.basename(file_path))
            
    return np.array(images), filenames

def main():
    # --- CONFIGURATION ---
    TRAIN_FOLDER = 'processed_images/train_good'  # Only PERFECT pellets
    TEST_FOLDER = 'processed_images/test_mixed'   # 3D printed pellets (Defects & Normal)
    
    # PCA Components: You found 15 gave 0.86 variance, which is good!
    N_COMPONENTS = 15 
    
    # 1. Load Data
    X_train, train_names = load_images_from_folder(TRAIN_FOLDER)
    X_test, test_names = load_images_from_folder(TEST_FOLDER)
    
    if len(X_train) == 0:
        print("Error: No images found in train folder!")
        return

    # 2. Normalize Data (Scale pixel values to 0-1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 3. Dimensionality Reduction (PCA)
    print(f"Reducing dimensions with PCA (keeping {N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS)
    
    # Fit PCA on training data only, then transform both
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2f}")

    # 4. Train Gaussian Mixture Model (GMM)
    print("Training GMM on 'Good' data...")
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(X_train_pca)

    # 5. Score the Samples
    train_scores = gmm.score_samples(X_train_pca)
    test_scores = gmm.score_samples(X_test_pca)

    # 6. Calculate Dynamic Threshold (Statistical Approach)
    # We use the Mean minus 3 Standard Deviations (Sigma).
    # This covers 99.7% of the normal distribution.
    train_mean = np.mean(train_scores)
    train_std = np.std(train_scores)
    
    suggested_threshold = train_mean - (1 * train_std)
    
    # Safety Check: Ensure threshold isn't higher than the worst training sample
    # (This prevents a good training sample from failing its own test)
    min_train_score = np.min(train_scores)
    if suggested_threshold > min_train_score:
        suggested_threshold = min_train_score - 5 # Give a tiny buffer below the worst good sample

    print("\n--- RESULTS ---")
    print(f"Training Data Stats -> Mean: {train_mean:.2f} | Std Dev: {train_std:.2f}")
    print(f"Suggested Threshold: {suggested_threshold:.2f} (Calculated via 3-Sigma Rule)")
    
    print("\nTest Image Scores:")
    for name, score in zip(test_names, test_scores):
        status = "PASS" if score > suggested_threshold else "DEFECT DETECTED"
        print(f"{name}: {score:.2f} -> {status}")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(train_scores, bins=10, alpha=0.7, label='Training (Good)', color='green')
    plt.hist(test_scores, bins=10, alpha=0.7, label='Test (Mixed)', color='red')
    plt.axvline(suggested_threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('Log-Likelihood Score')
    plt.ylabel('Count')
    plt.title('GMM Anomaly Detection Results')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()