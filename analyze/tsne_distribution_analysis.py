"""
t-SNE Distribution Analysis for Training vs Validation Sets
This script uses t-SNE to visualize the distribution differences between training and validation sets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import math

def compute_jsd_consistency(train_tsne, val_tsne, bins=50):
    """Compute Jensen-Shannon divergence between 2D t-SNE distributions.
    Returns (jsd_value_0_to_1, conclusion_str).
    """
    # Determine common bounds
    xmin = min(train_tsne[:, 0].min(), val_tsne[:, 0].min())
    xmax = max(train_tsne[:, 0].max(), val_tsne[:, 0].max())
    ymin = min(train_tsne[:, 1].min(), val_tsne[:, 1].min())
    ymax = max(train_tsne[:, 1].max(), val_tsne[:, 1].max())

    # Compute 2D histograms
    H_train, xedges, yedges = np.histogram2d(train_tsne[:, 0], train_tsne[:, 1],
                                             bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H_val, _, _ = np.histogram2d(val_tsne[:, 0], val_tsne[:, 1],
                                 bins=bins, range=[[xmin, xmax], [ymin, ymax]])

    # Normalize to probability distributions, add epsilon to avoid zeros
    eps = 1e-12
    P = H_train.astype(np.float64)
    Q = H_val.astype(np.float64)
    P = (P + eps) / (P.sum() + eps * P.size)
    Q = (Q + eps) / (Q.sum() + eps * Q.size)

    M = 0.5 * (P + Q)

    # Use log base 2 so JSD is in [0, 1]
    def kl_div(A, B):
        return np.sum(A * np.log2(A / B))

    jsd = 0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M)

    # Simple rule-of-thumb for consistency
    conclusion = "分布一致" if jsd < 0.1 else "分布不一致"

    return float(jsd), conclusion

def extract_image_features(image_path):
    """Extract comprehensive features from an image"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. Color features (mean and std for each channel)
        for channel in range(3):
            features.extend([
                np.mean(img_rgb[:, :, channel]),
                np.std(img_rgb[:, :, channel])
            ])
        
        # 2. Brightness
        brightness = np.mean(gray)
        features.append(brightness)
        
        # 3. Contrast (standard deviation)
        contrast = np.std(gray)
        features.append(contrast)
        
        # 4. Image variance
        variance = np.var(gray.astype(np.float32))
        features.append(variance)
        
        # 5. Clarity (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = np.var(laplacian)
        features.append(clarity)
        
        # 6. Histogram features (simplified)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_features = hist.flatten() / np.sum(hist)  # Normalize
        features.extend(hist_features)
        
        # 7. Texture features (using Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_x = np.mean(np.abs(sobel_x))
        texture_y = np.mean(np.abs(sobel_y))
        features.extend([texture_x, texture_y])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_dataset_features(data_dir, category, split, max_samples=1000):
    """Load features from a specific category and split"""
    split_dir = os.path.join(data_dir, split, str(category))
    
    if not os.path.exists(split_dir):
        print(f"Directory not found: {split_dir}")
        return None, []
    
    image_files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit samples for faster processing
    if len(image_files) > max_samples:
        np.random.seed(42)  # For reproducibility
        image_files = np.random.choice(image_files, max_samples, replace=False)
    
    features_list = []
    valid_files = []
    
    print(f"Processing {len(image_files)} images from {split} set, category {category}...")
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(image_files)} images...")
        
        image_path = os.path.join(split_dir, filename)
        features = extract_image_features(image_path)
        
        if features is not None:
            features_list.append(features)
            valid_files.append(filename)
    
    if features_list:
        return np.array(features_list), valid_files
    else:
        return None, []

def create_tsne_visualization(category, data_dir='./data', max_samples_per_split=1000):
    """Create t-SNE visualization for a specific category"""
    print(f"\n=== t-SNE Analysis for Category {category} ===")
    
    # Load training features
    train_features, train_files = load_dataset_features(data_dir, category, 'train', max_samples_per_split)
    if train_features is None:
        print(f"No training data found for category {category}")
        return
    
    # Load validation features
    val_features, val_files = load_dataset_features(data_dir, category, 'val', max_samples_per_split)
    if val_features is None:
        print(f"No validation data found for category {category}")
        return
    
    print(f"Loaded {len(train_features)} training samples and {len(val_features)} validation samples")
    
    # Combine features
    all_features = np.vstack([train_features, val_features])
    labels = ['Train'] * len(train_features) + ['Validation'] * len(val_features)
    
    print("Standardizing features...")
    # Standardize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    print("Applying t-SNE dimensionality reduction...")
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
    tsne_results = tsne.fit_transform(all_features_scaled)
    
    # Split results back
    train_tsne = tsne_results[:len(train_features)]
    val_tsne = tsne_results[len(train_features):]

    # Compute distribution consistency metric (Jensen-Shannon Divergence)
    jsd_value, jsd_conclusion = compute_jsd_consistency(train_tsne, val_tsne, bins=60)
    print(f"分布一致性指标 (JSD, 0~1): {jsd_value:.4f} -> {jsd_conclusion}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot training samples
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], 
               c='blue', alpha=0.6, s=20, label=f'Training (n={len(train_features)})')
    
    # Plot validation samples
    plt.scatter(val_tsne[:, 0], val_tsne[:, 1], 
               c='red', alpha=0.6, s=20, label=f'Validation (n={len(val_features)})')
    
    # plt.title(f't-SNE Visualization: Category {category} (Train vs Validation)', fontsize=16, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics
    plt.figtext(0.02, 0.02,
                f'Perplexity: 30, Iterations: 1000, Features: {all_features.shape[1]}\n'
                f'JSD(Train vs Val): {jsd_value:.4f} -> {jsd_conclusion}',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f'category_{category}_tsne_distribution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved as: {output_filename}")
    
    plt.show()
    
    return tsne_results, labels

def main():
    """Main function to run t-SNE analysis"""
    print("t-SNE Distribution Analysis for Training vs Validation Sets")
    print("=" * 60)
    
    # You can modify these parameters
    categories_to_analyze = [2, 3, 4]  # Categories to analyze
    data_directory = r'C:\Users\mjynj\Desktop\traffic\vis_recognize\img_data\data'  # Path to your data directory
    max_samples = 2000  # Maximum samples per split to speed up processing
    
    for category in categories_to_analyze:
        try:
            create_tsne_visualization(category, data_directory, max_samples)
        except Exception as e:
            print(f"Error analyzing category {category}: {e}")
            continue
    
    print("\nt-SNE analysis completed!")

if __name__ == "__main__":
    main()