"""
Individual Category Comparison Visualization
Generate separate comparison charts for each category (0-4) between training and validation sets
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import random
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class IndividualCategoryAnalyzer:
    def __init__(self, data_dir):
        """
        Initialize the analyzer with data directory
        
        Args:
            data_dir: Path to the data directory containing train/val folders
        """
        self.data_dir = data_dir
        self.data = {}
        
    def load_data_for_category(self, category):
        """
        Load data for a specific category from both train and val sets
        
        Args:
            category: Category number (0-4)
            
        Returns:
            dict: Dictionary containing file paths for train and val sets
        """
        category_data = {'train': [], 'val': []}
        
        for dataset in ['train', 'val']:
            category_path = os.path.join(self.data_dir, dataset, str(category))
            if os.path.exists(category_path):
                image_files = []
                for file in os.listdir(category_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(category_path, file))
                category_data[dataset] = image_files
                print(f"Loaded {len(image_files)} images for {dataset.upper()} - Category {category}")
        
        return category_data
    
    def extract_image_features(self, image_paths, sample_size=None):
        """
        Extract features from images
        
        Args:
            image_paths: List of image file paths
            sample_size: Number of images to sample for feature extraction (None for all images)
        
        Returns:
            Dictionary containing feature arrays
        """
        if sample_size is not None and len(image_paths) > sample_size:
            image_paths = random.sample(image_paths, sample_size)
        
        features = {
            'brightness': [],
            'contrast': [],
            'image_variance': [],
            'clarity': []
        }
        
        total_images = len(image_paths)
        print(f"Processing {total_images} images...")
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  Progress: {i + 1}/{total_images} ({(i + 1)/total_images*100:.1f}%)")
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Calculate brightness (average of RGB values)
                brightness = np.mean(img_array)
                features['brightness'].append(brightness)
                
                # Calculate contrast (standard deviation of pixel values)
                contrast = np.std(img_array)
                features['contrast'].append(contrast)
                
                # Calculate image variance (overall pixel value variance)
                image_variance = np.var(img_array)
                features['image_variance'].append(image_variance)
                
                # Calculate clarity using gradient magnitude
                gray = np.mean(img_array, axis=2)  # Convert to grayscale
                # Calculate gradients using Sobel operators
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                
                grad_x = ndimage.convolve(gray, sobel_x)
                grad_y = ndimage.convolve(gray, sobel_y)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                clarity = np.mean(gradient_magnitude)
                features['clarity'].append(clarity)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return features
    
    def create_category_comparison(self, category, sample_size=None):
        """
        Create comparison visualization for a specific category
        
        Args:
            category: Category number (0-4)
            sample_size: Number of images to sample for analysis (None for all images)
        """
        print(f"\nCreating comparison for Category {category}...")
        
        # Load data for this category
        category_data = self.load_data_for_category(category)
        
        # Extract features for both train and val sets
        train_features = self.extract_image_features(category_data['train'], sample_size)
        val_features = self.extract_image_features(category_data['val'], sample_size)
        
        # Prepare data for plotting
        plot_data = []
        
        feature_names = ['brightness', 'contrast', 'image_variance', 'clarity']
        feature_labels = ['Brightness', 'Contrast', 'Image Variance', 'Clarity']
        feature_units = ['(0-255)', '(Std Dev)', '(Variance)', '(Gradient)']
        
        for i, feature in enumerate(feature_names):
            # Add train data
            for value in train_features[feature]:
                plot_data.append({
                    'Feature': f"{feature_labels[i]} {feature_units[i]}",
                    'Value': value,
                    'Dataset': 'Train',
                    'Category': f'Category {category}'
                })
            
            # Add validation data
            for value in val_features[feature]:
                plot_data.append({
                    'Feature': f"{feature_labels[i]} {feature_units[i]}",
                    'Value': value,
                    'Dataset': 'Validation',
                    'Category': f'Category {category}'
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create the plot
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature Comparison: Category {category} (Train vs Validation)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Color scheme
        colors = ['#3498db', '#e74c3c']  # Blue for Train, Red for Validation
        
        for i, feature in enumerate(feature_names):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Filter data for current feature
            feature_data = df[df['Feature'] == f"{feature_labels[i]} {feature_units[i]}"]
            
            # Create violin plot
            sns.violinplot(data=feature_data, x='Dataset', y='Value', 
                          palette=colors, ax=ax, inner='box')
            
            # Customize the subplot
            ax.set_title(f'{feature_labels[i]} {feature_units[i]}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistical information
            train_values = feature_data[feature_data['Dataset'] == 'Train']['Value']
            val_values = feature_data[feature_data['Dataset'] == 'Validation']['Value']
            
            if len(train_values) > 0 and len(val_values) > 0:
                train_mean = np.mean(train_values)
                val_mean = np.mean(val_values)
                train_std = np.std(train_values)
                val_std = np.std(val_values)
                
                # Add text box with statistics
                stats_text = f'Train: μ={train_mean:.1f}, σ={train_std:.1f}\nVal: μ={val_mean:.1f}, σ={val_std:.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_filename = f'category_{category}_comparison.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Comparison plot for Category {category} saved as: {output_filename}")
        
        return output_filename

def main():
    """
    Main function to generate individual category comparisons
    """
    # Set data directory
    data_dir = r'c:\Users\mjynj\Desktop\vis_recognize\img_data\data'
    
    # Initialize analyzer
    analyzer = IndividualCategoryAnalyzer(data_dir)
    
    # Generate comparison for category 4 only
    generated_files = []
    
    for category in [4]:
        try:
            filename = analyzer.create_category_comparison(category)
            generated_files.append(filename)
        except Exception as e:
            print(f"Error generating comparison for Category {category}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully generated {len(generated_files)} comparison chart for category 4:")
    for filename in generated_files:
        print(f"  - {filename}")

if __name__ == "__main__":
    main()