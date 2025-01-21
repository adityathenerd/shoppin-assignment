# Fashion Reverse Image Search: Shoppin’ (Google Lens Alternative)

## Overview
This README provides a detailed description of the data preparation steps, variable usage, and processing logic found in the accompanying Jupyter Notebook. The primary focus of this script is to train and fine-tune models, process data for analysis, and create features for image similarity search.

### Major Processes:
1. **Loading Model Checkpoints**: Scripts to load pre-trained models.
2. **Finetuning ResNet50 for Image Similarity Search**: Preparing ResNet50 with CNN feature extraction.
3. **Training ResNet50**: Fine-tuning and training pipelines for feature extraction.
4. **Finetuning CLIP**: Steps to adapt CLIP for specific tasks related to image similarity.
5. **Using ViT and ResNet with ViT Embeddings**: Integrating Vision Transformers (ViT) and combining embeddings from ResNet and ViT for enhanced feature representations.

## Prerequisites
- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Pandas
- Numpy
- Pickle

## Code Details

### Imports and Libraries
```python
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import random
import pickle
```

### Functions

#### Vision Transformer (ViT)
1. **`initialize_vit_model()`**:
   - Loads a pre-trained Vision Transformer model.
   - Configures input sizes and patch embeddings.
   - Returns a model ready for fine-tuning or inference.

   ```python
   def initialize_vit_model():
       vit_model = load_vit_pretrained_model()
       vit_model.compile(optimizer='adam', loss='categorical_crossentropy')
       return vit_model
   ```

2. **`extract_vit_features(data)`**:
   - Extracts features from input images using the ViT model.
   - Outputs high-dimensional embeddings.

   ```python
   def extract_vit_features(data):
       features = vit_model.predict(data)
       return features
   ```

#### ResNet
1. **`initialize_resnet50()`**:
   - Loads a pre-trained ResNet50 model.
   - Configures the model for image feature extraction.

   ```python
   def initialize_resnet50():
       resnet_model = ResNet50(weights='imagenet', include_top=False)
       return resnet_model
   ```

2. **`extract_resnet_features(data)`**:
   - Processes input data through the ResNet50 model to generate embeddings.

   ```python
   def extract_resnet_features(data):
       features = resnet_model.predict(data)
       return features
   ```

#### CLIP
1. **`initialize_clip_model()`**:
   - Loads a pre-trained CLIP model for multimodal tasks.
   - Configures text and image encoders.

   ```python
   def initialize_clip_model():
       clip_model = load_clip_pretrained_model()
       return clip_model
   ```

2. **`generate_clip_embeddings(images, texts)`**:
   - Takes image and text inputs.
   - Produces joint embeddings for both modalities.

   ```python
   def generate_clip_embeddings(images, texts):
       image_embeddings = clip_model.encode_image(images)
       text_embeddings = clip_model.encode_text(texts)
       return image_embeddings, text_embeddings
   ```

#### ResNet with ViT Embeddings
1. **`combine_resnet_vit_embeddings(resnet_features, vit_features)`**:
   - Merges embeddings from ResNet50 and ViT models.
   - Performs feature concatenation or other fusion techniques.

   ```python
   def combine_resnet_vit_embeddings(resnet_features, vit_features):
       combined_features = np.concatenate([resnet_features, vit_features], axis=1)
       return combined_features
   ```

2. **`train_combined_model(combined_features, labels)`**:
   - Uses combined embeddings to train a downstream model.
   - Optimizes for tasks such as classification or similarity search.

   ```python
   def train_combined_model(combined_features, labels):
       model = build_model()  # Custom model architecture
       model.fit(combined_features, labels, epochs=10, batch_size=32)
       return model
   ```

## Evaluation Results

### Summary Table
| Model                   | Classification Accuracy | Precision@1 | Precision@5 | Precision@10 |
|-------------------------|-------------------------|-------------|-------------|--------------|
| ResNet                 | 0.8626                 | 0.8511      | 0.9714      | 0.9866       |
| ViT                    | 0.8933                 | 0.8769      | 0.9635      | 0.9769       |
| CLIP                   | 0.3800 (low accuracy)  | -           | -           | -            |
| ResNet + ViT Embeddings | 0.7481                 | 0.7654      | 0.9442      | 0.9769       |

### Detailed Evaluation Metrics
#### ResNet
- Classification Accuracy: **0.8626**
- Precision@K:
  - Top-1: **0.8511**
  - Top-5: **0.9714**
  - Top-10: **0.9866**

#### ViT
- Classification Accuracy: **0.8933**
- Precision@K:
  - Top-1: **0.8769**
  - Top-5: **0.9635**
  - Top-10: **0.9769**

#### CLIP
- Classification Accuracy: **0.3800** (Unusually low accuracy; similarity metrics not calculated.)

#### ResNet with ViT Embeddings
- Classification Accuracy: **0.7481**
- Precision@K:
  - Top-1: **0.7654**
  - Top-5: **0.9442**
  - Top-10: **0.9769**

## Data Processing Steps
1. **Standardization**: Data is scaled using `StandardScaler` to normalize features.
    ```python
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)
    ```
2. **Feature Extraction**: CNN layers from pre-trained models extract image features.
    ```python
    features = model.predict(image_batch)
    ```
3. **Random Sampling**: Used to create smaller subsets of the dataset for validation.
    ```python
    sampled_data = filtered_df.sample(n=100, random_state=42)
    ```

## Outputs
The outputs of this script include:
- Preprocessed datasets saved as `.pkl` files.
- Trained model checkpoints.
- Visualization plots of data distributions and feature embeddings.

## Instructions for Use
1. Ensure all required libraries are installed.
2. Place the `.pkl` files in the working directory.
3. Run the notebook sequentially to reproduce the results.
4. Modify hyperparameters in the relevant cells as needed for experimentation.

## Notes
- The script assumes the presence of labeled datasets and pre-trained model weights.
- Results may vary slightly due to random sampling or library updates.

For further clarification, refer to inline comments in the notebook.

