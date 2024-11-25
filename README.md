# Monet Style Transfer Using GANs

## Project Overview
This project uses Generative Adversarial Networks (GANs) to transform real-world photos into paintings in the style of Claude Monet. The project is part of a Kaggle competition that evaluates the quality of generated images based on the MiFID (Memorization-informed Fréchet Inception Distance) metric.

## Objectives
- Train a GAN to generate Monet-style paintings from real-world photos.
- Generate 7,000–10,000 Monet-style images in JPG format with dimensions 256x256x3.
- Submit a compressed file (`images.zip`) containing the generated images for evaluation on Kaggle.

## Dataset Description
The dataset provided by Kaggle consists of:
- **monet_jpg**: 300 Monet paintings in JPG format.
- **photo_jpg**: 7,028 real-world photos in JPG format.
- Both types of images have a resolution of 256x256.

https://www.kaggle.com/competitions/gan-getting-started/data

Put the folders under the "Data" folder or adjust the path.

## Methodology
1. **Exploratory Data Analysis (EDA):**
   - Visualized Monet paintings and real-world photos.
   - Identified differences in style and structure.

2. **Preprocessing:**
   - Resized all images to 256x256.
   - Normalized pixel values to the range [-1, 1] for GAN input.

3. **Model Architecture:**
   - **CycleGAN**:
     - Generator: Converts photos to Monet-style paintings.
     - Discriminator: Distinguishes real Monet paintings from generated images.
   - Loss Functions:
     - Adversarial loss to train the generator and discriminator.
     - Cycle-consistency loss to ensure content preservation during transformation.

4. **Training:**
   - Trained the GAN over multiple epochs.
   - Used checkpointing to save model progress.

5. **Image Generation:**
   - Transformed the photo dataset into Monet-style paintings.
   - Saved the generated images into a single zip file (`images.zip`) for Kaggle submission.
  
## Required Libraries
The project requires the following Python libraries:

```python
import os
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

## Installation and Requirements
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/monet-style-transfer.git
   cd monet-style-transfer
