# Rice Leaf Disease Classification (CNN + CBAM) | TensorFlow + Google Colab

This project builds an image classification model to detect **6 types of rice leaf conditions** (5 diseases + healthy) using a **CNN with CBAM (Convolutional Block Attention Module)**.  
The notebook is designed to run on **Google Colab** and loads the dataset directly from **Google Drive**.

---

## Classes (6)
- bacterial_leaf_blight  
- brown_spot  
- healthy  
- leaf_blast  
- leaf_scald  
- narrow_brown_spot  

---

## Project Highlights
- Loads images using `tf.keras.preprocessing.image_dataset_from_directory`
- Splits dataset into **train / validation / test** using a custom function
- Applies:
  - **Rescaling (1/255)**
  - **Data Augmentation** (random flip + rotation)
  - **CBAM attention block** after convolution layers
- Trains the model and plots:
  - Training vs Validation Accuracy
  - Training vs Validation Loss
- Makes predictions on test images with confidence
- Generates a **Confusion Matrix** and **Classification Report**

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn (confusion matrix + report)
- Google Colab + Google Drive

---

## Dataset Structure
Keep your dataset in Google Drive like this:

MyDrive/
└── train/
├── bacterial_leaf_blight/
├── brown_spot/
├── healthy/
├── leaf_blast/
├── leaf_scald/
└── narrow_brown_spot/

Each folder should contain images of that class.

---

## How to Run (Google Colab)
1. Open the notebook in **Google Colab**
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Set dataset path:
dataset_path = "/content/drive/MyDrive/train"


Run all cells in order:
Load dataset

Create train/val/test split
Apply preprocessing + augmentation
Build CNN + CBAM model
Train model
Evaluate on test set
Confusion matrix

Model Architecture
CNN Feature Extraction

3 convolution blocks:

Conv2D → BatchNorm → CBAM → MaxPooling → Dropout
CBAM Attention
CBAM helps the model focus on important disease regions rather than background noise:
Channel Attention: “what features are important”
Spatial Attention: “where to look in the image”
Classification Head
Flatten → Dense(128) → Dropout → Dense(6 softmax)

Training Settings
Image size: 256 × 256
Batch size: 32
Epochs: 50
Optimizer: Adam
Loss: Sparse Categorical Crossentropy

Metric: Accuracy
Results & Evaluation
The notebook tracks:
accuracy, loss, val_accuracy, val_loss

Visualization:
Accuracy curve (train vs val)
Loss curve (train vs val)
Predictions on sample test images with confidence
Confusion Matrix (per-class performance)

Example Output

Disease name prediction (e.g., leaf_blast)
Confidence score (e.g., 92.40%)
Confusion matrix heatmap
Future Improvements
Use transfer learning (EfficientNet / ResNet) + CBAM
Add EarlyStopping + ModelCheckpoin

Improve evaluation with k-fold cross validation
Deploy as a web app (Streamlit / Flask)
