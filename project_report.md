# Alzheimer's Disease Stage Classification from Brain MRI Images using Deep Learning (EfficientNetB0 Transfer Learning)

---

## ABSTRACT

Alzheimer's disease (AD) is a progressive neurodegenerative disorder and the most common cause of dementia worldwide. Early and accurate identification of its stage — NonDemented, Very Mild Demented, Mild Demented, or Moderate Demented — is critical for timely intervention and care planning. Manual interpretation of brain Magnetic Resonance Imaging (MRI) scans is time-consuming, requires specialist radiological expertise, and is subject to inter-observer variability.

This project presents a deep learning–based system for automated classification of Alzheimer's disease stages from brain MRI scans. The system uses **transfer learning with EfficientNetB0**, pre-trained on ImageNet, as a feature extractor, with a custom classification head consisting of global average pooling, batch normalization, dropout, and dense layers. The model is trained in two phases — first with the EfficientNetB0 base frozen to train the classification head, then with the top 30 layers of the base unfrozen for fine-tuning at a lower learning rate. Class imbalance in the dataset is addressed using computed class weights. The trained model is deployed through an interactive **Streamlit** web application, which allows a user to upload an MRI scan and receive a predicted dementia stage along with class-wise confidence scores, visualized using Plotly.

The dataset used is the publicly available "Alzheimer MRI Dataset" from Kaggle, containing pre-processed axial brain MRI slices categorized into the four classes described above. The resulting system demonstrates that transfer learning is an effective and computationally efficient approach for medical image classification with limited training data, and provides a usable proof-of-concept educational tool for Alzheimer's stage screening, while explicitly being framed as **not a substitute for professional medical diagnosis**.

---

## LIST OF FIGURES

| Figure No. | Title |
|---|---|
| 1.1 | Stages of Alzheimer's Disease progression |
| 3.1 | Overall system workflow diagram |
| 3.2 | Sample MRI images from each class |
| 3.3 | EfficientNetB0-based model architecture |
| 3.4 | Two-phase training strategy (frozen vs. fine-tuned) |
| 4.1 | System architecture (frontend–backend interaction) |
| 4.2 | Streamlit application — upload interface |
| 4.3 | Streamlit application — prediction result and confidence chart |
| 5.1 | Training and validation accuracy/loss curves |
| 5.2 | Confusion matrix on the validation set |
| 5.3 | Class-wise precision, recall, and F1-score comparison |

---

# 1. INTRODUCTION

## 1.1 Background of the Study

Alzheimer's disease (AD) is a chronic, progressive neurological disorder that causes the brain to shrink (atrophy) and brain cells to die. It is the most common cause of dementia, accounting for an estimated 60–70% of all dementia cases globally. The disease progresses through distinguishable stages, ranging from no impairment to very mild, mild, and moderate-to-severe cognitive decline, each associated with characteristic structural changes in the brain that can be observed in MRI scans — such as hippocampal shrinkage and ventricular enlargement.

Magnetic Resonance Imaging (MRI) is one of the most widely used non-invasive imaging modalities for studying brain structure and is routinely used to assist in the diagnosis and staging of Alzheimer's disease. However, interpreting these scans requires trained radiologists/neurologists, and subtle structural differences between early stages can be difficult to distinguish even for experienced clinicians.

Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs), have shown strong performance in medical image analysis tasks, including tumor detection, retinal disease screening, and neurodegenerative disease classification. Transfer learning — reusing a CNN pre-trained on a large general-purpose dataset (such as ImageNet) and adapting it to a new, smaller, domain-specific dataset — has become a standard technique to achieve high accuracy on medical imaging tasks where labeled data is limited.

This project explores the application of transfer learning, specifically using the **EfficientNetB0** architecture, to classify brain MRI scans into four stages of Alzheimer's disease, and packages the resulting model into an accessible web-based diagnostic-aid tool.

## 1.2 Problem Statement

Manual classification of MRI scans for Alzheimer's disease staging is:

- **Time-consuming**, requiring detailed visual inspection by specialists.
- **Subjective**, with classification accuracy varying between different radiologists (inter-observer variability).
- **Not scalable**, especially in regions with limited access to specialist neurologists/radiologists.

There is a need for an automated, consistent, and rapid screening tool that can analyze an MRI scan and provide a preliminary classification of the Alzheimer's stage, which can then be reviewed by a medical professional. This project addresses the problem of building such a tool using deep learning and making it accessible via a simple web interface.

## 1.3 Objectives of the Study

The objectives of this project are to:

1. Acquire and prepare a labeled brain MRI dataset covering four Alzheimer's disease stages (NonDemented, Very Mild Demented, Mild Demented, Moderate Demented).
2. Design and implement a deep learning model using transfer learning (EfficientNetB0) for multi-class image classification.
3. Address class imbalance in the dataset through class-weighted training.
4. Train the model in two phases — feature extraction (frozen base) and fine-tuning (partially unfrozen base) — to balance training speed and accuracy.
5. Evaluate the model using standard classification metrics: accuracy, precision, recall, F1-score, and confusion matrix.
6. Develop an interactive web application using Streamlit that allows users to upload an MRI image and view the predicted class with confidence scores.
7. Provide clear visualizations (bar charts, confidence scores) to make the model's predictions interpretable to end users.

## 1.4 Scope and Limitations

### 1.4.1 Scope

- The system focuses exclusively on **2D axial brain MRI images** in JPG/PNG format, classified into four categories: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented.
- The model is built using **EfficientNetB0** as a fixed feature extractor with a custom classification head, trained via transfer learning on the Kaggle "Alzheimer MRI Dataset".
- The deployment is a **Streamlit-based web application** that runs locally (or can be hosted), accepting a single image upload at a time and returning a classification result with confidence scores.
- The scope covers data acquisition, preprocessing, model training/evaluation, and a functional front-end demonstration.

### 1.4.2 Limitations

- The dataset consists of **pre-processed, single 2D slices**, not full 3D MRI volumes, which may not capture the complete spatial information available in a real clinical scan.
- The model's performance is dependent on the quality and diversity of the Kaggle dataset; it may not generalize well to MRI scans from different scanners, protocols, or populations.
- The system is intended as an **educational/research prototype** and explicitly displays a medical disclaimer — it is **not validated for clinical use** and should not be used for real diagnosis.
- The current implementation processes one image at a time and does not integrate with hospital PACS/DICOM systems.
- No external test set (outside the Kaggle dataset's validation split) has been used to assess real-world generalization.

## 1.5 Significance of the Study

This study demonstrates a practical, end-to-end pipeline — from raw dataset to a deployed, interactive application — for applying deep learning to a real-world medical imaging problem. Its significance includes:

- Showing how **transfer learning** can be leveraged to achieve competitive classification performance on medical images without training a CNN from scratch, reducing computational cost and training time.
- Providing a **reproducible workflow** (data download script, training script, deployment app) that can serve as a template for similar medical image classification tasks.
- Offering an **accessible, visual interface** (via Streamlit) that bridges the gap between a trained machine learning model and an end user, making AI-assisted screening tools more approachable.
- Serving as a foundation for future research into more advanced architectures, multi-modal data (combining MRI with clinical/tabular data), and explainability techniques for medical AI.

---

# 2. LITERATURE REVIEW

## 2.1 Introduction to the Literature

The application of machine learning and deep learning to Alzheimer's disease detection and staging using neuroimaging has been an active research area for over a decade. Early approaches relied on hand-crafted features extracted from MRI scans (e.g., hippocampal volume, cortical thickness) combined with traditional classifiers such as Support Vector Machines (SVM) and Random Forests. With the advent of deep learning, Convolutional Neural Networks (CNNs) have largely replaced manual feature engineering, learning hierarchical spatial features directly from raw or minimally processed images.

This chapter reviews relevant prior work on Alzheimer's disease classification using MRI, compares existing approaches, and identifies gaps that this project aims to address.

## 2.2 Related Work

A range of approaches have been explored for Alzheimer's MRI classification:

- **Custom CNN architectures**: Several studies have designed CNNs trained from scratch on MRI datasets to classify subjects into categories such as Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN). These approaches often require large datasets and significant training time to avoid overfitting.

- **Transfer learning with pre-trained CNNs**: Architectures such as VGG16, ResNet, InceptionV3, and EfficientNet, pre-trained on ImageNet, have been fine-tuned for Alzheimer's classification tasks. These approaches generally achieve high accuracy with smaller datasets and reduced training time, since the early convolutional layers (which detect generic edges, textures, and shapes) are reused, and only the later layers are adapted to the medical imaging domain.

- **3D CNNs and volumetric analysis**: Some research uses full 3D MRI volumes (e.g., from ADNI — Alzheimer's Disease Neuroimaging Initiative) with 3D convolutions to capture volumetric atrophy patterns, which can be more informative than single 2D slices but are computationally expensive.

- **Hybrid and ensemble methods**: Combining multiple CNN architectures, or combining CNN-extracted features with traditional classifiers (SVM, Random Forest) or metaheuristic optimization techniques for hyperparameter tuning, has also been explored to improve robustness and accuracy.

- **Multi-modal approaches**: Some studies combine imaging data with clinical/demographic/genetic data (e.g., age, MMSE scores, APOE genotype) to improve classification accuracy beyond what imaging alone can achieve.

### 2.2.1 Comparison of Existing Systems

| Approach | Data Type | Typical Accuracy Range | Advantages | Disadvantages |
|---|---|---|---|---|
| Hand-crafted features + SVM/RF | 2D/3D MRI features | ~70–85% | Interpretable, low data requirement | Requires domain expertise for feature design |
| Custom CNN (trained from scratch) | 2D MRI slices | ~75–90% | Learns task-specific features | Needs large dataset, prone to overfitting on small data |
| Transfer learning (VGG/ResNet/EfficientNet) | 2D MRI slices | ~85–98% | High accuracy, faster convergence, works with limited data | Pre-trained features may not be fully optimal for grayscale medical images |
| 3D CNNs | Volumetric MRI | ~85–95% | Captures full spatial context | High computational and memory cost |
| Multi-modal (image + clinical data) | MRI + tabular | ~85–97% | Combines complementary information sources | Requires multiple aligned data sources, more complex pipeline |

The system developed in this project falls into the **transfer learning with 2D MRI slices** category, using EfficientNetB0, chosen for its strong accuracy-to-computation ratio compared to larger architectures like VGG16 or ResNet50.

## 2.3 Research Gaps

Based on the reviewed literature, the following gaps are identified:

1. Many research prototypes remain as **offline experiments** (Jupyter notebooks reporting accuracy metrics) without a deployable, user-facing interface, limiting practical demonstration and usability testing.
2. **Class imbalance** in publicly available Alzheimer's MRI datasets (e.g., far fewer "Moderate Demented" samples than "NonDemented") is often inadequately addressed, leading to biased models that favor majority classes.
3. There is limited focus on **lightweight architectures** (such as EfficientNetB0) suitable for deployment on standard hardware without requiring GPU clusters, which is important for accessibility in low-resource settings.
4. Few systems provide **transparent, interpretable outputs** (e.g., per-class confidence scores and visualizations) that communicate model uncertainty to non-expert end users.

This project addresses gaps (1), (2), and (4) by combining a transfer-learning-based EfficientNetB0 model with class-weighted training and a fully interactive Streamlit interface that visualizes per-class confidence scores.

---

# 3. PROPOSED METHODOLOGY

## 3.1 Workflow Diagram

The overall workflow of the proposed system consists of the following stages:

```
 ┌────────────────────┐
 │  Kaggle Dataset      │
 │ (Alzheimer MRI       │
 │  Dataset, 4 classes) │
 └─────────┬────────────┘
           │  download_data.py
           ▼
 ┌────────────────────┐
 │ Data Preprocessing   │
 │ - Resize to 224x224  │
 │ - Rescale (1/255)     │
 │ - Augmentation        │
 │   (rotation, flip,    │
 │    zoom, brightness)  │
 │ - Train/Val split 80/20│
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Model Building        │
 │ EfficientNetB0 (frozen)│
 │ + GAP + BatchNorm      │
 │ + Dropout + Dense(256) │
 │ + Dropout + Dense(4,   │
 │   softmax)             │
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Phase 1: Train Head    │
 │ (base frozen, Adam     │
 │ lr=1e-3, class weights,│
 │ EarlyStopping,          │
 │ ReduceLROnPlateau)      │
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Phase 2: Fine-Tune     │
 │ (unfreeze last 30       │
 │ EfficientNetB0 layers,  │
 │ Adam lr=1e-5)            │
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Evaluation              │
 │ - Classification report │
 │ - Confusion matrix       │
 │ - Training curves         │
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Save Model              │
 │ (alzheimer_model.h5)    │
 └─────────┬────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Streamlit Web App       │
 │ - Upload MRI image       │
 │ - Preprocess               │
 │ - Predict class + probs    │
 │ - Display result &          │
 │   confidence chart           │
 └────────────────────┘
```

*(Figure 3.1: Overall system workflow diagram — to be redrawn as a formal flowchart for the final report)*

## 3.2 Proposed System/Methodology

### 3.2.1 Data Collection

The dataset used is the **"Alzheimer MRI Dataset"** (also referred to as the "Alzheimer MRI Preprocessed Dataset") sourced from Kaggle (`sachinkumar413/alzheimer-mri-dataset`), downloaded programmatically using `download_data.py` via the Kaggle API. The dataset contains MRI brain images organized into four class folders:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

The `download_data.py` script:
1. Verifies the presence of a valid Kaggle API token (`kaggle.json`).
2. Downloads the dataset ZIP file via the Kaggle CLI.
3. Extracts the contents into a local `dataset/` directory.
4. Verifies that all four expected class folders are present.

### 3.2.2 Data Preprocessing

Preprocessing is performed using Keras's `ImageDataGenerator` in `train.py`:

- **Resizing**: All images are resized to **224 × 224** pixels to match the EfficientNetB0 input size.
- **Normalization**: Pixel values are rescaled from `[0, 255]` to `[0, 1]` (`rescale=1.0/255`).
- **Train/Validation Split**: An 80/20 split is applied (`validation_split=0.2`), with a fixed random seed (`seed=42`) for reproducibility.
- **Data Augmentation** (training set only), to improve generalization and reduce overfitting:
  - Random rotation up to 15°
  - Width/height shifts up to 10%
  - Horizontal flipping
  - Zoom up to 10%
  - Brightness variation between 80%–120%
- The validation set is **not augmented**, only rescaled, to provide a fair evaluation.

### 3.2.3 Model Architecture (Transfer Learning with EfficientNetB0)

The classification model is built using **transfer learning**:

- **Base model**: `EfficientNetB0`, pre-trained on ImageNet, with `include_top=False` (i.e., excluding its original 1000-class classification head) and an input shape of `(224, 224, 3)`.
- **Custom classification head**, added on top of the base model:
  1. `GlobalAveragePooling2D` — reduces the spatial feature maps to a single feature vector per channel.
  2. `BatchNormalization` — stabilizes and accelerates training.
  3. `Dropout(0.4)` — regularization to reduce overfitting.
  4. `Dense(256, activation="relu")` — fully connected layer for learning task-specific representations.
  5. `Dropout(0.3)` — additional regularization.
  6. `Dense(num_classes, activation="softmax")` — output layer producing probability distribution over the 4 classes.

This architecture leverages the rich, generic visual features learned by EfficientNetB0 on millions of natural images, adapting them to the specific task of distinguishing Alzheimer's disease stages from MRI scans.

### 3.2.4 Training Strategy (Two-Phase Training)

Training proceeds in **two phases**:

**Phase 1 — Training the Classification Head (Feature Extraction)**
- The EfficientNetB0 base is **frozen** (`base_model.trainable = False`), so only the newly added classification head layers are trained.
- Optimizer: Adam with learning rate `1e-3`.
- Loss function: Categorical cross-entropy.
- Trained for up to 10 epochs (`EPOCHS_FROZEN = 10`).

**Phase 2 — Fine-Tuning**
- The EfficientNetB0 base is **unfrozen**, but all layers except the **last 30 layers** are re-frozen, allowing only the higher-level, more task-specific layers to adapt.
- Optimizer: Adam with a much smaller learning rate `1e-5`, to avoid destroying the pre-trained weights.
- Trained for up to 10 additional epochs (`EPOCHS_FINETUNE = 10`).

**Callbacks** (applied in both phases):
- `EarlyStopping(patience=4, restore_best_weights=True)` — stops training if validation performance does not improve for 4 consecutive epochs, restoring the best-performing weights.
- `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)` — halves the learning rate if validation performance plateaus for 2 epochs.

### 3.2.5 Handling Class Imbalance

The Alzheimer MRI dataset has an uneven number of samples per class (e.g., far fewer "ModerateDemented" samples than "NonDemented"). To prevent the model from being biased toward majority classes:

- `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", ...)` is used to compute a weight for each class, inversely proportional to its frequency.
- These class weights are passed to `model.fit(..., class_weight=class_weights, ...)` in both training phases, so that misclassifications on minority classes are penalized more heavily during loss computation.

### 3.2.6 System Evaluation (Metrics)

After training, the model is evaluated on the validation split:

- **Classification Report**: precision, recall, and F1-score for each of the 4 classes, plus overall accuracy, generated using `sklearn.metrics.classification_report`.
- **Confusion Matrix**: a 4×4 matrix visualized as a heatmap (`confusion_matrix.png`) using `seaborn`, showing true vs. predicted labels for each class.
- **Training Curves**: accuracy and loss curves for both training and validation sets across both training phases, saved as `training_curves.png`.

### 3.2.7 Prediction and Recommendation

The trained model (`alzheimer_model.h5`) is loaded by the Streamlit application (`app.py`) and used for inference on user-uploaded images:

1. The uploaded image is converted to RGB, resized to 224×224, and normalized (same preprocessing as training).
2. The model outputs a probability distribution (softmax) over the 4 classes.
3. The class with the highest probability is selected as the **predicted stage**, and its probability is displayed as a **confidence percentage**.
4. A short, human-readable **description** of the predicted stage is shown (e.g., "Mild stage of Alzheimer's detected.").
5. All four class probabilities are visualized as a **color-coded bar chart** (via Plotly), and also listed as raw values in an expandable section.

This output acts as a "recommendation" in the sense of flagging a likely stage for further clinical review — it is explicitly **not** presented as a diagnosis.

### 3.2.8 Deployment

The final model is deployed as a **Streamlit web application**:

- The app checks for the existence of the trained model file (`alzheimer_model.h5`) at startup and displays an error if it is missing, instructing the user to run `train.py` first.
- The model is loaded once and cached using `@st.cache_resource` to avoid reloading on every interaction.
- The UI is structured with a title, medical disclaimer, file uploader, image preview with metadata, prediction result panel (color-coded by class), confidence bar chart, and a raw-scores expander.
- The app can be run locally via `streamlit run app.py`, and can be deployed to cloud platforms (e.g., Streamlit Community Cloud) for wider access.

## 3.3 Module Description

The system is organized into three main modules:

1. **Data Acquisition Module** (`download_data.py`)
   - Responsible for verifying Kaggle API credentials, downloading the dataset, extracting it, and validating the expected folder structure.

2. **Model Training and Evaluation Module** (`train.py`)
   - Responsible for data loading/augmentation, building the EfficientNetB0-based model, two-phase training with class balancing, evaluation (classification report, confusion matrix, training curves), and saving the trained model.

3. **Web Application / Inference Module** (`app.py`)
   - Responsible for loading the trained model, providing a user interface for image upload, preprocessing the uploaded image, running inference, and presenting results (predicted class, description, confidence scores, and visualizations) to the end user.

## 3.4 Summary

This chapter described the proposed methodology for classifying Alzheimer's disease stages from brain MRI images. The approach uses transfer learning with EfficientNetB0, a two-phase training strategy (frozen feature extraction followed by partial fine-tuning), class-weighted loss to handle class imbalance, and standard evaluation metrics. The trained model is deployed through a Streamlit web application that provides an accessible interface for end users to obtain predictions with associated confidence scores. The next chapter discusses the implementation details of each module.

---

# 4. SYSTEM IMPLEMENTATION

## 4.1 Introduction

This chapter describes the practical implementation of the proposed methodology, including the hardware and software environment used, the tools and libraries involved, and details of the backend (model training) and frontend (web application) implementations.

## 4.2 Hardware and Software Specifications

### 4.2.1 Hardware Requirements

| Component | Minimum Requirement | Recommended |
|---|---|---|
| Processor | Intel Core i5 (or equivalent) | Intel Core i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB or more |
| GPU | Not mandatory (CPU training possible but slow) | NVIDIA GPU with CUDA support (e.g., GTX 1660 / RTX series) for faster training |
| Storage | 5 GB free space (for dataset, model, dependencies) | SSD with 10+ GB free space |

### 4.2.2 Software Requirements

| Component | Specification |
|---|---|
| Operating System | Windows 10/11 (also compatible with Linux/macOS) |
| Programming Language | Python 3.9+ |
| Deep Learning Framework | TensorFlow / Keras (≥ 2.12.0) |
| Web Framework | Streamlit (≥ 1.32.0) |
| IDE | VS Code / Jupyter Notebook |
| Version Control | Git |

## 4.3 Tools / Libraries Used

| Library | Purpose |
|---|---|
| `tensorflow` / `tensorflow.keras` | Model building, training (EfficientNetB0), saving/loading |
| `numpy` | Numerical operations, array handling |
| `Pillow (PIL)` | Image loading and preprocessing |
| `scikit-learn` | Class weight computation, classification report, confusion matrix |
| `matplotlib` / `seaborn` | Plotting confusion matrix and training curves |
| `plotly` | Interactive confidence bar chart in the web app |
| `streamlit` | Web application framework for the user interface |
| `kaggle` | Programmatic dataset download from Kaggle |

## 4.4 Implementation Details

### 4.4.1 Backend Implementation

The backend consists of two scripts:

**`download_data.py`**
- Checks for a valid Kaggle API token at `~/.kaggle/kaggle.json`.
- Downloads the `sachinkumar413/alzheimer-mri-dataset` dataset as a ZIP file using the Kaggle CLI via `subprocess`.
- Extracts the ZIP into a `dataset/` directory and prints the resulting folder structure.
- Verifies that all four expected class folders (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`) exist after extraction.

**`train.py`**
- Locates the dataset root directory by walking the `dataset/` folder until the expected class subfolders are found (`find_dataset_root()`), to robustly handle nested ZIP extraction structures.
- Configures `ImageDataGenerator` instances for training (with augmentation) and validation (without augmentation), both with an 80/20 split.
- Computes class weights using `compute_class_weight(class_weight="balanced", ...)` to address class imbalance.
- Builds the EfficientNetB0-based model with a custom classification head as described in Section 3.2.3.
- Trains the model in two phases (frozen feature extraction, then fine-tuning of the last 30 layers) using Adam optimizer, categorical cross-entropy loss, `EarlyStopping`, and `ReduceLROnPlateau` callbacks.
- Saves the trained model to `alzheimer_model.h5`.
- Evaluates the model on the validation set, printing a classification report and saving a confusion matrix heatmap (`confusion_matrix.png`) and training curves (`training_curves.png`).

### 4.4.2 Frontend Implementation

The frontend is implemented in `app.py` using **Streamlit**:

- **Page configuration**: Sets the page title ("Alzheimer's MRI Classifier"), a brain emoji icon, and a centered layout.
- **Model loading**: The trained model (`alzheimer_model.h5`) is loaded once using a cached function (`@st.cache_resource`). If the model file is missing, the app displays an error message instructing the user to run `train.py`, and halts further execution (`st.stop()`).
- **Header and disclaimer**: Displays the application title, a short description of its purpose, and a prominent medical disclaimer stating the tool is for educational/research purposes only.
- **Image upload**: A file uploader accepts JPG/PNG images. Once uploaded, the image is displayed alongside its format, dimensions, and color mode.
- **Preprocessing**: The uploaded image is converted to RGB, resized to 224×224, normalized to `[0, 1]`, and expanded into a batch dimension via `preprocess_image()`.
- **Prediction**: The preprocessed image is passed to the model's `predict()` method (inside a spinner for user feedback), producing a probability for each of the 4 classes.
- **Result display**:
  - The predicted class (highest probability) and its description are shown in a color-coded panel (color determined by `CLASS_COLORS`).
  - The confidence percentage for the predicted class is displayed.
  - A Plotly bar chart shows confidence scores for all 4 classes, color-coded consistently with the result panel.
  - An expandable section shows the raw probability values for all classes.
- **Footer**: Displays a caption noting the model architecture (EfficientNetB0) and the dataset used.

## 4.5 Data Collection and Descriptions

- **Source**: Kaggle dataset `sachinkumar413/alzheimer-mri-dataset` ("Alzheimer MRI Preprocessed Dataset").
- **Format**: Pre-processed 2D grayscale/RGB axial MRI brain image slices, organized into class-labeled folders.
- **Classes** (4 total):
  - `NonDemented` — No signs of dementia.
  - `VeryMildDemented` — Very mild cognitive decline.
  - `MildDemented` — Mild stage of Alzheimer's disease.
  - `ModerateDemented` — Moderate stage of Alzheimer's disease.
- **Split**: 80% training, 20% validation, applied via `ImageDataGenerator`'s `validation_split` parameter with a fixed seed for reproducibility.
- **Input size**: All images resized to 224 × 224 × 3 (RGB) to match EfficientNetB0's expected input shape.

*(Note: exact per-class image counts should be obtained by running `download_data.py` and inspecting the extracted `dataset/` folder, then inserted here.)*

## 4.6 Summary

This chapter detailed the hardware/software environment, libraries, and implementation of both the backend (data pipeline and model training) and frontend (Streamlit web application). Together, these components form a complete pipeline from raw dataset to an interactive prediction tool. The next chapter presents the results obtained from training and evaluating the model.

---

# 5. RESULTS AND DISCUSSION

> **Note**: The values in this chapter are placeholders (`[XX]`). After running `python train.py`, replace these with the actual values printed in the console output and the generated `confusion_matrix.png` / `training_curves.png` files.

## 5.1 Experimental Results

The model was trained in two phases as described in Section 3.2.4:

- **Phase 1 (frozen base)**: Trained for up to `[10]` epochs. Final training accuracy: `[XX]%`, final validation accuracy: `[XX]%`.
- **Phase 2 (fine-tuning)**: Trained for up to `[10]` additional epochs. Final training accuracy: `[XX]%`, final validation accuracy: `[XX]%`.

Overall validation performance (after Phase 2):

| Metric | Value |
|---|---|
| Overall Accuracy | `[XX]%` |
| Macro-average Precision | `[XX]` |
| Macro-average Recall | `[XX]` |
| Macro-average F1-score | `[XX]` |

Per-class metrics (from `classification_report`):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NonDemented | `[XX]` | `[XX]` | `[XX]` | `[XX]` |
| VeryMildDemented | `[XX]` | `[XX]` | `[XX]` | `[XX]` |
| MildDemented | `[XX]` | `[XX]` | `[XX]` | `[XX]` |
| ModerateDemented | `[XX]` | `[XX]` | `[XX]` | `[XX]` |

## 5.2 Performance Comparison / Benchmarks

The EfficientNetB0-based transfer learning model can be compared against simpler baselines to demonstrate its effectiveness:

| Model | Accuracy | Notes |
|---|---|---|
| Simple CNN (3–4 conv layers, trained from scratch) | `[XX]%` | Baseline for comparison; typically lower due to limited data |
| EfficientNetB0 (frozen base only — Phase 1) | `[XX]%` | Feature extraction only |
| EfficientNetB0 (fine-tuned — Phase 1 + 2, proposed) | `[XX]%` | Final proposed model |

*(These comparison rows can be populated by additionally training a simple baseline CNN for reference, or by recording Phase 1 vs Phase 2 accuracy separately, which `train.py` already computes via `history1` and `history2`.)*

## 5.3 Analysis of Results

- The **training and validation accuracy/loss curves** (Figure 5.1, `training_curves.png`) show the learning behavior across both training phases. A converging gap between training and validation accuracy indicates good generalization; a widening gap would indicate overfitting, in which case dropout rates or augmentation strength could be increased.
- The **confusion matrix** (Figure 5.2, `confusion_matrix.png`) shows which classes are most frequently confused. Adjacent stages (e.g., `VeryMildDemented` vs `MildDemented`) are expected to show more confusion than distant stages (e.g., `NonDemented` vs `ModerateDemented`), since the underlying structural changes are more subtle between adjacent stages.
- The use of **class weights** is expected to improve recall on the minority class(es) (likely `ModerateDemented`), at a potential small cost to precision on majority classes.
- **Fine-tuning** (Phase 2) is expected to improve accuracy over Phase 1 alone, since unfreezing the last 30 layers allows the model to adapt its higher-level feature representations specifically to MRI textures, which differ significantly from the natural images in ImageNet.

## 5.4 Discussion of Key Findings

- Transfer learning with EfficientNetB0 provides a strong starting point even with a relatively small medical imaging dataset, due to the rich low/mid-level visual features (edges, textures, shapes) already learned from ImageNet.
- The two-phase training strategy balances training time and accuracy: Phase 1 is fast (only the small head is trained), while Phase 2 incurs higher computational cost but yields the larger accuracy improvement.
- The `EarlyStopping` and `ReduceLROnPlateau` callbacks help prevent overfitting and wasted training time by halting training once validation performance stagnates.
- The Streamlit interface successfully translates raw model output (a 4-element probability vector) into an interpretable result for non-technical users, via color-coding, descriptive text, and a confidence bar chart.

## 5.5 Limitations of Results

- Results are based on a **single validation split** (80/20) rather than k-fold cross-validation, so reported metrics may have some variance depending on the random seed.
- The dataset consists of **pre-processed 2D slices** from a single public source; performance on MRI scans from other scanners/hospitals/protocols (domain shift) has not been evaluated.
- No comparison against a from-scratch CNN or alternative architectures (ResNet, VGG, MobileNet) has been performed in the current implementation — only EfficientNetB0 was used.
- The model provides a single predicted class with confidence scores but does not provide visual explanations (e.g., Grad-CAM heatmaps) to indicate *which regions* of the MRI influenced the prediction.

---

# 6. CONCLUSION AND FUTURE WORK

## 6.1 Summary of the Research

This project developed an end-to-end deep learning system for classifying brain MRI images into four stages of Alzheimer's disease (NonDemented, Very Mild Demented, Mild Demented, Moderate Demented). The pipeline includes automated dataset acquisition from Kaggle, data preprocessing and augmentation, a transfer-learning-based model using EfficientNetB0 with a custom classification head, a two-phase training strategy (frozen feature extraction followed by fine-tuning), class-weighted training to handle data imbalance, standard evaluation via classification reports and confusion matrices, and deployment as an interactive Streamlit web application.

## 6.2 Conclusion Drawn from the Results

The results (Chapter 5) indicate that transfer learning with EfficientNetB0, combined with class-weighted training and a two-phase fine-tuning strategy, is an effective approach for multi-class Alzheimer's stage classification from MRI images, achieving `[XX]%` overall validation accuracy. The fine-tuning phase provides a measurable improvement over training the classification head alone, confirming the value of adapting pre-trained features to the medical imaging domain. The deployed Streamlit application successfully demonstrates the model's predictions in an accessible, interpretable format suitable for educational and research demonstration purposes.

## 6.3 Contributions of the Study

1. A **reproducible pipeline** for Alzheimer's MRI classification, from dataset download to deployed application, implemented in three clearly separated modules (`download_data.py`, `train.py`, `app.py`).
2. Application of **class-weighted, two-phase transfer learning** with EfficientNetB0 to a 4-class Alzheimer's MRI dataset, addressing both data imbalance and limited dataset size.
3. An **interactive, user-friendly web interface** that visualizes model predictions and per-class confidence scores, including appropriate medical disclaimers for responsible AI deployment.

## 6.4 Future Work

Potential directions for extending this project include:

1. **Explainability**: Integrate Grad-CAM or similar techniques to visually highlight the brain regions most influential to each prediction, increasing clinical trust and interpretability.
2. **Model comparison**: Benchmark EfficientNetB0 against other architectures (ResNet50, VGG16, MobileNetV2, Vision Transformers) and report comparative results.
3. **Cross-validation**: Use k-fold cross-validation instead of a single train/validation split to obtain more robust performance estimates.
4. **External validation**: Test the model on MRI data from different sources/scanners to assess generalization (domain shift).
5. **3D/volumetric data**: Extend the approach to full 3D MRI volumes for richer spatial information.
6. **Multi-modal integration**: Combine MRI-based predictions with tabular clinical data (e.g., the `alzheimers_disease_data.csv` available in this project) to build a multi-modal classifier.
7. **Batch processing and PACS/DICOM integration**: Extend the application to support multiple image uploads and standard medical imaging formats for more realistic clinical workflows.
8. **Deployment hardening**: Add user authentication, logging, and model versioning for a production-ready deployment.

---

# APPENDICES

## Appendix 1 – Code

Include full source listings of:
- `download_data.py`
- `train.py`
- `app.py`
- `requirements.txt`

*(Insert formatted code listings here in the final report.)*

## Appendix 2 – Screenshots

Include screenshots of:
- The Streamlit application home page (with medical disclaimer).
- The image upload interface.
- A sample prediction result for each of the 4 classes, showing the result panel and confidence bar chart.
- The confusion matrix and training curves generated after running `train.py`.

*(Capture these screenshots after running the application and insert them here.)*

---

# REFERENCES

*(To be completed with actual citations in the required format, e.g., IEEE/APA. Suggested categories of references to include:)*

1. Original papers/documentation for **EfficientNet** (Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks").
2. Papers on **transfer learning for medical image classification**.
3. The **Kaggle "Alzheimer MRI Dataset"** page (`sachinkumar413/alzheimer-mri-dataset`) — dataset citation.
4. **TensorFlow/Keras** official documentation.
5. **Streamlit** official documentation.
6. **scikit-learn** documentation for `class_weight`, `classification_report`, `confusion_matrix`.
7. General review papers on **deep learning for Alzheimer's disease diagnosis using MRI**.
