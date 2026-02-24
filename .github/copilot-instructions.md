You are an expert medical AI engineer helping build a pneumonia detection CNN from chest X-ray images using the Kaggle dataset located locally in backend\datasets\chest_xray.  Apply this as the website's chest prediction model that is empty now.

## Project Context
- Task: Binary classification — NORMAL vs PNEUMONIA from chest X-ray images
- Architecture: EfficientNetB3 + custom dense head (Transfer Learning)
- Framework: TensorFlow 2.10.0 / Keras (last native Windows GPU version)
- Target: ≥95% accuracy AND ≥95% precision on the test set
- Dataset split: train (~5,216 images), val (16 images — use 15% of train instead), test (624 images)
- Class imbalance: PNEUMONIA (~3,875) >> NORMAL (~1,341) — always use class weights

## Hardware — Laptop (Dell/ASUS/etc)
- CPU: Intel i7-12700H (14 cores, 20 threads)
- GPU: NVIDIA RTX 3050 Laptop (4GB VRAM) 
- OS: Windows 11
- TF Stack: TensorFlow 2.10.0 + CUDA 11.2 + cuDNN 8.1
- Python env: Conda (python=3.9)

## GPU Safety — ALWAYS Include at Top of Every Script
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("NO GPU DETECTED — fix CUDA before training!")
else:
    print(f"GPU found: {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
    )

tf.config.optimizer.set_jit(True)  # XLA — 10-20% speedup
tf.keras.mixed_precision.set_global_policy('mixed_float16')

## CONFIGDict — All Hyperparameters (never hardcode values)
CONFIG = {
    "DATA_DIR":         "chest_xray",
    "TRAIN_DIR":        "chest_xray/train",
    "TEST_DIR":         "chest_xray/test",
    "CHECKPOINT_PATH":  "best_model.h5",
    "LOG_DIR":          "logs/",
    "IMG_SIZE":         (224, 224),
    "CHANNELS":         3,
    "BATCH_SIZE":       16,          # 4GB VRAM limit — do NOT increase
    "EPOCHS_FROZEN":    10,
    "EPOCHS_FINETUNE":  30,
    "LR_PHASE1":        1e-3,
    "LR_PHASE2":        1e-5,
    "DROPOUT_RATE":     0.5,
    "L2_REG":           1e-4,
    "VAL_SPLIT":        0.15,
    "CLASSES":          ["NORMAL", "PNEUMONIA"],
    "NUM_CLASSES":      1,
    "UNFREEZE_LAYERS":  100,
}

## Model Architecture (follow strictly)
Input (224,224,3)
→ EfficientNetB3 backbone (frozen Phase 1 / top 100 layers unfrozen Phase 2)
→ GlobalAveragePooling2D
→ BatchNorm
→ Dense(512, relu, L2=1e-4)
→ BatchNorm → Dropout(0.5)
→ Dense(256, relu, L2=1e-4)
→ BatchNorm → Dropout(0.4)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(1, sigmoid, dtype='float32')   ← explicit float32 for mixed precision

## Two-Phase Training Strategy
Phase 1 — Head Only (10 epochs):
  - Freeze entire EfficientNetB3 backbone
  - LR = 1e-3, Adam optimizer
  - Only ~960K params trained
  - Expected: ~91-93% val accuracy

Phase 2 — Fine-Tune (up to 30 epochs):
  - Unfreeze top 100 EfficientNetB3 layers
  - Keep ALL BatchNormalization layers frozen (critical)
  - LR = 1e-5, recompile model
  - Expected: 95-97% val accuracy

## Callbacks (always use all four)
- ModelCheckpoint: monitor=val_auc, mode=max, save_best_only=True
- ReduceLROnPlateau: monitor=val_loss, factor=0.3, patience=4, min_lr=1e-7
- EarlyStopping: monitor=val_auc, mode=max, patience=8, restore_best_weights=True
- TensorBoard: log_dir=logs/phase{n}, histogram_freq=1

## Compile Settings
optimizer: Adam(learning_rate=LR)
loss: binary_crossentropy
metrics: [accuracy, Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]

## Augmentation Rules (clinically valid chest X-ray only)
ALLOWED:
  rotation_range=10          # patients aren't always perfectly aligned
  zoom_range=0.15            # varying distances to detector
  width_shift_range=0.1
  height_shift_range=0.1
  horizontal_flip=True       # lateral inversion is acceptable
  brightness_range=[0.8,1.2] # exposure variation
  shear_range=5
  fill_mode='reflect'
  validation_split=0.15      # use train split, NOT the tiny 16-image official val

FORBIDDEN:
  vertical_flip=True         # anatomically INVALID for chest X-rays
  large shears or heavy distortions

## Class Imbalance — Always Handle
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1], y=train_gen.classes)
class_weight_dict = dict(enumerate(class_weights))
# Pass to model.fit(class_weight=class_weight_dict)

## Threshold Optimization — Never Use Hardcoded 0.5
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
best_threshold = thresholds[np.argmax(tpr - fpr)]  # Youden's J statistic

## Data Generator Settings for RTX 3050
- Always use ImageDataGenerator (not tf.data) for this dataset size
- Always call gen.reset() before model.predict()
- color_mode='rgb' (even though X-rays are grayscale — EfficientNet expects 3 channels)
- shuffle=False for val and test generators

## VRAM Emergency Fallbacks (if OOM crash occurs)
Level 1: BATCH_SIZE = 8
Level 2: Switch to EfficientNetB0 (5.3M params, still hits 94-96%)
Level 3: IMG_SIZE = (192, 192)
Never compromise on mixed precision — keep it enabled always

## Performance Monitoring
Expected nvidia-smi readings during training:
  GPU Utilization: 70-95%     ← healthy
  VRAM Used: 2.5-3.5GB        ← safe for 4GB card
  GPU Temp: below 85°C        ← laptop thermal limit
  Power Draw: 35-50W          ← RTX 3050 laptop TDP

Expected training times on this hardware:
  Phase 1 per epoch: ~3-4 min
  Phase 2 per epoch: ~5-6 min
  Total with early stopping: ~2-2.5 hours

## Output Files (always save all)
- best_model.h5              (best checkpoint by val_auc)
- pneumonia_detector_final.h5
- pneumonia_detector_final.keras
- training_history.png       (accuracy, loss, precision, AUC curves)
- confusion_matrix.png
- roc_curve.png
- sample_predictions.png     (16 test images with confidence scores)
- gradcam.png                (Grad-CAM explainability overlay)

## Code Standards — Always Follow
1. GPU init block at the very top of every script — crash fast if no GPU
2. CONFIG dict for ALL hyperparameters — never hardcode
3. gen.reset() before every model.predict() call
4. Cast output Dense layer to float32 explicitly (mixed precision requirement)
5. Keep BatchNormalization frozen during Phase 2 fine-tuning
6. Prioritize recall over precision — missing pneumonia > false alarm
7. Include Grad-CAM for any prediction/inference function
8. Save both .h5 and .keras formats
9. Always print GPU memory usage after model.build()
10. Use seed=42 everywhere for reproducibility

## Expected Final Results
Test Accuracy:  95-97%
Test Precision: 96-98%  (PNEUMONIA class)
Test Recall:    97-99%  (sensitivity — must not miss pneumonia)
Test AUC:       0.98-0.99
Training time:  ~2-2.5 hours on RTX 3050 laptop

## Windows-Specific Notes
- Always run terminal as Administrator for CUDA access
- Power mode must be "Best Performance" (not battery saver)
- TF 2.10.0 is the LAST version with native Windows GPU support
- Do NOT upgrade to TF 2.11+ on Windows without WSL2
- conda activate pneumonia_gpu before every session