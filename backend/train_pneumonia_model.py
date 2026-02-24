"""
Pneumonia Detection CNN — EfficientNetB3 Transfer Learning
Binary classification: NORMAL vs PNEUMONIA from chest X-ray images
Target: >=95% accuracy AND >=95% precision on held-out test set
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus}")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
)
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Training on RTX 3050")

# ─────────────────────────────────────────────────────────────────
# GPU SAFETY — must be at top of every script
# ─────────────────────────────────────────────────────────────────
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable XLA/JIT fully — ptxas.exe not available in conda cudatoolkit pkg
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=C:\\ProgramData\\miniconda3\\envs\\pneumonia_gpu\\Library\\bin'
warnings.filterwarnings('ignore')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print('WARNING: NO GPU DETECTED — training will run on CPU (slower)')
else:
    print(f'GPU found: {gpus}')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
    )

# XLA JIT disabled — ptxas.exe not in conda cudatoolkit (install full CUDA toolkit to enable)
# tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f'Compute dtype : {tf.keras.mixed_precision.global_policy().compute_dtype}')
print(f'Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}')

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score
)

# ─────────────────────────────────────────────────────────────────
# CONFIG — all hyperparameters live here, never hardcode
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    'train_dir':    r'backend\datasets\chest_xray\train',
    'test_dir':     r'backend\datasets\chest_xray\test',
    'model_dir':    r'backend\models',

    # Image
    'img_size':     (224, 224),
    'channels':     3,

    # Training — BATCH_SIZE 16 is the 4 GB VRAM limit, do NOT increase
    'batch_size':   16,
    'val_split':    0.15,           # use 15% of train as validation
    'seed':         42,

    # Phase 1 — head only
    'phase1_lr':    1e-3,
    'phase1_epochs': 10,

    # Phase 2 — fine-tune top 100 backbone layers
    'phase2_lr':    1e-5,
    'phase2_epochs': 30,
    'unfreeze_top': 100,

    # Regularisation
    'l2':           1e-4,
    'dropout1':     0.5,
    'dropout2':     0.4,
    'dropout3':     0.3,

    # Callbacks
    'lr_factor':    0.3,
    'lr_patience':  4,
    'es_patience':  8,
    'monitor':      'val_auc',
    'monitor_mode': 'max',

    # Augmentation (clinically valid chest X-ray only)
    'rotation':     10,
    'zoom':         0.15,
    'h_shift':      0.1,
    'w_shift':      0.1,
    'brightness':   [0.8, 1.2],
    'shear':        5,
}

# ─────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=CONFIG['val_split'],
    rotation_range=CONFIG['rotation'],
    zoom_range=CONFIG['zoom'],
    width_shift_range=CONFIG['w_shift'],
    height_shift_range=CONFIG['h_shift'],
    brightness_range=CONFIG['brightness'],
    shear_range=CONFIG['shear'],
    horizontal_flip=True,
    vertical_flip=False,           # FORBIDDEN for chest X-rays
    fill_mode='reflect',
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=CONFIG['val_split'],
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("\nLoading training data …")
train_gen = train_datagen.flow_from_directory(
    CONFIG['train_dir'],
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    color_mode='rgb',
    subset='training',
    seed=CONFIG['seed'],
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    CONFIG['train_dir'],
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    color_mode='rgb',
    subset='validation',
    seed=CONFIG['seed'],
    shuffle=False,
)

print("\nLoading test data …")
test_gen = test_datagen.flow_from_directory(
    CONFIG['test_dir'],
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

print(f"\nClass indices: {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
print(f"Test samples  : {test_gen.samples}")

# ──────────────────────────────────────────────
# Class weights  (PNEUMONIA >> NORMAL imbalance)
# ──────────────────────────────────────────────
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weight_dict = dict(enumerate(class_weights_array))
print(f"\nClass weights: {class_weight_dict}")

# ─────────────────────────────────────────────────────────────────
# Model — EfficientNetB3 backbone + custom head
# ─────────────────────────────────────────────────────────────────
def build_model(freeze_backbone: bool = True) -> keras.Model:
    inputs = keras.Input(shape=(*CONFIG['img_size'], CONFIG['channels']), name='input')

    backbone = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
    )
    backbone.trainable = not freeze_backbone

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    x = layers.BatchNormalization(name='bn_head_1')(x)
    x = layers.Dense(
        512, activation='relu',
        kernel_regularizer=regularizers.l2(CONFIG['l2']),
        dtype='float32', name='dense_512',
    )(x)
    x = layers.BatchNormalization(dtype='float32', name='bn_head_2')(x)
    x = layers.Dropout(CONFIG['dropout1'], name='drop_1')(x)

    x = layers.Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(CONFIG['l2']),
        dtype='float32', name='dense_256',
    )(x)
    x = layers.BatchNormalization(dtype='float32', name='bn_head_3')(x)
    x = layers.Dropout(CONFIG['dropout2'], name='drop_2')(x)

    x = layers.Dense(128, activation='relu', dtype='float32', name='dense_128')(x)
    x = layers.Dropout(CONFIG['dropout3'], name='drop_3')(x)

    # Output must be float32 — required when using mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# ─────────────────────────────────────────────────────────────────
# Callbacks factory
# ─────────────────────────────────────────────────────────────────
def make_callbacks(ckpt_path: str, phase: int) -> list:
    log_dir = os.path.join(CONFIG['model_dir'], '..', 'logs', f'phase{phase}')
    return [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=CONFIG['monitor'],
            mode=CONFIG['monitor_mode'],
            save_best_only=True,
            save_weights_only=True,   # avoids EagerTensor JSON serialization bug in TF 2.10
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CONFIG['lr_factor'],
            patience=CONFIG['lr_patience'],
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor=CONFIG['monitor'],
            mode=CONFIG['monitor_mode'],
            patience=CONFIG['es_patience'],
            restore_best_weights=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
    ]


# ──────────────────────────────────────────────
# Phase 1 — train head only
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 — Training head  (backbone frozen)")
print("=" * 60)

model = build_model(freeze_backbone=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['phase1_lr']),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ],
)
model.summary(line_length=120)

# Print GPU memory after build
if gpus:
    try:
        mem = tf.config.experimental.get_memory_info('GPU:0')
        print(f'GPU memory after build — current: {mem["current"]/1e6:.1f} MB, peak: {mem["peak"]/1e6:.1f} MB')
    except Exception:
        pass

ckpt_p1 = os.path.join(CONFIG['model_dir'], 'pneumonia_phase1_best.h5')

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=CONFIG['phase1_epochs'],
    class_weight=class_weight_dict,
    callbacks=make_callbacks(ckpt_p1, phase=1),
    verbose=1,
)

# ──────────────────────────────────────────────
# Phase 2 — fine-tune top 100 backbone layers
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 — Fine-tuning top-100 backbone layers")
print("=" * 60)

# Load best Phase-1 weights
model.load_weights(ckpt_p1)

# When EfficientNetB3 is built with input_tensor=inputs in the Functional API,
# its layers are flattened directly into model.layers — there is no sub-model
# named 'efficientnetb3'. We identify backbone vs. head layers by their names:
# head layers have names starting with one of these prefixes.
HEAD_NAMES = {'gap', 'bn_head', 'dense_', 'drop_', 'output'}

def is_head_layer(layer):
    return any(layer.name.startswith(p) for p in HEAD_NAMES)

# First freeze everything
for layer in model.layers:
    layer.trainable = False

# Collect backbone layers only (i.e. not the head)
backbone_layers = [l for l in model.layers if not is_head_layer(l)]

# Unfreeze only the top `unfreeze_top` backbone layers, but keep BN frozen
frozen_until = max(0, len(backbone_layers) - CONFIG['unfreeze_top'])
for i, layer in enumerate(backbone_layers):
    if i >= frozen_until and not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

trainable_count = sum(1 for l in model.layers if l.trainable)
bn_frozen = sum(1 for l in model.layers if isinstance(l, layers.BatchNormalization) and not l.trainable)
print(f"Trainable layers : {trainable_count} / {len(model.layers)}")
print(f"BatchNorm frozen : {bn_frozen}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['phase2_lr']),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ],
)

ckpt_p2 = os.path.join(CONFIG['model_dir'], 'best_model.h5')

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=CONFIG['phase2_epochs'],
    class_weight=class_weight_dict,
    callbacks=make_callbacks(ckpt_p2, phase=2),
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────
# Save final model in all required formats
# ─────────────────────────────────────────────────────────────────
print("\nLoading best Phase-2 checkpoint …")
model.load_weights(ckpt_p2)

final_keras = os.path.join(CONFIG['model_dir'], 'pneumonia_detector_final.keras')
final_h5    = os.path.join(CONFIG['model_dir'], 'pneumonia_detector_final.h5')
pneumonia_h5 = os.path.join(CONFIG['model_dir'], 'pneumonia_model.h5')

model.save(final_keras)
print(f'Saved → {final_keras}')
model.save(final_h5)
print(f'Saved → {final_h5}')
model.save(pneumonia_h5)
print(f'Saved → {pneumonia_h5}')

# ──────────────────────────────────────────────
# Youden's J threshold on ROC curve
# ──────────────────────────────────────────────
print("\nComputing optimal decision threshold (Youden's J) …")
test_gen.reset()
y_true = test_gen.classes
y_prob = model.predict(test_gen, verbose=1).ravel()

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
j_stat = tpr - fpr
optimal_idx = np.argmax(j_stat)
optimal_threshold = float(thresholds[optimal_idx])
print(f"ROC AUC : {roc_auc:.4f}")
print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")

# ──────────────────────────────────────────────
# Evaluate at optimal threshold
# ──────────────────────────────────────────────
y_pred = (y_prob >= optimal_threshold).astype(int)

test_acc   = float(np.mean(y_pred == y_true))
test_prec  = float(precision_score(y_true, y_pred, zero_division=0))
test_rec   = float(recall_score(y_true, y_pred, zero_division=0))

print(f"\n{'=' * 60}")
print(f"TEST SET RESULTS  (threshold = {optimal_threshold:.4f})")
print(f"  Accuracy  : {test_acc  * 100:.2f}%")
print(f"  Precision : {test_prec * 100:.2f}%")
print(f"  Recall    : {test_rec  * 100:.2f}%")
print(f"  AUC       : {roc_auc  * 100:.2f}%")
print(f"{'=' * 60}")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# ─────────────────────────────────────────────────────────────────
# Merge training histories
# ─────────────────────────────────────────────────────────────────
def merge_histories(h1, h2):
    out = {}
    for key in h1.history:
        out[key] = h1.history[key] + h2.history[key]
    return out

history = merge_histories(history1, history2)

# ─────────────────────────────────────────────────────────────────
# Training history plot
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pneumonia Detector — EfficientNetB3 Training History', fontsize=14)
phase_split = CONFIG['phase1_epochs']
epochs_range = range(1, len(history['accuracy']) + 1)

for ax, metric, title in [
    (axes[0, 0], 'accuracy',  'Accuracy'),
    (axes[0, 1], 'auc',       'AUC'),
    (axes[1, 0], 'precision', 'Precision'),
    (axes[1, 1], 'recall',    'Recall'),
]:
    ax.plot(epochs_range, history[metric],     label='Train', linewidth=2)
    ax.plot(epochs_range, history[f'val_{metric}'], label='Val', linewidth=2)
    ax.axvline(phase_split + 0.5, color='red', linestyle='--', alpha=0.5, label='Phase split')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
hist_path = os.path.join(CONFIG['model_dir'], 'training_history.png')
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nTraining history → {hist_path}")

# ─────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im)
classes = ['NORMAL', 'PNEUMONIA']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
ax.set_title(f'Confusion Matrix  (threshold={optimal_threshold:.3f})')
thresh_cm = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh_cm else 'black',
                fontsize=16, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(CONFIG['model_dir'], 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Confusion matrix  → {cm_path}')

# ─────────────────────────────────────────────────────────────────
# ROC curve
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], '--', color='grey', label='Random')
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
           label=f'Youden J  (th={optimal_threshold:.3f})')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(CONFIG['model_dir'], 'roc_curve.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'ROC curve         → {roc_path}')

# ─────────────────────────────────────────────────────────────────
# Sample predictions (16 test images with confidence scores)
# ─────────────────────────────────────────────────────────────────
test_gen.reset()
batch_imgs, batch_labels = next(test_gen)
batch_probs = model.predict(batch_imgs, verbose=0).ravel()
batch_preds = (batch_probs >= optimal_threshold).astype(int)

n_show = min(16, len(batch_imgs))
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()
label_names = {v: k for k, v in train_gen.class_indices.items()}
for idx in range(n_show):
    img = batch_imgs[idx]
    true_label = label_names[int(batch_labels[idx])]
    pred_label = label_names[int(batch_preds[idx])]
    conf = batch_probs[idx] * 100
    color = 'green' if true_label == pred_label else 'red'
    axes[idx].imshow(img)
    axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}  ({conf:.1f}%)', color=color, fontsize=9)
    axes[idx].axis('off')
for idx in range(n_show, 16):
    axes[idx].axis('off')
plt.suptitle('Sample Predictions (green=correct, red=wrong)', fontsize=13)
plt.tight_layout()
sp_path = os.path.join(CONFIG['model_dir'], 'sample_predictions.png')
plt.savefig(sp_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'Sample predictions→ {sp_path}')

# ─────────────────────────────────────────────────────────────────
# Grad-CAM visualisation (4 test images)
# ─────────────────────────────────────────────────────────────────
try:
    import cv2
    last_conv_name = 'top_activation'
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )
    test_gen.reset()
    cam_imgs, cam_labels = next(test_gen)
    n_cam = min(4, len(cam_imgs))
    fig, axes = plt.subplots(n_cam, 2, figsize=(8, 4 * n_cam))
    if n_cam == 1:
        axes = axes.reshape(1, 2)
    for idx in range(n_cam):
        img_arr = np.expand_dims(cam_imgs[idx], axis=0).astype(np.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_arr)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out_np = conv_out[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        for ch in range(pooled_grads_np.shape[-1]):
            conv_out_np[:, :, ch] *= pooled_grads_np[ch]
        heatmap = np.mean(conv_out_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        h_resized = cv2.resize(heatmap, (224, 224))
        h_uint8 = np.uint8(255 * h_resized)
        h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        orig_rgb = np.uint8(cam_imgs[idx] * 255)
        overlay = cv2.addWeighted(orig_rgb, 0.6, h_color_rgb, 0.4, 0)
        axes[idx, 0].imshow(orig_rgb);  axes[idx, 0].set_title('Original'); axes[idx, 0].axis('off')
        axes[idx, 1].imshow(overlay);   axes[idx, 1].set_title('Grad-CAM');  axes[idx, 1].axis('off')
    plt.suptitle('Grad-CAM Explainability', fontsize=13)
    plt.tight_layout()
    gradcam_path = os.path.join(CONFIG['model_dir'], 'gradcam.png')
    plt.savefig(gradcam_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Grad-CAM          → {gradcam_path}')
except Exception as e:
    print(f'Grad-CAM skipped: {e}')

# ─────────────────────────────────────────────────────────────────
# Save threshold for inference
# ─────────────────────────────────────────────────────────────────
threshold_path = os.path.join(CONFIG['model_dir'], 'pneumonia_threshold.json')
with open(threshold_path, 'w') as f:
    json.dump({'optimal_threshold': optimal_threshold}, f, indent=2)
print(f"Threshold saved → {threshold_path}")

# ─────────────────────────────────────────────────────────────────
# Update model_info.json
# ─────────────────────────────────────────────────────────────────
info_path = os.path.join(CONFIG['model_dir'], 'model_info.json')
with open(info_path, 'r') as f:
    model_info = json.load(f)

model_info['pneumonia'] = {
    'accuracy':          round(test_acc, 6),
    'precision':         round(test_prec, 6),
    'recall':            round(test_rec, 6),
    'auc':               round(roc_auc, 6),
    'test_accuracy':     round(test_acc, 6),
    'optimal_threshold': round(optimal_threshold, 6),
    'last_trained':      datetime.now().isoformat(),
    'architecture':      'EfficientNetB3 + custom dense head',
    'img_size':          list(CONFIG['img_size']),
    'train_samples':     int(train_gen.samples),
    'val_samples':       int(val_gen.samples),
    'test_samples':      int(test_gen.samples),
    'class_indices':     train_gen.class_indices,
    'phase1_epochs':     CONFIG['phase1_epochs'],
    'phase2_epochs':     CONFIG['phase2_epochs'],
    'mixed_precision':   True,
    'batch_size':        CONFIG['batch_size'],
}

with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"\nmodel_info.json updated → {info_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Test Accuracy  : {test_acc  * 100:.2f}%")
print(f"  Test Precision : {test_prec * 100:.2f}%")
print(f"  Test Recall    : {test_rec  * 100:.2f}%")
print(f"  ROC AUC        : {roc_auc  * 100:.2f}%")
print(f"  Threshold      : {optimal_threshold:.4f}")
print("=" * 60)
