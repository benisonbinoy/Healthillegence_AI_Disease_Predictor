"""
Healthiligence - Prediction API Server
Flask API for serving predictions from trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import json
import base64
import numpy as np
import joblib
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Model paths
MODEL_DIR = 'models'
# ── Global model cache (loaded once, reused across requests) ──────────────────
_model_cache: dict = {}

def get_keras_model(name: str, model_path: str):
    """Load a Keras model once and cache it in memory."""
    if name not in _model_cache:
        print(f'[cache] Loading {name} from {model_path} ...')
        _model_cache[name] = keras.models.load_model(model_path, compile=False)
        print(f'[cache] {name} loaded OK')
    return _model_cache[name]
# Load model info
def load_model_info():
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Healthiligence API is running'})

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about all models"""
    try:
        info = load_model_info()
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """Predict diabetes"""
    try:
        data = request.get_json()
        
        # Load model and scaler
        model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'))
        
        # Prepare input with original features
        pregnancies = float(data['Pregnancies'])
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        skin_thickness = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        dpf = float(data['DiabetesPedigreeFunction'])
        age = float(data['Age'])
        
        # Create feature vector with all engineered features (exact same order as training)
        features = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            bmi * age,  # BMI_Age
            glucose * bmi,  # Glucose_BMI
            glucose * insulin,  # Glucose_Insulin
            age * dpf,  # Age_DPF
            glucose * age,  # Glucose_Age
            blood_pressure * bmi,  # BP_BMI
            insulin * bmi,  # Insulin_BMI
            age ** 2,  # Age_Squared
            bmi ** 2,  # BMI_Squared
            glucose ** 2,  # Glucose_Squared
            glucose * bmi * age,  # Glucose_BMI_Age
            (glucose * 0.4 + bmi * 0.3 + age * 0.3) / 100,  # Risk_Score
            np.log1p(bmi),  # BMI_Log
            np.log1p(glucose),  # Glucose_Log
            np.log1p(insulin + 1),  # Insulin_Log
            glucose / (blood_pressure + 1),  # Glucose_BP_Ratio
            bmi / (age + 1),  # BMI_Age_Ratio
            (bmi * glucose) ** 2,  # BMI_Glucose_Squared
            age * bmi * dpf,  # Age_BMI_DPF
            insulin * glucose * age  # Insulin_Glucose_Age
        ]
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get model accuracy
        info = load_model_info()
        accuracy = info.get('diabetes', {}).get('accuracy', 0)
        
        result = {
            'success': True,
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(probability)) * 100,
            'probability': {
                'negative': float(probability[0]) * 100,
                'positive': float(probability[1]) * 100
            },
            'accuracy': float(accuracy) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/kidney', methods=['POST'])
def predict_kidney():
    """Predict kidney disease"""
    try:
        data = request.get_json()
        
        # Load model, scaler, encoders, and imputer
        model = joblib.load(os.path.join(MODEL_DIR, 'kidney_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'kidney_scaler.pkl'))
        encoders_data = joblib.load(os.path.join(MODEL_DIR, 'kidney_encoders.pkl'))
        num_imputer = joblib.load(os.path.join(MODEL_DIR, 'kidney_imputer.pkl'))
        
        label_encoders = encoders_data['label_encoders']
        
        # Base features that need to be provided by user (in order)
        base_features_order = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        
        # Numerical columns
        numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
        
        # Extract and prepare base features
        import pandas as pd
        import numpy as np
        
        # Create a dictionary with the base features
        feature_dict = {}
        for feature in base_features_order:
            value = data.get(feature, '')
            
            # Handle empty values
            if value == '' or value is None:
                if feature in numerical_cols:
                    feature_dict[feature] = np.nan  # Will be imputed
                else:
                    # Use most common value for categorical
                    if feature == 'rbc':
                        value = 'normal'
                    elif feature == 'pc':
                        value = 'normal'
                    elif feature == 'pcc':
                        value = 'notpresent'
                    elif feature == 'ba':
                        value = 'notpresent'
                    elif feature in ['htn', 'dm', 'cad', 'pe', 'ane']:
                        value = 'no'
                    elif feature == 'appet':
                        value = 'good'
                    else:
                        value = '0'
                    feature_dict[feature] = value
            else:
                feature_dict[feature] = value
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Impute numerical features
        df[numerical_cols] = num_imputer.transform(df[numerical_cols])
        
        # Encode categorical features
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # If unseen category, use most frequent class (0)
                    df[col] = 0
        
        # Feature engineering (same as training)
        if 'age' in df.columns and 'bp' in df.columns:
            df['age_bp_ratio'] = df['age'] / (df['bp'] + 1)
            
        if 'bu' in df.columns and 'sc' in df.columns:
            df['bu_sc_ratio'] = df['bu'] / (df['sc'] + 1)
            df['kidney_function_score'] = df['bu'] * df['sc']
            
        if 'hemo' in df.columns and 'pcv' in df.columns:
            df['hemo_pcv_ratio'] = df['hemo'] / (df['pcv'] + 1)
            
        if 'sod' in df.columns and 'pot' in df.columns:
            df['electrolyte_balance'] = df['sod'] / (df['pot'] + 1)
            
        if 'wc' in df.columns and 'rc' in df.columns:
            df['wc_rc_ratio'] = df['wc'] / (df['rc'] + 1)
            
        if 'bgr' in df.columns:
            df['bgr_squared'] = df['bgr'] ** 2
            df['bgr_log'] = np.log1p(df['bgr'])
            
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['age_log'] = np.log1p(df['age'])
            
        if 'bu' in df.columns:
            df['bu_squared'] = df['bu'] ** 2
            df['bu_log'] = np.log1p(df['bu'])
            
        if 'sc' in df.columns:
            df['sc_squared'] = df['sc'] ** 2
            df['sc_log'] = np.log1p(df['sc'])
        
        # Risk score
        risk_factors = []
        if 'bu' in df.columns:
            risk_factors.append(df['bu'] * 0.3)
        if 'sc' in df.columns:
            risk_factors.append(df['sc'] * 0.3)
        if 'age' in df.columns:
            risk_factors.append(df['age'] * 0.2)
        if 'bp' in df.columns:
            risk_factors.append(df['bp'] * 0.2)
            
        if risk_factors:
            df['kidney_risk_score'] = sum(risk_factors) / 100
        
        # Get features in correct order
        feature_names = load_model_info()['kidney']['features']
        features_array = df[feature_names].values
        
        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get model accuracy
        info = load_model_info()
        accuracy = info.get('kidney', {}).get('accuracy', 0)
        
        result = {
            'success': True,
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(probability)) * 100,
            'probability': {
                'negative': float(probability[0]) * 100,
                'positive': float(probability[1]) * 100
            },
            'accuracy': float(accuracy) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/liver', methods=['POST'])
def predict_liver():
    """Predict liver disease"""
    try:
        data = request.get_json()
        
        # Load model, scaler, and feature list
        model = joblib.load(os.path.join(MODEL_DIR, 'liver_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'liver_scaler.pkl'))
        
        # Try to load selected features, fall back to all features if not available
        try:
            selected_features = joblib.load(os.path.join(MODEL_DIR, 'liver_features.pkl'))
        except:
            selected_features = None
        
        # Get base features from input
        age = float(data['Age'])
        gender = float(data['Gender'])  # 1 for Male, 0 for Female
        total_bilirubin = float(data['Total_Bilirubin'])
        direct_bilirubin = float(data['Direct_Bilirubin'])
        alkaline_phosphotase = float(data['Alkaline_Phosphotase'])
        alamine_aminotransferase = float(data['Alamine_Aminotransferase'])
        aspartate_aminotransferase = float(data['Aspartate_Aminotransferase'])
        total_protiens = float(data['Total_Protiens'])
        albumin = float(data['Albumin'])
        albumin_and_globulin_ratio = float(data['Albumin_and_Globulin_Ratio'])
        
        # Engineer ALL features (same as training)
        all_features = {
            'Age': age,
            'Gender': gender,
            'Total_Bilirubin': total_bilirubin,
            'Direct_Bilirubin': direct_bilirubin,
            'Alkaline_Phosphotase': alkaline_phosphotase,
            'Alamine_Aminotransferase': alamine_aminotransferase,
            'Aspartate_Aminotransferase': aspartate_aminotransferase,
            'Total_Protiens': total_protiens,
            'Albumin': albumin,
            'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio,
            # Bilirubin features
            'Bilirubin_Ratio': direct_bilirubin / (total_bilirubin + 1e-5),
            'Bilirubin_Product': total_bilirubin * direct_bilirubin,
            'Bilirubin_Diff': total_bilirubin - direct_bilirubin,
            # Enzyme features
            'ALT_AST_Ratio': alamine_aminotransferase / (aspartate_aminotransferase + 1e-5),
            'Enzyme_Product': alamine_aminotransferase * aspartate_aminotransferase,
            'Enzyme_Sum': alamine_aminotransferase + aspartate_aminotransferase,
            'Enzyme_Diff': abs(alamine_aminotransferase - aspartate_aminotransferase),
            # Protein features
            'Globulin': total_protiens - albumin,
            'Protein_Ratio': albumin / (total_protiens + 1e-5),
            'Protein_Product': albumin * total_protiens,
            # Alkaline Phosphotase interactions
            'ALP_Age': alkaline_phosphotase * age,
            'ALP_Bilirubin': alkaline_phosphotase * total_bilirubin,
            'ALP_AST': alkaline_phosphotase * aspartate_aminotransferase,
            # Age-related features
            'Age_Squared': age ** 2,
            'Age_Log': np.log1p(age),
            'Age_Gender': age * gender,
            # Logarithmic transformations
            'Total_Bilirubin_Log': np.log1p(total_bilirubin),
            'Direct_Bilirubin_Log': np.log1p(direct_bilirubin),
            'ALP_Log': np.log1p(alkaline_phosphotase),
            'ALT_Log': np.log1p(alamine_aminotransferase),
            'AST_Log': np.log1p(aspartate_aminotransferase),
            # Squared features
            'Bilirubin_Squared': total_bilirubin ** 2,
            'ALP_Squared': alkaline_phosphotase ** 2,
            'ALT_Squared': alamine_aminotransferase ** 2,
            'AST_Squared': aspartate_aminotransferase ** 2,
            # Liver Enzyme Score
            'Liver_Enzyme_Score': (alamine_aminotransferase * 0.4 + aspartate_aminotransferase * 0.3 + alkaline_phosphotase * 0.3) / 100,
            # Bilirubin_Enzyme_Interaction
            'Bilirubin_Enzyme_Interaction': total_bilirubin * ((alamine_aminotransferase * 0.4 + aspartate_aminotransferase * 0.3 + alkaline_phosphotase * 0.3) / 100),
            # Protein_Enzyme_Score
            'Protein_Enzyme_Score': albumin * (alamine_aminotransferase / (aspartate_aminotransferase + 1e-5)),
            # Liver Risk Score
            'Liver_Risk_Score': (total_bilirubin * 0.2 + alamine_aminotransferase * 0.002 + aspartate_aminotransferase * 0.002 + 
                                alkaline_phosphotase * 0.001 + (10 - total_protiens) * 0.5 + (5 - albumin) * 0.3),
            # Age and protein interaction
            'Age_Protein_Interaction': age * total_protiens,
            'Age_Albumin_Ratio': age / (albumin + 1e-5),
            # Gender-specific features
            'Gender_Bilirubin': gender * total_bilirubin,
            'Gender_ALT': gender * alamine_aminotransferase,
        }
        
        # If feature selection was used, extract only selected features
        if selected_features:
            features = [all_features[feat] for feat in selected_features]
        else:
            # Use all features in order
            features = [all_features[feat] for feat in sorted(all_features.keys())]
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get model accuracy
        info = load_model_info()
        accuracy = info.get('liver', {}).get('accuracy', 0)
        
        result = {
            'success': True,
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(probability)) * 100,
            'probability': {
                'negative': float(probability[0]) * 100,
                'positive': float(probability[1]) * 100
            },
            'accuracy': float(accuracy) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────
# GRAD-CAM UTILITY
# ─────────────────────────────────────────────────────────────────
def generate_gradcam_overlay(
    model,
    img_array: np.ndarray,
    original_rgb: np.ndarray,
    sub_model_name: str = 'efficientnetb0',
    target_layer: str = 'top_conv',
) -> str | None:
    """
    Three-stage chained Grad-CAM for EfficientNetB0 binary classifier.

    Stage 1 — feat_extractor : model_input  → top_conv feature maps
    Stage 2 — effnet_tail    : top_conv out → EfficientNet output
    Stage 3 — head_model     : EfficientNet output → final sigmoid score

    A tf.Variable wraps the feature maps so the GradientTape treats them as
    a differentiable node and can compute dLoss/dConvFeatures exactly.

    Returns:
        data-URI string  'data:image/png;base64,...'  or  None on failure.
    """
    try:
        effnet = model.get_layer(sub_model_name)

        # ── Collect pre-layers (before efficientnetb0) ────────────
        pre_layers, post_layers = [], []
        found_sub = False
        for layer in model.layers:
            if layer.name == sub_model_name:
                found_sub = True
                continue
            if not found_sub:
                pre_layers.append(layer)
            else:
                post_layers.append(layer)

        # ── Stage 1: model input → top_conv feature maps ──────────
        feat_extractor = keras.Model(
            inputs=effnet.input,
            outputs=effnet.get_layer('top_conv').output,
            name='_gradcam_feat',
        )

        # ── Stage 2: top_conv feature maps → EfficientNet output ──
        effnet_tail = keras.Model(
            inputs=effnet.get_layer('top_conv').output,
            outputs=effnet.output,
            name='_gradcam_tail',
        )

        # ── Stage 3: EfficientNet output → final prediction ────────
        effnet_out_shape = effnet.output_shape[1:]
        _hi = keras.Input(shape=effnet_out_shape, name='_gradcam_head_in')
        _x  = _hi
        for layer in post_layers:
            try:
                _x = layer(_x, training=False)
            except Exception:
                pass
        head_model = keras.Model(_hi, _x, name='_gradcam_head')

        # ── Forward through pre-EfficientNet layers ───────────────
        inp = tf.cast(img_array, tf.float32)
        for layer in pre_layers:
            try:
                inp = layer(inp, training=False)
            except Exception:
                pass   # skip InputLayer / augmentation no-ops

        # ── Extract conv feature maps (outside tape) ───────────────
        conv_features = feat_extractor(inp, training=False)

        # Wrap as tf.Variable — guarantees the tape tracks gradients through it
        conv_var = tf.Variable(
            tf.cast(conv_features, tf.float32), trainable=True, dtype=tf.float32
        )

        # ── Single connected forward pass inside tape ──────────────
        with tf.GradientTape() as tape:
            effnet_out = effnet_tail(conv_var, training=False)
            pred       = head_model(effnet_out, training=False)
            loss       = pred[:, 0]   # binary positive-class score

        grads = tape.gradient(loss, conv_var)   # (1, H, W, C)

        if grads is None:
            print('[Grad-CAM] gradient is None — check model layers')
            return None

        # ── Grad-CAM: channel-wise importance weighting ────────────
        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])           # (C,)
        heatmap = tf.reduce_sum(
            tf.cast(conv_var[0], tf.float32) * pooled_grads, axis=-1   # (H, W)
        ).numpy().astype(np.float32)

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # ── Resize → JET colormap → blend ─────────────────────────
        h_orig, w_orig  = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        colormap_bgr    = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb    = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        overlay         = cv2.addWeighted(original_rgb, 0.55, colormap_rgb, 0.45, 0)

        # ── Encode as PNG data-URI ─────────────────────────────────
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')

    except Exception as exc:
        import traceback
        print(f'[Grad-CAM] error: {exc}')
        traceback.print_exc()
        return None

@app.route('/api/predict/malaria', methods=['POST'])
def predict_malaria():
    """Predict malaria from blood cell image — EfficientNetB0 (224×224)"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Load model (cached — EfficientNetB0 backbone with built-in rescaling)
        model_path = os.path.join(MODEL_DIR, 'best_malaria_model.keras')
        model = get_keras_model('malaria', model_path)

        # Load Youden's-J optimal threshold
        threshold_path = os.path.join(MODEL_DIR, 'malaria_threshold.json')
        threshold = 0.5
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold = json.load(f).get('optimal_threshold', 0.5)

        # ── Image preprocessing ──────────────────────────────────────
        # EfficientNetB0 has built-in Rescaling(1/255) + Normalization layers.
        # Pass raw float32 pixels in [0, 255] – NO external normalisation.
        img = Image.open(image_file).convert('RGB')
        img_resized  = img.resize((224, 224))
        original_rgb = np.array(img_resized, dtype=np.uint8)          # for overlay
        img_array    = np.expand_dims(
            np.array(img_resized, dtype=np.float32), axis=0
        )  # (1, 224, 224, 3)  range [0, 255]

        # Predict
        # raw_prob is model output for class 1 = Uninfected (sigmoid).
        # Parasitized is class 0, so prob < threshold → Parasitized.
        raw_prob = float(model.predict(img_array, verbose=0)[0][0])
        predicted_class = 'Uninfected' if raw_prob >= threshold else 'Parasitized'
        confidence = raw_prob * 100 if predicted_class == 'Uninfected' else (1.0 - raw_prob) * 100

        # Grad-CAM
        gradcam_image = generate_gradcam_overlay(
            model, img_array, original_rgb,
            sub_model_name='efficientnetb0',
            target_layer='top_activation',
        )

        # Model metrics from model_info.json
        info         = load_model_info()
        malaria_info = info.get('malaria', {})

        result = {
            'success':       True,
            'prediction':    predicted_class,
            'confidence':    round(confidence, 2),
            'probability': {
                'uninfected':  round(raw_prob * 100, 2),
                'parasitized': round((1.0 - raw_prob) * 100, 2),
            },
            'threshold':     round(threshold, 4),
            'accuracy':      round(float(malaria_info.get('accuracy',  0)) * 100, 2),
            'precision':     round(float(malaria_info.get('precision', 0)) * 100, 2),
            'recall':        round(float(malaria_info.get('recall',    0)) * 100, 2),
            'auc':           round(float(malaria_info.get('auc',       0)) * 100, 2),
            'architecture':  malaria_info.get('architecture', 'EfficientNetB0'),
            'gradcam_image': gradcam_image,
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/pneumonia', methods=['POST'])
def predict_pneumonia():
    """Predict pneumonia from chest X-ray — EfficientNetB0 (224×224)"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Load model (cached)
        model_path = os.path.join(MODEL_DIR, 'pneumonia_final.keras')
        model = get_keras_model('pneumonia', model_path)

        threshold = 0.15
        
        print(f"[THRESHOLD] {threshold}")
        # ── Image preprocessing ──────────────────────────────────────
        img          = Image.open(image_file).convert('RGB')
        img_resized  = img.resize((224, 224))
        original_rgb = np.array(img_resized, dtype=np.uint8)

        # Model has built-in Rescaling + Normalization layers (include_preprocessing=True).
        # Pass raw [0, 255] float32 — do NOT call preprocess_input() or it double-scales.
        img_array = np.expand_dims(
            np.array(img_resized, dtype=np.float32), axis=0
        )  # (1, 224, 224, 3)  range [0, 255]

        # ── Predict ──────────────────────────────────────────────────
        raw_prob        = float(model.predict(img_array, verbose=0)[0][0])
        predicted_class = 'Pneumonia' if raw_prob >= threshold else 'Normal'
        pneumonia_pct   = round(raw_prob * 100, 2)
        normal_pct      = round((1.0 - raw_prob) * 100, 2)
        confidence      = pneumonia_pct if predicted_class == 'Pneumonia' else normal_pct

        # ── Grad-CAM ─────────────────────────────────────────────────
        gradcam_image = generate_gradcam_overlay(
            model, img_array, original_rgb,
            sub_model_name='efficientnetb0',
            target_layer='top_conv',
        )

        # ── Response ─────────────────────────────────────────────────
        info           = load_model_info()
        pneumonia_info = info.get('pneumonia', {})

        result = {
            'success':       True,
            'prediction':    predicted_class,
            'confidence':    confidence,
            'probability': {
                'normal':    normal_pct,
                'pneumonia': pneumonia_pct,
            },
            'threshold':     round(threshold, 4),
            'raw_score':     round(raw_prob, 4),
            'accuracy':      round(float(pneumonia_info.get('accuracy',  0)) * 100, 2),
            'precision':     round(float(pneumonia_info.get('precision', 0)) * 100, 2),
            'recall':        round(float(pneumonia_info.get('recall',    0)) * 100, 2),
            'auc':           round(float(pneumonia_info.get('auc',       0)) * 100, 2),
            'architecture':  pneumonia_info.get('architecture', 'EfficientNetB0'),
            'gradcam_image': gradcam_image,
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("HEALTHILIGENCE PREDICTION API")
    print("="*60)
    print("API Server starting on http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
