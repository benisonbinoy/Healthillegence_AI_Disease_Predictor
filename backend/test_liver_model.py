"""
Test the liver disease prediction model
"""

import json
import joblib
import numpy as np

print("="*60)
print("TESTING LIVER DISEASE PREDICTION MODEL")
print("="*60)

# Load model info
with open('models/model_info.json', 'r') as f:
    info = json.load(f)

liver_info = info.get('liver', {})
print(f"\nModel Type: {liver_info.get('model_type', 'Unknown')}")
print(f"Accuracy: {liver_info.get('accuracy', 0)*100:.2f}%")
print(f"Last Trained: {liver_info.get('last_trained', 'Unknown')}")
print(f"Number of Features: {len(liver_info.get('features', []))}")

# Load model and test
try:
    model = joblib.load('models/liver_model.pkl')
    scaler = joblib.load('models/liver_scaler.pkl')
    features = joblib.load('models/liver_features.pkl')
    
    print(f"\nModel loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Features used: {len(features)}")
    
    # Test with sample data (from dataset first row)
    test_data = {
        'Age': 65,
        'Gender': 0,  # Female
        'Total_Bilirubin': 0.7,
        'Direct_Bilirubin': 0.1,
        'Alkaline_Phosphotase': 187,
        'Alamine_Aminotransferase': 16,
        'Aspartate_Aminotransferase': 18,
        'Total_Protiens': 6.8,
        'Albumin': 3.3,
        'Albumin_and_Globulin_Ratio': 0.9
    }
    
    print(f"\nTest Data (Sample Patient):")
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    # Engineer features (same as training)
    age = test_data['Age']
    gender = test_data['Gender']
    total_bilirubin = test_data['Total_Bilirubin']
    direct_bilirubin = test_data['Direct_Bilirubin']
    alkaline_phosphotase = test_data['Alkaline_Phosphotase']
    alamine_aminotransferase = test_data['Alamine_Aminotransferase']
    aspartate_aminotransferase = test_data['Aspartate_Aminotransferase']
    total_protiens = test_data['Total_Protiens']
    albumin = test_data['Albumin']
    albumin_and_globulin_ratio = test_data['Albumin_and_Globulin_Ratio']
    
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
        'Bilirubin_Ratio': direct_bilirubin / (total_bilirubin + 1e-5),
        'Bilirubin_Product': total_bilirubin * direct_bilirubin,
        'Bilirubin_Diff': total_bilirubin - direct_bilirubin,
        'ALT_AST_Ratio': alamine_aminotransferase / (aspartate_aminotransferase + 1e-5),
        'Enzyme_Product': alamine_aminotransferase * aspartate_aminotransferase,
        'Enzyme_Sum': alamine_aminotransferase + aspartate_aminotransferase,
        'Enzyme_Diff': abs(alamine_aminotransferase - aspartate_aminotransferase),
        'Globulin': total_protiens - albumin,
        'Protein_Ratio': albumin / (total_protiens + 1e-5),
        'Protein_Product': albumin * total_protiens,
        'ALP_Age': alkaline_phosphotase * age,
        'ALP_Bilirubin': alkaline_phosphotase * total_bilirubin,
        'ALP_AST': alkaline_phosphotase * aspartate_aminotransferase,
        'Age_Squared': age ** 2,
        'Age_Log': np.log1p(age),
        'Age_Gender': age * gender,
        'Total_Bilirubin_Log': np.log1p(total_bilirubin),
        'Direct_Bilirubin_Log': np.log1p(direct_bilirubin),
        'ALP_Log': np.log1p(alkaline_phosphotase),
        'ALT_Log': np.log1p(alamine_aminotransferase),
        'AST_Log': np.log1p(aspartate_aminotransferase),
        'Bilirubin_Squared': total_bilirubin ** 2,
        'ALP_Squared': alkaline_phosphotase ** 2,
        'ALT_Squared': alamine_aminotransferase ** 2,
        'AST_Squared': aspartate_aminotransferase ** 2,
        'Liver_Enzyme_Score': (alamine_aminotransferase * 0.4 + aspartate_aminotransferase * 0.3 + alkaline_phosphotase * 0.3) / 100,
        'Bilirubin_Enzyme_Interaction': total_bilirubin * ((alamine_aminotransferase * 0.4 + aspartate_aminotransferase * 0.3 + alkaline_phosphotase * 0.3) / 100),
        'Protein_Enzyme_Score': albumin * (alamine_aminotransferase / (aspartate_aminotransferase + 1e-5)),
        'Liver_Risk_Score': (total_bilirubin * 0.2 + alamine_aminotransferase * 0.002 + aspartate_aminotransferase * 0.002 + 
                            alkaline_phosphotase * 0.001 + (10 - total_protiens) * 0.5 + (5 - albumin) * 0.3),
        'Age_Protein_Interaction': age * total_protiens,
        'Age_Albumin_Ratio': age / (albumin + 1e-5),
        'Gender_Bilirubin': gender * total_bilirubin,
        'Gender_ALT': gender * alamine_aminotransferase,
    }
    
    # Select only the features used in the model
    feature_vector = [all_features[feat] for feat in features]
    
    # Scale and predict
    feature_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_scaled)[0]
    probability = model.predict_proba(feature_scaled)[0]
    
    print(f"\nPrediction Result:")
    print(f"  Prediction: {'Liver Disease' if prediction == 1 else 'No Disease'}")
    print(f"  Confidence: {max(probability)*100:.2f}%")
    print(f"  Probabilities:")
    print(f"    No Disease: {probability[0]*100:.2f}%")
    print(f"    Liver Disease: {probability[1]*100:.2f}%")
    
    print(f"\n{'='*60}")
    print("MODEL TEST SUCCESSFUL!")
    print(f"{'='*60}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
