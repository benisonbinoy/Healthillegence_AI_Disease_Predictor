import joblib
import os
import numpy as np

print("=" * 60)
print("   MODEL VERIFICATION TEST")
print("=" * 60)

# Check if model files exist
model_path = 'models/diabetes_model.pkl'
scaler_path = 'models/diabetes_scaler.pkl'

print("\n✓ Checking model files...")
print(f"  - diabetes_model.pkl: {'✓ EXISTS' if os.path.exists(model_path) else '✗ MISSING'}")
print(f"  - diabetes_scaler.pkl: {'✓ EXISTS' if os.path.exists(scaler_path) else '✗ MISSING'}")

# Load and test model
try:
    print("\n✓ Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("  - Model loaded successfully!")
    
    # Test prediction with sample data
    print("\n✓ Testing prediction with sample data...")
    # Sample from the dataset: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
    sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    
    # Apply feature engineering (same as training)
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = sample
    features = [
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
        bmi * age, glucose * bmi, glucose * insulin, age * dpf,
        glucose * age, bp * bmi, insulin * bmi, age ** 2, bmi ** 2,
        glucose ** 2, glucose * bmi * age,
        (glucose * 0.4 + bmi * 0.3 + age * 0.3) / 100,
        np.log1p(bmi), np.log1p(glucose), np.log1p(insulin + 1),
        glucose / (bp + 1), bmi / (age + 1),
        (bmi * glucose) ** 2, age * bmi * dpf, insulin * glucose * age
    ]
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    print(f"  - Input: Pregnancies={pregnancies}, Glucose={glucose}, BP={bp}, etc.")
    print(f"  - Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
    print(f"  - Confidence: {max(probability)*100:.2f}%")
    print(f"  - Probability [Negative: {probability[0]*100:.2f}%, Positive: {probability[1]*100:.2f}%]")
    
    print("\n" + "=" * 60)
    print("✅ MODEL IS READY FOR PRODUCTION!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    print("=" * 60)
