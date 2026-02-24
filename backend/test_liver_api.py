"""
Test the liver disease prediction API endpoint
"""

import requests
import json

print("="*60)
print("TESTING LIVER DISEASE PREDICTION API")
print("="*60)

# Test data (from dataset - first row, Female patient with liver disease)
test_data = {
    "Age": 65,
    "Gender": "0",  # Female
    "Total_Bilirubin": 0.7,
    "Direct_Bilirubin": 0.1,
    "Alkaline_Phosphotase": 187,
    "Alamine_Aminotransferase": 16,
    "Aspartate_Aminotransferase": 18,
    "Total_Protiens": 6.8,
    "Albumin": 3.3,
    "Albumin_and_Globulin_Ratio": 0.9
}

print("\nTest Patient Data:")
for key, value in test_data.items():
    print(f"  {key}: {value}")

try:
    # Make API request
    url = "http://localhost:5000/api/predict/liver"
    print(f"\nSending POST request to: {url}")
    
    response = requests.post(url, json=test_data)
    
    print(f"\nResponse Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n{'='*60}")
        print("API RESPONSE:")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"\n{'='*60}")
            print("PREDICTION SUMMARY:")
            print(f"{'='*60}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Model Accuracy: {result['accuracy']:.2f}%")
            print(f"\nProbabilities:")
            print(f"  No Disease: {result['probability']['negative']:.2f}%")
            print(f"  Liver Disease: {result['probability']['positive']:.2f}%")
            print(f"\n{'='*60}")
            print("API TEST SUCCESSFUL!")
            print(f"{'='*60}")
        else:
            print(f"\nAPI returned error: {result.get('error')}")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to API server")
    print("Please make sure the backend API server is running:")
    print("  python api_server.py")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
