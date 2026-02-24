import json

# Load model info
with open('models/model_info.json', 'r') as f:
    info = json.load(f)

diabetes_info = info['diabetes']

print("=" * 60)
print("   DIABETES MODEL ACCURACY REPORT")
print("=" * 60)
print(f"\n✓ Accuracy: {diabetes_info['accuracy']*100:.2f}%")
print(f"✓ Test Accuracy: {diabetes_info['test_accuracy']*100:.2f}%")
print(f"✓ Random Seed Used: {diabetes_info['best_random_seed']}")
print(f"✓ Last Trained: {diabetes_info['last_trained']}")
print(f"✓ Total Features: {len(diabetes_info['features'])}")
print(f"\n✓ Status: {'✅ EXCEEDS TARGET (>95%)' if diabetes_info['accuracy'] >= 0.95 else '❌ BELOW TARGET'}")
print("=" * 60)
