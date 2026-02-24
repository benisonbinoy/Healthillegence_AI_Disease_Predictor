import json

with open('models/model_info.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("  CURRENT MODEL ACCURACY IN DATABASE")
print("=" * 60)
print(f"\nDiabetes Model:")
print(f"  - Accuracy: {data['diabetes']['accuracy']*100:.2f}%")
print(f"  - Last Updated: {data['diabetes']['last_trained']}")
print(f"  - Total Features: {len(data['diabetes']['features'])}")
print(f"  - Random Seed: {data['diabetes'].get('best_random_seed', 'N/A')}")
print("\n" + "=" * 60)
print("✅ This accuracy will be displayed on the website")
print("=" * 60)
