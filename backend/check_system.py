"""
Healthiligence - System Check Script
Verifies that all dependencies and requirements are properly installed
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor}.{version.micro} is too old. Required: 3.9+")
        return False

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow',
        'keras',
        'PIL',
        'joblib',
        'flask',
        'flask_cors'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'models',
        'datasets'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"✗ Directory: {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist

def check_datasets():
    """Check if datasets are downloaded"""
    datasets = [
        ('datasets/diabetes.csv', 'Diabetes dataset'),
        ('datasets/kidney_disease.csv', 'Kidney disease dataset'),
        ('datasets/indian_liver_patient.csv', 'Liver dataset'),
        ('datasets/cell_images', 'Malaria dataset'),
        ('datasets/chest_xray', 'Pneumonia dataset')
    ]
    
    found = 0
    total = len(datasets)
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"✓ {name}")
            found += 1
        else:
            print(f"✗ {name} (not found)")
    
    return found, total

def check_models():
    """Check if models are trained"""
    models = [
        ('models/diabetes_model.pkl', 'Diabetes model'),
        ('models/kidney_model.pkl', 'Kidney model'),
        ('models/liver_model.pkl', 'Liver model'),
        ('models/malaria_model.h5', 'Malaria model'),
        ('models/pneumonia_model.h5', 'Pneumonia model')
    ]
    
    found = 0
    total = len(models)
    
    for path, name in models:
        if os.path.exists(path):
            print(f"✓ {name}")
            found += 1
        else:
            print(f"✗ {name} (not trained)")
    
    return found, total

def main():
    print("\n" + "="*60)
    print("HEALTHILIGENCE - SYSTEM CHECK")
    print("="*60 + "\n")
    
    # Check Python version
    print("Python Version:")
    python_ok = check_python_version()
    print()
    
    # Check packages
    print("Required Packages:")
    packages_ok, missing = check_packages()
    print()
    
    # Check directories
    print("Required Directories:")
    dirs_ok = check_directories()
    print()
    
    # Check datasets
    print("Datasets:")
    datasets_found, datasets_total = check_datasets()
    print()
    
    # Check models
    print("Trained Models:")
    models_found, models_total = check_models()
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    if python_ok:
        print("✓ Python version is compatible")
    else:
        print("✗ Python version is incompatible")
    
    if packages_ok:
        print("✓ All packages installed")
    else:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
    
    if dirs_ok:
        print("✓ All directories exist")
    else:
        print("✗ Some directories are missing")
    
    print(f"📊 Datasets: {datasets_found}/{datasets_total} found")
    if datasets_found < datasets_total:
        print("  See backend/DATASETS.md for download instructions")
    
    print(f"🤖 Models: {models_found}/{models_total} trained")
    if models_found < models_total:
        print("  Run: python train_models.py && python train_image_models.py")
    
    print()
    
    if python_ok and packages_ok and dirs_ok and datasets_found == datasets_total and models_found == models_total:
        print("✅ System is ready! You can start the API server.")
        print("   Run: python api_server.py")
    else:
        print("⚠️  System is not fully ready. Please complete the setup steps above.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
