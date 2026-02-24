"""
Healthiligence - Dataset Download Guide

To train the models, you need to download the following datasets from Kaggle:

1. Diabetes Dataset:
   URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
   Save as: datasets/diabetes.csv

2. Kidney Disease Dataset:
   URL: https://www.kaggle.com/datasets/mansoordaku/ckdisease
   Save as: datasets/kidney_disease.csv

3. Liver Disease Dataset:
   URL: https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
   Save as: datasets/indian_liver_patient.csv

4. Malaria Dataset:
   URL: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
   Extract to: datasets/cell_images/

5. Pneumonia Dataset:
   URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   Extract to: datasets/chest_xray/

Using Kaggle CLI:
-----------------
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
3. Run the following commands:

kaggle datasets download -d uciml/pima-indians-diabetes-database -p datasets --unzip
kaggle datasets download -d mansoordaku/ckdisease -p datasets --unzip
kaggle datasets download -d uciml/indian-liver-patient-records -p datasets --unzip
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p datasets --unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p datasets --unzip

Training Models:
---------------
After downloading the datasets:
1. For numerical models: python backend/train_models.py
2. For image models: python backend/train_image_models.py
