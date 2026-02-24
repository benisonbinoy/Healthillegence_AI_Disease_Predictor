# Introduction

This report presents the design and implementation of a pneumonia detection module within the MedX web application. The work focuses on building a deployment-ready deep learning model for binary classification of chest X-ray images (NORMAL vs PNEUMONIA), including the end-to-end considerations required for reliable real-world use such as data splitting strategy, imbalance handling, evaluation metrics, and inference integration.

The objective is to achieve strong generalization performance on a held-out test set while prioritizing clinically safer behavior—minimizing missed pneumonia cases—within practical constraints (Windows-based development, TensorFlow/Keras stack, and limited laptop GPU memory). The following sections establish the context, justify the need, identify the core problems, and formally define the task to be solved.

## 1.1 Context and Background Study
Respiratory infections remain a major cause of morbidity and mortality worldwide, and pneumonia is among the most common and clinically significant conditions within this group. Chest radiography (CXR) is a primary imaging modality for pneumonia screening and diagnosis because it is relatively fast, inexpensive, and widely available. However, interpreting CXR images requires expertise and time, and diagnostic performance can be impacted by factors such as image quality, workload, reader fatigue, and limited access to radiologists—particularly in resource-constrained settings.

In parallel, advances in deep learning—especially convolutional neural networks (CNNs) and transfer learning—have enabled high-performing medical image classification systems. Transfer learning leverages representations learned from large-scale image datasets and adapts them to a medical domain task with fewer labeled images and reduced training time. EfficientNet-family architectures are widely used in practice due to their strong accuracy–efficiency trade-off.

This project, **MedX**, is a web-based medical AI application that integrates multiple predictive models (e.g., diabetes, kidney disease, liver disease, malaria, and pneumonia). The pneumonia component targets binary classification of chest X-ray images into **NORMAL** vs **PNEUMONIA** using a transfer-learning CNN pipeline trained on the locally stored Kaggle chest X-ray dataset under `backend/datasets/chest_xray`. The solution is intended to be deployed as part of an end-to-end system consisting of a Next.js frontend and a Python backend for inference.

## 1.2 Need Analysis
A practical pneumonia screening model within MedX is motivated by the following needs:

- **Timely triage support:** Rapid identification of likely pneumonia can help prioritize patients for further clinical evaluation, confirmatory tests, or early treatment—especially in busy emergency and outpatient environments.
- **Decision support under limited expertise:** Many care settings have limited access to radiology specialists. A reliable AI assistant can act as a second-reader tool to reduce missed findings.
- **Consistency and workload reduction:** Automated pre-screening can standardize preliminary assessments and reduce the burden on clinicians, helping mitigate variability caused by fatigue and high case volumes.
- **Integrated clinical workflow:** The application requires a model that can be executed reliably in an API-based backend, returning predictions and confidence scores suitable for a user-facing interface.
- **Safety-oriented performance requirements:** For pneumonia screening, clinical risk typically prioritizes **high sensitivity/recall** (minimizing false negatives) while maintaining strong precision to avoid excessive false alarms.

Given the project’s constraints (Windows 11 development environment, TensorFlow 2.10, and a laptop GPU with limited VRAM), the approach must also be computationally feasible while maintaining high diagnostic performance.

## 1.3 Problem Identification
Despite the availability of CXR imaging and modern ML techniques, several issues make pneumonia detection in a real application non-trivial:

- **Human interpretation challenges:** CXR findings can be subtle and overlap with other conditions. Performance depends on expertise and is affected by workload and fatigue.
- **Data limitations and imbalance:** The Kaggle chest X-ray dataset is notably imbalanced (PNEUMONIA ≫ NORMAL). Without explicit mitigation (e.g., class weights), models may become biased toward the majority class.
- **Inadequate validation split in the raw dataset:** The provided validation folder is extremely small (e.g., 16 images) and is not statistically reliable for model selection or early stopping. A more appropriate validation strategy is needed.
- **Deployment and calibration gap:** A model that appears accurate can still be poorly calibrated for decision-making. Fixed thresholds (e.g., 0.5) may not yield the best clinical trade-off; threshold selection should be optimized on validation data.
- **Operational constraints:** The inference pipeline must handle image preprocessing consistently (e.g., RGB conversion for EfficientNet input), operate under VRAM limits, and integrate with the backend API used by the frontend.

These issues collectively indicate that building a robust pneumonia module requires more than training a classifier—it requires careful data splitting, imbalance handling, training strategy design, and deployment-ready inference logic.

## 1.4 Problem Formulation
The core task is formulated as a **binary image classification** problem:

- **Input:** A chest X-ray image resized to a fixed resolution (e.g., 224×224) and represented as a 3-channel tensor suitable for an EfficientNet-based backbone.
- **Output:** A probability score \(\hat{p}\in[0,1]\) representing the model’s confidence that the image indicates **PNEUMONIA**.
- **Decision rule:** Convert \(\hat{p}\) to a class label using a threshold \(t\) chosen via validation-based optimization (e.g., maximizing Youden’s J statistic on an ROC curve), rather than assuming \(t=0.5\).

Let \(x\) be an input CXR image and \(y\in\{0,1\}\) be the ground-truth label where \(y=1\) denotes PNEUMONIA and \(y=0\) denotes NORMAL. The model \(f_\theta\) is trained to minimize binary cross-entropy:

\[
\mathcal{L}(\theta) = -\mathbb{E}\big[y\log(f_\theta(x)) + (1-y)\log(1-f_\theta(x))\big].
\]

Because the dataset is imbalanced, training incorporates **class weighting** so that errors on the minority class are penalized appropriately. Model selection targets strong generalization on the held-out test set, with performance measured by accuracy, precision, recall, and AUC. For a clinically safer screening utility, the formulation emphasizes **high recall** while maintaining high precision, and integrates the trained model into an API that can serve predictions to the MedX frontend.
