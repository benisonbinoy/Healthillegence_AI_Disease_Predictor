@echo off
:: ─────────────────────────────────────────────────────────
:: Train Pneumonia Detection Model (EfficientNetB3)
:: Run from: the MedX project root directory
:: Usage: train_pneumonia.bat
:: ─────────────────────────────────────────────────────────

echo ====================================================
echo  Pneumonia Model Training — EfficientNetB3
echo ====================================================

:: Activate conda environment (sets Library\bin in PATH which includes CUDA DLLs)
call C:\ProgramData\miniconda3\Scripts\activate.bat pneumonia_gpu
if errorlevel 1 (
    echo ERROR: Failed to activate pneumonia_gpu conda environment.
    echo Make sure you have created it with the right packages.
    pause
    exit /b 1
)

echo Conda env: %CONDA_DEFAULT_ENV%

:: Confirm GPU libs in PATH
where cudart64_110.dll >nul 2>&1
if errorlevel 1 (
    echo WARNING: cudart64_110.dll not in PATH — adding conda Library\bin manually
    set PATH=%CONDA_PREFIX%\Library\bin;%PATH%
)

:: Verify TF + GPU
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('TF:', tf.__version__, '| GPUs:', gpus)"
if errorlevel 1 (
    echo ERROR: TensorFlow import failed. Check your environment.
    pause
    exit /b 1
)

:: Run training from project root (paths in script are relative to root)
python backend\train_pneumonia_model.py

if errorlevel 1 (
    echo.
    echo ====================================================
    echo  TRAINING FAILED — see output above for details
    echo ====================================================
    pause
    exit /b 1
)

echo.
echo ====================================================
echo  TRAINING COMPLETE — models saved in backend\models\
echo ====================================================
pause
