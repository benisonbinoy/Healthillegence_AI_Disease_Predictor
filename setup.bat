@echo off
echo ==================================
echo Healthillegence - Setup Script
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.9+ first.
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo X Node.js is not installed. Please install Node.js 18+ first.
    exit /b 1
)

echo + Python and Node.js are installed
echo.

REM Install frontend dependencies
echo Installing frontend dependencies...
call npm install

REM Install backend dependencies
echo.
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Download datasets (see backend\DATASETS.md)
echo 2. Train models: cd backend ^&^& python train_models.py ^&^& python train_image_models.py
echo 3. Start backend: cd backend ^&^& python api_server.py
echo 4. Start frontend: npm run dev
echo.
pause
