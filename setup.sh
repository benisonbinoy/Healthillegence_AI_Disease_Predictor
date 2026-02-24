#!/bin/bash

echo "=================================="
echo "Healthiligence - Setup Script"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null
then
    echo "❌ Python is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null
then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✓ Python and Node.js are installed"
echo ""

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download datasets (see backend/DATASETS.md)"
echo "2. Train models: cd backend && python train_models.py && python train_image_models.py"
echo "3. Start backend: cd backend && python api_server.py"
echo "4. Start frontend: npm run dev"
echo ""
