#!/bin/bash
# Quick Start Script - Test the system without full installation

echo "╔═══════════════════════════════════════════════════════╗"
echo "║       RETAIL FLYER AI - Quick Start                  ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "💡 Tip: Consider using a virtual environment:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Menu
echo "Choose an option:"
echo ""
echo "  1) Install dependencies (required for first run)"
echo "  2) Start Legacy API (v1)"
echo "  3) Start Modern API (v2)"
echo "  4) Run quick demo"
echo "  5) Show Makefile commands"
echo "  6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "📦 Installing dependencies..."
        pip install -q fastapi uvicorn pydantic pillow numpy
        echo "✅ Basic dependencies installed!"
        echo ""
        echo "💡 For full functionality, also install:"
        echo "   pip install ultralytics paddleocr opencv-python"
        echo ""
        ;;
    2)
        echo ""
        echo "🚀 Starting Legacy API (v1)..."
        python api.py
        ;;
    3)
        echo ""
        echo "🔮 Starting Modern API (v2)..."
        python src/serving/api_v2.py
        ;;
    4)
        echo ""
        python quick_demo.py
        ;;
    5)
        echo ""
        make help
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
