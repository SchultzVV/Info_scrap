#!/bin/bash

# Quick Start Script
# Retail Flyer Understanding System

echo "🏷️  RETAIL FLYER UNDERSTANDING SYSTEM"
echo "====================================="
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    else
        echo "⚠️  No virtual environment found"
        echo "   Run ./install.sh first"
        exit 1
    fi
fi

echo "🚀 Starting API server..."
echo ""
echo "📚 Documentation: http://localhost:8000/docs"
echo "🔍 Health check:  http://localhost:8000/health"
echo "📊 Info:          http://localhost:8000/info"
echo ""
echo "Press Ctrl+C to stop"
echo "====================================="
echo ""

python api.py
