#!/bin/bash
# Quick Start Script for Flyer Extraction System

set -e

echo "🚀 Flyer Extraction System - Quick Start"
echo "========================================"
echo ""

# Check which version to start
VERSION=${1:-v2}

case $VERSION in
  v1|legacy)
    echo "Starting Legacy Pipeline (v1)..."
    echo "Architecture: YOLO + OCR + LayoutLM"
    echo "Port: 8000"
    echo ""
    docker-compose up
    ;;
    
  v2|modern)
    echo "Starting Modern Pipeline (v2)..."
    echo "Architecture: Multimodal Transformers"
    echo "Port: 8000"
    echo ""
    docker-compose -f docker-compose-v2.yml up flyer-extraction-v2
    ;;
    
  both)
    echo "Starting Both Versions..."
    echo "v1 (Legacy) → Port 8000"
    echo "v2 (Modern) → Port 8001"
    echo ""
    echo "⚠️  Starting both requires two terminals:"
    echo "   Terminal 1: docker-compose up"
    echo "   Terminal 2: docker-compose -f docker-compose-v2.yml up"
    echo ""
    echo "Starting v2 only (or use separate terminals)..."
    docker-compose -f docker-compose-v2.yml up
    ;;
    
  *)
    echo "Usage: ./quick_start.sh [v1|v2|both]"
    echo ""
    echo "Options:"
    echo "  v1, legacy  - Start Legacy Pipeline (YOLO + OCR)"
    echo "  v2, modern  - Start Modern Pipeline (Multimodal Transformers) [default]"
    echo "  both        - Start both versions"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh              # Start v2 (modern)"
    echo "  ./quick_start.sh v1           # Start v1 (legacy)"
    echo "  ./quick_start.sh both         # Start both"
    exit 1
    ;;
esac
