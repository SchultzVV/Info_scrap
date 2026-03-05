#!/bin/bash
# Test Automation Script - Both Versions

echo "======================================================================"
echo "🧪 Flyer Extraction System - Automated Tests"
echo "======================================================================"
echo ""

# Check if image exists
if [ ! -f "examples/image_example.png" ]; then
    echo "❌ Error: examples/image_example.png not found!"
    echo "   Please add a test image to examples/"
    exit 1
fi

# Function to test v2
test_v2() {
    echo "======================================================================"
    echo "TEST 1: Modern Pipeline (v2)"
    echo "======================================================================"
    echo ""
    echo "Testing: Multimodal Transformer pipeline..."
    python tests/test_modern_pipeline.py examples/image_example.png
    
    echo ""
    echo "======================================================================"
    echo "TEST 2: Modern API (v2)"
    echo "======================================================================"
    echo ""
    echo "⚠️  Note: Start server first with: python src/serving/api_v2.py"
    echo "Then run: python tests/test_api_v2.py examples/image_example.png"
    echo ""
}

# Function to test v1
test_v1() {
    echo "======================================================================"
    echo "TEST 3: Legacy Pipeline (v1)"
    echo "======================================================================"
    echo ""
    echo "Testing: YOLO + OCR + LayoutLM pipeline..."
    python tests/test_pipeline.py examples/image_example.png
    
    echo ""
    echo "======================================================================"
    echo "TEST 4: Legacy API (v1)"
    echo "======================================================================"
    echo ""
    echo "⚠️  Note: Start server first with: python api.py"
    echo "Then run: python tests/test_api.py examples/image_example.png"
    echo ""
}

# Menu
case ${1:-v2} in
    v1|legacy)
        test_v1
        ;;
    v2|modern)
        test_v2
        ;;
    all|both)
        test_v2
        echo ""
        test_v1
        ;;
    *)
        echo "Usage: ./run_tests.sh [v1|v2|all]"
        echo ""
        echo "Options:"
        echo "  v2, modern  - Test Modern Pipeline (v2) [default]"
        echo "  v1, legacy  - Test Legacy Pipeline (v1)"
        echo "  all, both   - Test both versions"
        exit 1
        ;;
esac

echo "======================================================================"
echo "✨ Tests configuration complete!"
echo "======================================================================"
