#!/bin/bash
# Script para testar build e execução do Docker

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════╗"
echo "║        DOCKER BUILD & TEST - Flyer API               ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Build image
echo -e "${YELLOW}📦 Step 1/4: Building Docker image...${NC}"
docker-compose build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image built successfully!${NC}"
else
    echo -e "${RED}❌ Failed to build image${NC}"
    exit 1
fi

echo ""

# Step 2: Start container
echo -e "${YELLOW}🚀 Step 2/4: Starting container...${NC}"
docker-compose up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container started!${NC}"
else
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi

echo ""

# Step 3: Wait for API to be ready
echo -e "${YELLOW}⏳ Step 3/4: Waiting for API to be ready...${NC}"
sleep 10

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is ready!${NC}"
        break
    else
        echo -n "."
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n${RED}❌ API failed to start after ${MAX_RETRIES} retries${NC}"
    echo ""
    echo "Container logs:"
    docker-compose logs
    exit 1
fi

echo ""

# Step 4: Test API
echo -e "${YELLOW}🧪 Step 4/4: Testing API endpoint...${NC}"

if [ -f "examples/image_example.png" ]; then
    echo "Testing with examples/image_example.png..."
    
    RESPONSE=$(curl -s -X POST http://localhost:8000/analyze-flyer \
        -F "file=@examples/image_example.png" \
        -H "accept: application/json")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ API test successful!${NC}"
        echo ""
        echo "Response:"
        echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
    else
        echo -e "${RED}❌ API test failed${NC}"
    fi
else
    echo "⚠️  No test image found at examples/image_example.png"
    echo "Testing health endpoint only..."
    curl -s http://localhost:8000/health | python -m json.tool
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 Docker setup complete!${NC}"
echo ""
echo "API is running at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f     # View logs"
echo "  docker-compose down        # Stop container"
echo "  docker-compose restart     # Restart container"
echo ""
echo "Or use Makefile:"
echo "  make logs                  # View logs"
echo "  make down                  # Stop container"
echo ""
