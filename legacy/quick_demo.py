#!/usr/bin/env python3
"""
Quick demo script - process image without full dependencies
This will run the API server and test with curl
"""

import subprocess
import time
import sys
import signal
from pathlib import Path

def check_dependencies():
    """Check if basic dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        return True
    except ImportError:
        print("❌ Missing dependencies. Run: make install")
        return False

def start_api(use_modern=False):
    """Start API server in background"""
    if use_modern:
        cmd = ["python", "src/serving/api_v2.py"]
        endpoint = "/flyer/extract"
    else:
        cmd = ["python", "api.py"]
        endpoint = "/analyze-flyer"
    
    print(f"🚀 Starting {'Modern' if use_modern else 'Legacy'} API server...")
    print(f"📍 Endpoint: POST http://localhost:8000{endpoint}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Check if server is running
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/health"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            print("✅ Server started successfully!")
            return proc, endpoint
        else:
            print("⚠️  Server might not be ready yet...")
            return proc, endpoint
    except:
        print("⚠️  Could not verify server status")
        return proc, endpoint

def test_endpoint(endpoint, image_path):
    """Test API endpoint with image"""
    print(f"\n📤 Testing endpoint with: {image_path}")
    
    cmd = [
        "curl",
        "-X", "POST",
        f"http://localhost:8000{endpoint}",
        "-F", f"file=@{image_path}",
        "-H", "accept: application/json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ Response received:")
        print("=" * 60)
        print(result.stdout)
        print("=" * 60)
    else:
        print(f"\n❌ Request failed: {result.stderr}")

def main():
    if not check_dependencies():
        print("\n💡 Install dependencies first:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    
    # Parse arguments
    use_modern = "--modern" in sys.argv or "-m" in sys.argv
    image_path = "examples/image_example.png"
    
    # Check if custom image provided
    for arg in sys.argv[1:]:
        if not arg.startswith("-") and Path(arg).exists():
            image_path = arg
            break
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    proc = None
    try:
        # Start API server
        proc, endpoint = start_api(use_modern)
        
        # Test endpoint
        test_endpoint(endpoint, image_path)
        
        print("\n💡 Server is still running. Press Ctrl+C to stop.")
        print("   You can test more images with:")
        print(f"   curl -X POST http://localhost:8000{endpoint} -F 'file=@image.png'")
        
        # Keep server running
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=5)
        print("✅ Done!")

if __name__ == "__main__":
    main()
