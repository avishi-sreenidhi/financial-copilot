#!/usr/bin/env python3
"""
Data Analytics Agent Swarm - Startup Script
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    try:
        import uvicorn
        from api.main import app
        
        print("🚀 Starting Data Analytics Agent Swarm...")
        print("📊 Features: Advanced Analytics, Smart Visualizations, Data Search")
        print("🔗 Access at: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running from the src directory:")
        print("   cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
