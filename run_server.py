"""
Server execution runner for MAUSAM.
Supports local execution and production cloud environments (Render, Railway, Hugging Face).
"""
import uvicorn
import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.config import settings

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    host = os.environ.get("HOST", settings.HOST)
    is_debug = os.environ.get("DEBUG", str(settings.DEBUG)).lower() in ["true", "1"]
    
    print(f"Starting {settings.PROJECT_NAME}...")
    print(f"Server running at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=is_debug)
