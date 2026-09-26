"""
Server execution runner for MAUSAM.
"""
import uvicorn
import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.PROJECT_NAME}...")
    print(f"Server running at: http://{settings.HOST}:{settings.PORT}")
    print(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
