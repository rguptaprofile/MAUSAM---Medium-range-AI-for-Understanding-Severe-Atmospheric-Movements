"""
Root ASGI entrypoint for Render, Vercel, Docker, and Production Deployments.
Exposes 'app' variable directly at the repository root.
"""
import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
