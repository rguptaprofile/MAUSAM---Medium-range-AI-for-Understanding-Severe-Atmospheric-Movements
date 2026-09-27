"""
Root ASGI entrypoint for Render, Vercel, Docker, and Production Deployments.
Exposes 'app' variable directly at the repository root.
"""
import os
import sys

# Ensure repository root and current working directory are on python path
root_dir = os.path.abspath(os.path.dirname(__file__))
for p in [root_dir, os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.main import app

# Alias for serverless runtime handlers
handler = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
