"""
Root ASGI entrypoint for Vercel, Docker, and Production Deployments.
Exposes 'app' variable directly at the repository root.
"""
from backend.app.main import app

# For local testing if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
