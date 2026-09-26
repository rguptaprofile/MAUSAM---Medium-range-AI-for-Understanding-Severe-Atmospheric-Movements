import os
import sys
import traceback

# Ensure root directory and current working directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [root_dir, os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.app.main import app
except Exception as e:
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    app = FastAPI(title="MAUSAM Diagnostic")
    error_trace = traceback.format_exc()
    
    @app.get("/{full_path:path}")
    @app.post("/{full_path:path}")
    def catch_all(full_path: str = ""):
        return PlainTextResponse(f"MAUSAM BACKEND STARTUP EXCEPTION:\n\n{error_trace}\n\nSYS_PATH:\n{sys.path}\n\nCWD:\n{os.getcwd()}\n\nDIR_CONTENTS:\n{os.listdir('.')}", status_code=200)

handler = app
