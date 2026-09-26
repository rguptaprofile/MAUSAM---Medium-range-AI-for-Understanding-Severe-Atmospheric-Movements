from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "status": "online",
            "message": "MAUSAM Serverless Backend is operational on Vercel!",
            "runtime": "python"
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
        return
