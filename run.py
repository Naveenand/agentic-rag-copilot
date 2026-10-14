"""Start the web app for local development: http://127.0.0.1:8000

In production, run it with Docker (see README), which uses:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=["app", "templates", "static"])
