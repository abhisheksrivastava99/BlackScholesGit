"""
backend/main.py
Main entry point for Black-Scholes NN FastAPI app.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predictions   # Import your prediction router

app = FastAPI(
    title="Black-Scholes Neural Network API",
    description="API for option pricing with neural network and analytical Greeks.",
    version="1.0.0",
    docs_url="/",  # Serve API docs at root by default
)

# CORS setup (allow all for development/demo; restrict origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predictions.router)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# For direct script execution (optional, allows python backend/main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
