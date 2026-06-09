from fastapi import FastAPI
from app.api import router as api_router

app = FastAPI(title="VeloceAI API", version="1.0.0")

@app.get("/")
def read_root():
    return {"status": "VeloceAI Engine Online"}

app.include_router(api_router, prefix="/api")
