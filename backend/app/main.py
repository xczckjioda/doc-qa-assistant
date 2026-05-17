from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload
from app.routes import ask
from app.routes import files
from app.routes import evaluate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(files.router)
app.include_router(evaluate.router)
@app.get("/")
def root():
    return {"message": "Backend is running"}
