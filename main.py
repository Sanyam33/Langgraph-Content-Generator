from fastapi import FastAPI
from api import router

app = FastAPI(title="LangGraph Content Generator API")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Welcome to the Content Generator API"}

@app.get("/help")
def help():
    return {
        "route": "POST /api/v1/generate-content",
        "example_payload": {
             "website_url": "https://www.google.com",
             "target_url": "https://www.google.com",
             "anchor_text": "Google",
             "word_count": 100,
             "category": "Search Engine",
             "writing_style": "Technical and Informative"
        }
    }