from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class Review(BaseModel):
    text: str

@app.post("/analyze-review")
def analyze(review: Review):
    result = classifier(review.text)[0]
    return {"label": result["label"], "score": result["score"]}
