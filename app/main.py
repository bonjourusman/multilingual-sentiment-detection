from fastapi import FastAPI
from .context import helpers as h

app = FastAPI()
model_path = './finetuned_model'

@app.get('/')
async def health_check():
    return {'health_check': 'OK'}

@app.get('/info')
async def info():
    return {
        'name':'multilingual-sentiment',
        'description': 'Returns sentiment (positive or negative). Supports multiple languages as input. Powered by a finetuned ChatGPT-2 LLM model.'
        }

@app.get('/analyze')
async def analyze(query: str):

    # Get sentiment
    sentiment = h.query_model(model_path, query)

    return {
        'query': query,
        'sentiment': sentiment
    }