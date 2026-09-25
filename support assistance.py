from fastapi import FastAPI
from app import AskRequest, Answer, ask

app=FastAPI(
    title='Zepto Support Assistant'
)
@app.post(
    '/ask',
    response_model=Answer
)
def endpoint(
    req:AskRequest
):
    return ask(
        req.query
    )