
# main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Testing...FastAPI at work!"}

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}