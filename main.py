import os
from dotenv import load_dotenv
load_dotenv()

import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from qa_pipeline import get_answer
# ADD these imports at the top if missing
from fastapi import FastAPI, HTTPException
from hybrid_search.hybrid_search import hybrid_search
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # or whatever embeddings you're using

# 👇 Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 👇 Load index.pkl using an absolute path
with open(os.path.join(BASE_DIR, "index.pkl"), "rb") as f:
    faiss_index = pickle.load(f)

# 👇 Load FAISS index using absolute path
faiss_index_path = os.path.join(BASE_DIR, "index")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)







from qa_pipeline import load_vectorstore, embed_text, answer_question

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace ["*"] with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the FAISS vectorstore once on startup
vectorstore = load_vectorstore()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded string

class LinkItem(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[LinkItem]

@app.post("/", response_model=QuestionResponse)
async def ask_question(data: QuestionRequest):
    try:
        answer, sources = answer_question(data.question, vectorstore)

        # sources is a list of dicts with keys 'url' and 'text'
        links = list(sources)

        return JSONResponse(
            content={
                "answer": answer,
                "links": links
            },
            media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "🎓 TDS Virtual TA API is running successfully!"}
@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = get_answer(request.question)
    return response
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    ...
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
@app.post("/ask")
def ask_question(request: QuestionRequest):
    print(f"Received question: {request.question}")
    print(f"Image: {request.image}")
    try:
        answer = hybrid_search(request.question)
        return answer
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qa_pipeline import get_answer

  # assuming this exists
def get_answer(question: str):
    # your logic here
    return "This is a dummy answer"


class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
import os
print("Current working directory:", os.getcwd())
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'index.pkl'), 'rb') as f:
    some_object = pickle.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

