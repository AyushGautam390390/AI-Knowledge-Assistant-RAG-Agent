from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_agent import agent

app = FastAPI(title="AI Knowledge Assistant")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        res = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.question
                }
            ]
        })

        return {
            "question": request.question,
            "answer": res["messages"][-1].content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
