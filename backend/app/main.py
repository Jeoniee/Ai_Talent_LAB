from fastapi import FastAPI
from pydantic import BaseModel
from frontend.graph import build_graph

app = FastAPI()

class DebateRequest(BaseModel):
    topic: str

@app.post("/debate")
def debate(req: DebateRequest):
    graph = build_graph()
    state = {
        "topic": req.topic,
        "plan": None,
        "retrieved_docs": [],
        "prosecution": [],
        "defense": [],
        "judge": None,
        "final_report": None,
    }

    for step in graph.stream(state):
        pass  # 중간결과는 무시, 최종 state만 리턴

    return {"topic": req.topic, "final_report": state["final_report"]}
