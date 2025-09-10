# graph.py
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from utils import get_retriever

# -----------------------------
# LLM 초기화
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-05-01-preview"),
    deployment_name=os.getenv("AOAI_DEPLOY_GPT4O_MINI"),
    temperature=0.3,
)

# -----------------------------
# 프롬프트
# -----------------------------
planner_prompt = "주제 {topic} 에 대해 검사·변호사·판사가 토론할 계획을 JSON으로 짜라."
prosecution_prompt = "주제 {topic} 에 대해 검사 입장에서 발언하라.\n{docs}"
defense_prompt = "주제 {topic} 에 대해 변호사 입장에서 반박하라.\n검사 주장: {pros}\n{docs}"
judge_prompt = "검사: {pros}\n변호사: {defs}\n주제 {topic} 에 대해 판사로서 요약·판단하라.\n{docs}"
writer_prompt = "최종 보고서를 작성하라.\n주제: {topic}\n검사: {pros}\n변호사: {defs}\n판사: {judge}"

# -----------------------------
# 상태 정의
# -----------------------------
def init_state() -> Dict[str, Any]:
    return {
        "topic": None,
        "plan": None,
        "retrieved_docs": [],
        "prosecution": [],
        "defense": [],
        "judge": None,
        "final_report": None,
    }

# -----------------------------
# 노드 함수들
# -----------------------------
def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = planner_prompt.format(topic=state["topic"])
    res = llm.invoke(prompt)
    state["plan"] = res.content
    return state

def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(state["topic"])
    state["retrieved_docs"] = docs
    return state

def prosecution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    docs_text = "\n".join([d.page_content for d in state["retrieved_docs"][:3]])
    prompt = prosecution_prompt.format(topic=state["topic"], docs=docs_text)
    res = llm.invoke(prompt)
    state["prosecution"].append(res.content)
    return state

def defense_node(state: Dict[str, Any]) -> Dict[str, Any]:
    docs_text = "\n".join([d.page_content for d in state["retrieved_docs"][:3]])
    pros_text = "\n".join(state["prosecution"])
    prompt = defense_prompt.format(topic=state["topic"], pros=pros_text, docs=docs_text)
    res = llm.invoke(prompt)
    state["defense"].append(res.content)
    return state

def judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    docs_text = "\n".join([d.page_content for d in state["retrieved_docs"][:3]])
    pros_text = "\n".join(state["prosecution"])
    defs_text = "\n".join(state["defense"])
    prompt = judge_prompt.format(topic=state["topic"], pros=pros_text, defs=defs_text, docs=docs_text)
    res = llm.invoke(prompt)
    state["judge"] = res.content
    return state

def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    pros_text = "\n".join(state["prosecution"])
    defs_text = "\n".join(state["defense"])
    prompt = writer_prompt.format(topic=state["topic"], pros=pros_text, defs=defs_text, judge=state["judge"])
    res = llm.invoke(prompt)
    state["final_report"] = res.content
    return state

# -----------------------------
# Graph 구성
# -----------------------------
def build_graph():
    workflow = StateGraph(dict)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("prosecution", prosecution_node)
    workflow.add_node("defense", defense_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "prosecution")
    workflow.add_edge("prosecution", "defense")
    workflow.add_edge("defense", "judge")
    workflow.add_edge("judge", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# -----------------------------
# 실행 예시
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    state = init_state()
    state["topic"] = "사형제도 유지 vs 폐지"

    for step in graph.stream(state):
        print(step)

    print("\n=== 최종 보고서 ===")
    print(state["final_report"])
