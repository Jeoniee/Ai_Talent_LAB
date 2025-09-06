import os
#LangSmith 시각화
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your_api_key"

from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ==========================
# Azure OpenAI 연결
# ==========================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    deployment_name=os.getenv("AOAI_DEPLOY_GPT4O_MINI"),
    api_version="2024-08-01-preview"
)

# ==========================
# Workers 설정
# ==========================
members = ["nutritionist", "dietitian", "recipe"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


# ==========================
# Router 정의
# ==========================
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["nutritionist", "dietitian", "recipe", "FINISH"]


class State(MessagesState):
    next: str


# ==========================
# Supervisor Node
# ==========================
def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


# ==========================
# Tools
# ==========================
@tool(return_direct=True)
def get_nutrition_info(food: str) -> str:
    """음식의 영양 성분/칼로리를 알려준다."""
    sample = {"김치찌개": "대략 300kcal. 나트륨 높음, 단백질 15g"}
    return sample.get(food, f"{food}의 영양정보 없음")


@tool(return_direct=True)
def get_diet_plan(day: str) -> str:
    """요일에 따라 하루 식단을 추천한다."""
    sample = {"월요일": "아침:바나나+요거트, 점심:샌드위치, 저녁:파스타"}
    return sample.get(day, f"{day}의 식단 정보 없음")


@tool(return_direct=True)
def get_recipe(dish: str) -> str:
    """요리를 간단한 레시피로 바꿔준다."""
    sample = {
        "두유 파스타": "두유, 크림, 새우, 마늘, 페퍼론치노를 넣기.",
        "샐러드 볼": "신선한 채소와 닭가슴살을 곁들여 간단히 준비",
        "닭가슴살 구이": "닭가슴살에 소금, 후추 간하고 오븐에 굽기",
        "렌틸콩 스튜": "렌틸콩과 토마토, 채소를 넣고 끓이기",
        "구운 연어": "연어에 올리브오일, 허브를 바르고 180도에서 15분 구워내기",
        "두부 스테이크": "두부를 팬에 구워 간장소스와 곁들이기",
        "퀴노아 샐러드": "퀴노아와 채소를 섞고 올리브오일 드레싱"
    }
    return sample.get(dish, f"{dish}의 레시피 정보 없음")


# ==========================
# Worker Agents
# ==========================
nutrition_agent = create_react_agent(
    llm, tools=[get_nutrition_info], prompt="너는 영양사야. 내가 원하는 음식의 영양 정보를 알려줘."
)
diet_agent = create_react_agent(
    llm, tools=[get_diet_plan], prompt="너는 헬스트레이너야. 나의 요일에 맞는 식단을 추천하고 관리해줘. "
)
chef_agent = create_react_agent(
    llm, tools=[get_recipe], prompt="너는 나의 요리사야. 나의 식단의 레시피를 알려줘."
)


# ==========================
# Worker Nodes
# ==========================
def nutrition_node(state: State) -> Command[Literal["supervisor"]]:
    result = nutrition_agent.invoke(state)
    print(f"🔄 Nutrition Node State: {state['messages']}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="nutrition")
            ]
        },
        goto="supervisor",
    )


def diet_node(state: State) -> Command[Literal["supervisor"]]:
    result = diet_agent.invoke(state)
    print(f"🔄 Diet Node State: {state['messages']}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="diet")
            ]
        },
        goto="supervisor",
    )


def recipe_node(state: State) -> Command[Literal["supervisor"]]:
    result = chef_agent.invoke(state)
    print(f"🔄 Recipe Node State: {state['messages']}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="recipe")
            ]
        },
        goto="supervisor",
    )


# ==========================
# Graph 빌드
# ==========================
builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("nutritionist", nutrition_node)
builder.add_node("dietitian", diet_node)
builder.add_node("recipe", recipe_node)
graph = builder.compile()

# ==========================
# 실행 예시
# ==========================
print("🍽️ 식단 플래너 실행 결과\n")

for step in graph.stream({"messages": [("user", "나의 일주일 식단이 궁금해")]}, subgraphs=True):
    if isinstance(step, tuple) and len(step) == 2:
        node, state = step
        if (node):
            print(f"🟢 현재 노드: {node}")
        else:
            pass

        # case 1: 기본 messages
        if "messages" in state and state["messages"]:
            last_msg = state["messages"][-1]
            print("💬 메시지:", getattr(last_msg, "content", str(last_msg)))

        # case 2: agent 결과
        elif "agent" in state and "messages" in state["agent"]:
            last_msg = state["agent"]["messages"][-1]
            print("🤖 Agent 응답:", getattr(last_msg, "content", str(last_msg)))

        # case 3: tool 결과
        elif "tools" in state and "messages" in state["tools"]:
            last_msg = state["tools"]["messages"][-1]
            print("🛠️ Tool 응답:", getattr(last_msg, "content", str(last_msg)))

        else:
            pass

        print("-" * 50)
    else:
        print("📌 Supervisor 단계:", step)
        print("-" * 50)
