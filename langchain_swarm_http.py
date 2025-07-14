import json
import uuid
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from pydantic import BaseModel

app = FastAPI()

# 全局变量存储各种组件
model = None
tools = {}
agents = {}
workflow = None
checkpointers = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    thread_id: Optional[str] = None


async def initialize_components():
    global model, tools, agents, workflow

    if model is not None:
        return

    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_proxy="http://127.0.0.1:7890",
        model="openai/gpt-4o-2024-11-20"
    )

    # Initialize tool
    tools = {
        "bridge_tools": await MultiServerMCPClient(
            {
                "bridge": {
                    "url": "http://36.189.252.2:18133/mcp/bridge",
                    "transport": "streamable_http",
                }
            }
        ).get_tools(),
        "swap_tools": await MultiServerMCPClient(
            {
                "swap": {
                    "url": "http://36.189.252.2:18133/mcp/swap",
                    "transport": "streamable_http",
                }
            }
        ).get_tools(),
        "transfer_tools": await MultiServerMCPClient(
            {
                "transfer": {
                    "url": "http://36.189.252.2:18133/mcp/transfer",
                    "transport": "streamable_http",
                }
            }
        ).get_tools(),
        "analysis_tools": await MultiServerMCPClient(
            {
                "analysis": {
                    "url": "http://36.189.252.2:18133/mcp/analysis",
                    "transport": "streamable_http",
                }
            }
        ).get_tools(),
    }

    # Initialize proxy tools
    agents = {
        "dispatch_agent": create_handoff_tool(agent_name="dispatch_agent",
                                              description="Transfer to dispatch agent, she can help with dispatch"),
        "bridge_agent": create_handoff_tool(agent_name="bridge_agent",
                                            description="Transfer to bridge agent, she can help with bridge token"),
        "transfer_agent": create_handoff_tool(agent_name="transfer_agent",
                                              description="Transfer to transfer agent, she can help with transfer token"),
        "swap_agent": create_handoff_tool(agent_name="swap_agent",
                                          description="Transfer to swap agent, she can help with swap token. e.g. swap ETH to USDT"),
        "analysis_agent": create_handoff_tool(agent_name="analysis_agent",
                                              description="Transfer to analysis agent, she can help with analysis chain info and token info"),
    }

    # Create each agent
    dispatch_agent = create_react_agent(
        model,
        tools=list({k: v for k, v in agents.items() if k != "dispatch_agent"}.values()),
        prompt="You are Dispatch Agent.",
        name="dispatch_agent",
    )

    analysis_agent = create_react_agent(
        model,
        tools=tools["analysis_tools"] + list({k: v for k, v in agents.items() if k != "analysis_agent"}.values()),
        prompt="You are asset analysis Agent. You can query user assets, chain information, and transaction information.",
        name="analysis_agent",
    )

    bridge_agent = create_react_agent(
        model,
        tools=tools["bridge_tools"] + list({k: v for k, v in agents.items() if k != "bridge_agent"}.values()),
        prompt="You are bridge agent. You can transfer the user's token from one chain to another.",
        name="bridge_agent",
    )

    swap_agent = create_react_agent(
        model,
        tools=tools["swap_tools"] + list({k: v for k, v in agents.items() if k != "swap_agent"}.values()),
        prompt="You are swap agent.",
        name="swap_agent",
    )

    transfer_agent = create_react_agent(
        model,
        tools=tools["transfer_tools"] + list({k: v for k, v in agents.items() if k != "transfer_agent"}.values()),
        prompt="You are transfer agent. Before sending a transaction, it is necessary to check if the user's balance is sufficient. If not, consider swapping or bridging equivalent tokens. Note that when swapping or bridging, a portion of the gas fee needs to be reserved.",
        name="transfer_agent",
    )

    # 创建工作流
    workflow = create_swarm(
        [dispatch_agent, analysis_agent, bridge_agent, swap_agent, transfer_agent],
        default_active_agent=dispatch_agent.name
    )


@app.on_event("startup")
async def startup_event():
    await initialize_components()


async def process_chat_stream(request_data: Dict[str, Any], thread_id: str):
    """Process chat streams and generate responses"""
    global workflow, checkpointers

    await initialize_components()

    if thread_id not in checkpointers:
        checkpointers[thread_id] = InMemorySaver()

    agent = workflow.compile(checkpointer=checkpointers[thread_id])

    config = {"configurable": {"thread_id": thread_id}}

    input_data = {"messages": request_data["messages"]}

    try:
        msg_idx = 0
        async for turn in agent.astream(
                input=input_data,
                config=config,
                stream_mode="values",
        ):
            while len(turn['messages']) > msg_idx:
                if turn['messages'][msg_idx].type in ('ai', 'tool'):
                    messages = turn['messages'][msg_idx]
                    content = messages.content if messages.content else messages.additional_kwargs
                    yield f"data: {json.dumps({'content': content, 'role': 'assistant'})}\n\n"
                msg_idx += 1
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def openai_compatible_chat(request: Request):
    data = await request.json()

    messages = data.get("messages", [])
    thread_id = data.get("thread_id") or str(uuid.uuid4())

    request_data = {"messages": messages}

    return StreamingResponse(
        process_chat_stream(request_data, thread_id),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        start_server()
    except Exception as e:
        print(f"Server error: {e}")
