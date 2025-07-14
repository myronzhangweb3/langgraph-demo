import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def main():
    client = MultiServerMCPClient(
        {
            "allmcp": {
                "url": "http://36.189.252.2:18133/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_proxy="http://127.0.0.1:7890",
        model="openai/gpt-4o-2024-11-20")

    agent = create_react_agent(model, tools)
    response = await agent.ainvoke(stream_mode="debug", input={"messages": "当前 ethereum 高度"})
    print(response)


asyncio.run(main())
