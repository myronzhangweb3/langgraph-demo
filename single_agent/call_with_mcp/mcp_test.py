from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def main():
    client = MultiServerMCPClient(
        {
            "allmcp": {
                "url": "http://36.189.252.2:18133/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools_map = {}
    async with client.session("allmcp") as session:
        tools = await load_mcp_tools(session)
        for tool in tools:
            tools_map[tool.name] = tool
        result = await tools_map['ethereum-get_chain_info_ethereum'].arun(tool_input={})
        print(result)


import asyncio

asyncio.run(main())
