import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_proxy="http://127.0.0.1:7890",
    model="openai/gpt-4o-2024-11-20")


async def main():
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
        prompt="You are bridge agent. You can bridge the user's token from one chain to another.",
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
        prompt="You are transfer agent. Before sending a transaction, it is necessary to check if the user's balance is sufficient. If not, consider swapping or bridging equivalent tokens. Note that swapping or bridging requires reserving a portion of the gas fee.",
        name="transfer_agent",
    )

    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [dispatch_agent, analysis_agent, bridge_agent, swap_agent, transfer_agent],
        default_active_agent=dispatch_agent.name
    )
    agent = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    async for turn in agent.astream(
            input={"messages": [
                # case 1
                # {"role": "user", "content": "Current Ethereum height"}]},

                # case 2
                {"role": "user", "content": "What assets does 0xF5054F94009B7E9999F6459f40d8EaB1A2ceA22D have?"}]},

                # case 3
                # {"role": "user",
                #  "content": "My wallet address is 0xF5054F94009B7E9999F6459f40d8EaB1A2ceA22D，I want to send 0xD64229dF1EB0354583F46e46580849B1572BB56d 0.1 USDT on Ethereum"}]},

            config=config,
            stream_mode="debug",
    ):
        print(turn)
        print(f"step: {turn.get('step')}")
        print(f"type: {turn.get('type')}")
        if turn.get('payload').__contains__('values'):
            for msg in turn['payload']['values']['messages']:
                print(f"{type(msg).__name__}: {msg.content if msg.content != '' else msg.additional_kwargs}")


# asyncio.run(main())

try:
    asyncio.run(main())
except Exception as e:
    pass  # 忽略特定异常