import asyncio

import httpx

from multi_agent.a2a.host_agent.host_agent import HostAgent
import json # Needed for pretty printing dicts

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "TESTAPP"
USER_ID = "test_user_456"
session_service = InMemorySessionService()
SESSION_ID_TOOL_AGENT = "SESSION_ID_TOOL_AGENT"

# --- 6. Define Agent Interaction Logic ---
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = await session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)

async def main():
    root_agent = HostAgent(remote_agent_addresses=[
        'http://localhost:10000', # analysis_agent
        'http://localhost:10001', # bridge_agent
        'http://localhost:10002', # swap_agent
        'http://localhost:10003', # transfer_agent
    ],
        http_client=httpx.AsyncClient(timeout=30)).create_agent()

    capital_runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)

    # await call_agent_and_print(capital_runner, root_agent, session_id=SESSION_ID_TOOL_AGENT, query_json='Query the current Ethereum height.')
    await call_agent_and_print(capital_runner, root_agent, session_id=SESSION_ID_TOOL_AGENT, query_json='What assets does 0xF5054F94009B7E9999F6459f40d8EaB1A2ceA22D have on all chains?')
    # await call_agent_and_print(capital_runner, root_agent, session_id=SESSION_ID_TOOL_AGENT, query_json='My wallet address is 0xF5054F94009B7E9999F6459f40d8EaB1A2ceA22D，I want to send 0xD64229dF1EB0354583F46e46580849B1572BB56d 0.1 USDT on Ethereum"')

    # while True:
    #     user_input = input("Human('exit' to exit):")
    #     if user_input.lower() == 'exit':
    #         print("The program has exited.")
    #         break
    #     else:
    #         await call_agent_and_print(capital_runner, root_agent, session_id=SESSION_ID_TOOL_AGENT,
    #                                    query_json=user_input)


asyncio.run(main())
