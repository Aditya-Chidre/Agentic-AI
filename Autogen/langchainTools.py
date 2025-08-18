import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv
from autogen_ext.tools.http import HttpTool
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper

api_key = os.getenv("GOOGLE_API_KEY")

model_client=OpenAIChatCompletionClient(model="gemini-2.0-flash",api_key=api_key)
os.environ['SERPER_API_KEY']='03efe53bf8044f01c84e9c0d43a252fc74dc64a7'

search_tool_wrapper = GoogleSerperAPIWrapper(type='search')

def search_web(query:str) ->str:
    """Search the web for the given query and return the results."""

    if query == 'ipl':
        return 'IPL is Indian Premier League, a professional Twenty20 cricket league in India.' # Mocking the call
    else:           
        try:
            results = search_tool_wrapper.run(query)
            return results
        except Exception as e:
            print(f"Error occurred while searching the web: {e}")
            return "No results found."  


search_agent = AssistantAgent(
    name="SearchAgent",
    model_client=model_client,
    tools=[search_web],
    description="An agent that can search the web for information.",
    system_message="You are a helpful assistant that can search the web for information using the search_web tool. If not give answer by ur own",
    reflect_on_tool_use=True,
)

async def run_serper_search():
    """Run the search agent with a sample query."""
    query = "ipl" 
    print(f"Querying: {query}")
    
    
    result = await search_agent.run(task=query)
    print("####### RESULT ######")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(run_serper_search())