import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.chains.llm_math.base import LLMMathChain



load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0)
llm_math = LLMMathChain.from_llm(llm=llm)

math_tool = Tool(
    name="Calculator",
    func=llm_math.invoke,
    description="Useful for when you need to answer questions related to math"
)
tools = load_tools(
    ['llm-math']
)

print(tools[0].name, tools[0].description)



agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3 
)

response = agent.invoke("what is 3.1^2.1")
print(response)

# response = llm.invoke("what is Langchain and how does it work?")
# print(response.content)
