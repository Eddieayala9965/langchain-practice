import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, load_tools, AgentExecutor, cr
from langchain.agents.react.agent import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Using OpenAI Chat API
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0)

# Adjusted PromptTemplate to include required variables for the React agent
prompt_template = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the tools as needed to answer the question. For reference, the tool names are: {tool_names}.

Question: {query}
Thought: {agent_scratchpad}
Action: Decide which tool to use based on the thought process and explain why. If no tool is needed, explain how you would directly answer the question.

Remember to think about which tool could best answer the question or assist with the task at hand. Utilize the tools effectively to provide the best answer you can.
'''

prompt = PromptTemplate(
    input_variables=["query", "agent_scratchpad", "tools", "tool_names"],  # Including tools and tool_names
    template=prompt_template
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

llm_tool = Tool(
    name="Language Model",
    func=llm_chain.invoke,
    description="Use this tool for general queries and logic"
)

tools = load_tools(['llm-math'], llm=llm)
tools.append(llm_tool)

# Creating the React agent with the correctly structured prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, max_iterations=3, verbose=True, )

query = "what is the capital of France?"
result = agent_executor.invoke({
    "query": query,  # Correcting this to match the expected variable name
    "agent_scratchpad": ""  # Initially, this could be an empty string or contain previous context
})
print(result["output"])
