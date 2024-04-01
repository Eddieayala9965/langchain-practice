import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, load_tools, AgentExecutor, initialize_agent
from langchain.agents.react.agent import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Using OpenAI Chat API
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0)

memory = ConversationBufferMemory()
# Second Generic Tool
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the LLM Tool
llm_tool = Tool(
    name="Language Model",
    func=llm_chain.run,
    description="Use this tool for general queries and logic"
)
 
tools = load_tools(
    ['llm-math'],
    llm=llm
)
tools.append(llm_tool) # adding the new tool to our tools list


#ReAct framework = Reasoning and Action
conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory
)

    
query = "How old is a person born in 1917 in 2023"
    
query_two = "How old would that person be if their age is multiplied by 100?"
    
print(conversational_agent.agent.llm_chain.prompt.template)

result = conversational_agent(query)
results = conversational_agent(query_two)
# print(result['output'])







