import os 
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import HumanMessage

# from langchain.openai import ChatOpenAI this is the newer syntax

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


llm_model = "gpt-4"


prompt = "how old is the universe?"
messaegs = [HumanMessage(content = prompt)]

llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(temperature=0.7, model=llm_model)


print(llm.invoke("What is the weather in Greenville South Carolina?"))
print("====================================")

result = chat_model.invoke(messaegs)
print(result.content)
# this is a way to check for the key value pairs in the result
# print(result)

# result = chat_model.invoke("What is the weather in Greenville South Carolina?")
# print(result.content)

