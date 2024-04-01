import openai
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory    

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4"

llm = ChatOpenAI(temperature=0.6, model=llm_model)

result = llm.invoke("What is the capital of France?")
print(result.content)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)

conversation.invoke(input="hey my name is eddie")
conversation.predict(input="Why is the sky blue?")
conversation.predict(input="If phenomenon called Rayleigh didn't exist, what color would the sky be?")
conversation.predict(input="What's my name?")

print(memory.load_memory_variables({}))