import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-4"

chat = ChatOpenAI(temperature=0.9, model=llm_model)
open_ai = OpenAI(temperature=0.7)


# LLMChain
prompt = PromptTemplate(
    input_variables=["language"],
    template="How do you say good morning in {language}"
)

chain = LLMChain(llm=chat, prompt=prompt)
print(chain.invoke(input="German"))