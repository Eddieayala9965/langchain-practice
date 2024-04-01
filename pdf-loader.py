import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)

loader = PyPDFLoader("./data/fighter.pdf")
pages = loader.load()


mma_template = """
    you are going to be answering questions based on the data provided in the PDF document about the fighter {pages}.
"""

mma_prompt = PromptTemplate(input_variables=["pages"], template=mma_template)

chain_mma = LLMChain(llm=llm, prompt=mma_prompt, output_key="mma")

response = chain_mma.invoke({"pages": pages})

print(response["mma"])



