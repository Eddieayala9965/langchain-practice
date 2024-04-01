import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)


with open("./data/i-have-a-dream.txt") as paper:
    speech = paper.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 40,
    chunk_overlap = 12,
    length_function = len,
    add_start_index=True
)

docs = text_splitter.create_documents([speech])

print(len(docs))
print(f"Doc 1: {docs[0]}")
print(f"Doc 2: {docs[1]}")


