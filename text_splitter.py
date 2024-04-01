import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

with open("./data/i-have-a-dream.txt") as paper:
    speech = paper.read()
    
text_splitter = CharacterTextSplitter(
    
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len
)

texts = text_splitter.create_documents([speech])
print(texts[0])