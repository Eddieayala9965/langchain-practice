import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
embeddings = OpenAIEmbeddings()

# text1 = "Math is a great subject to study."
# text2 = "Dogs are friendly when they are happy and well fed."
# text3 = "Physics is not one of my favorite subjects."


# text1 = "Dog"
# text2 = "Cat"
# text3 = "Rock"

# embed1 = embeddings.embed_query(text1)
# embed2 = embeddings.embed_query(text2)
# embed3 = embeddings.embed_query(text3)
# import numpy as np
# similarity = np.dot(embed1, embed2)
# print(f"Similarity %:{similarity*100}")


loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

# 2. Split the document into chunks
# Split

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
# print(len(splits))
# =============== ==================== # 

presist_directory = "./data/db/chroma"
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=presist_directory)
# print(vector_store._collection.count())


query = "what do they say about ReAct prompting method?"

docs_resp = vector_store.similarity_search(query=query, k=3)

print(len(docs_resp))
print(docs_resp[0].page_content)

vector_store.persist()
