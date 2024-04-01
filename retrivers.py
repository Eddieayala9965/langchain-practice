import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
embeddings = OpenAIEmbeddings()


loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
splits = text_splitter.split_documents(docs)

persist_directory = './data/db/chroma/'
# !rm -rf ./data/db/chroma  # remove old database files if any
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings, # openai embeddings
    persist_directory= persist_directory

)
vectorstore.persist()

vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})  
docs = retriever.get_relevant_documents("Tell me more about ReAct prompting")  
# print(retriever.search_type)
print(docs[0].page_content)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
    
)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        
query = "tell me more about ReAct prompting"
llm_response = qa_chain(query)
print(process_llm_response(llm_response=llm_response))