import os 
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)

pf_loader = PyPDFLoader('./data/fighter.pdf')
documents = pf_loader.load()

mma_template = """
    you are going to be answering questions based on the data provided in the PDF document about the fighter {pages}.
"""

mma_prompt = PromptTemplate(input_variables=["pages"], template=mma_template)

chain = load_qa_chain(llm, verbose=True, prompt_template=mma_prompt)

query = 'Who is welterweight champion?'
response = chain.run(input_documents=documents,
                     question=query)
print(response)
