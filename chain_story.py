import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Load the environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the model to use
llm_model = "gpt-4"

# Initialize the OpenAI and ChatOpenAI instances with specific temperatures
chat = ChatOpenAI(temperature=0.9, model=llm_model)
open_ai = OpenAI(temperature=0.7)

# Define the template for the lullaby
template = """
 As a children's book writer, please come up with a simple and short (90 words)
 lullaby based on the location {location} and the main character {name}.
 
 STORY:
"""

# Create a PromptTemplate instance with the specified template
prompt = PromptTemplate(input_variables=["location", "name"], template=template)

# Initialize LLMChain with the OpenAI instance and the prepared prompt
chain_story = LLMChain(llm=open_ai, prompt=prompt, verbose=True)

# invoke the run method with the input variables

result = chain_story.invoke({"location": "Zanzibar", "name": "Maya"})  # Adjust the method name as needed
print(result["text"])
