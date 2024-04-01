import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Load environment variables and set OpenAI API key
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI with the specified model
chat = ChatOpenAI(temperature=0.9, model="gpt-4")

# Story generation template
story_template = """ 
As a children's book writer, please come up with a simple and short (90 words)
lullaby based on the location {location} and the main character {name}

STORY:
"""

# Translation template
translate_template = """
Translate the story into {language}. Make sure the language is simple and fun.

{story}

TRANSLATION:
"""

# Setup the story prompt template
story_prompt = PromptTemplate(input_variables=["location", "name"], template=story_template)

# Setup the translation prompt template
translate_prompt = PromptTemplate(input_variables=["story", "language"], template=translate_template)

# Create the LLMChain for generating the story
chain_story = LLMChain(llm=chat, prompt=story_prompt, output_key="story")

# Create the LLMChain for translating the story
chain_translate = LLMChain(llm=chat, prompt=translate_prompt, output_key="translated")

# Setup SequentialChain with the LLMChain instances
overall_chain = SequentialChain(
    chains=[chain_story, chain_translate],
    input_variables=["location", "name", "language"],
    output_variables=["story", "translated"], 
    verbose=True 
    # Assuming you want both outputs
)

# Execute the overall chain with the provided inputs using invoke
response = overall_chain.invoke({"location": "Magical", "name": "Karyna", "language": "Russian"})

# Assuming response structure contains both outputs, adjust based on actual structure
# This is a placeholder print statement; you'll need to adjust based on your response handling
print(f"English Version ====> { response['story']} \n \n")
print(f"Translated Version ====> { response['translated']}")
