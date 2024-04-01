import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import streamlit as st

# Load environment variables and set OpenAI API key
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI with the specified model
chat = ChatOpenAI(temperature=0.9, model="gpt-4")



def generate_lullaby(location, name, language): 
    
    template = """
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on location {location} and the main character {name}
        
        STORY: 
    """
    template_update = """
    Translate the {story} into {language}.  Make sure  # Corrected placeholder
    the language is simple and fun.

    TRANSLATION:
    """
    prompt = PromptTemplate(input_variables=["location", "name"], template=template)
    
    prompt_translate = PromptTemplate(input_variables=["story", "language"], template=template_update)  
    
    chain_story = LLMChain(llm=chat, prompt=prompt, output_key="story")
    
    chain_translate = LLMChain(llm=chat, prompt=prompt_translate, output_key="translated")
    
    overall_chain = SequentialChain(
        chains=[chain_story, chain_translate], 
        input_variables=["location", "name", "language"], 
        output_variables=["story", "translated"]
    )
    response = overall_chain.invoke({"language": language, "location": location, "name": name})  
    
    return response

def main():
    st.set_page_config(page_title="Lullaby Generator", page_icon="🌙", layout="centered")
    st.title("Let AI Generate a Lullaby for You!")
    st.header("Get Started...")
    
    location_input = st.text_input("Enter the location: ") 
    main_character_input = st.text_input("Enter the main character: ")
    language_input = st.text_input("Enter the language: ")
    
    submit_button = st.button("Generate Lullaby")
    if location_input and main_character_input and language_input and submit_button:
        with st.spinner("Generating Lullaby..."):
            response = generate_lullaby(location=location_input, name=main_character_input, language=language_input)  
            
            with st.expander("English Version"):
                st.write(response["story"])
            with st.expander(f"{language_input.capitalize()} Version"):
                st.write(response["translated"])
        
        st.success("Lullaby Generated!")
      







if __name__ == '__main__':
    main() 