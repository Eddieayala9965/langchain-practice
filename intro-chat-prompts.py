import os 
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables and set API key
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4"

# Initialize LangChain's ChatOpenAI with the model and temperature settings
# model and model name seem to be interganable

chat_model = ChatOpenAI(api_key=openai_api_key, temperature=0.7, model=llm_model)

def get_completion(prompt):
    # Invoke the chat model with the prompt encapsulated in a message
    response = chat_model.invoke(prompt)
    
    # Extract and return the text content from the response
    # Note: This step might need adjustment based on the actual response structure from chat_model.invoke
    return response.content

customer_review = """
I ordered a pair of shoes online and they arrived in the wrong size. I want to return them and get the correct size, but the return process is so complicated. I've been trying to get in touch with customer service for days, but no one has responded to my emails or calls. I'm so frustrated and disappointed with this experience. I will never shop here again.
"""  
print(customer_review)
print("====================================")

prompt = f"""
Rewrite the following customer review in a more professional tone:
{customer_review}
"""

rewrite_review = get_completion(prompt)
print(rewrite_review)

print("====================================")




# using langchain and prompt templates still ChatAPi, this for templates

chat_model = ChatOpenAI(temperature=0.7, model=llm_model)

template_string = """ 
    Translate the follwing text {customer_review} to French in a polite tone.
    And teh comapny name is {company_name}
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

translation_message = prompt_template.format_messages(
    customer_review = customer_review,
    company_name = "Google"
)

response = chat_model.invoke(translation_message)
print(response.content)
