import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Load environment variables
load_dotenv()

def main():
    # Streamlit app title
    st.title("Home Automation Chatbot")
    st.write("Command the chatbot :")

    # Load the Hugging Face API token
    huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not huggingface_api_key:
        st.error("Hugging Face API token not found. Please set it in the `.env` file.")
        return
    
    # Define the Hugging Face model endpoint
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.7,
            token=huggingface_api_key
        )
    except Exception as e:
        st.error(f"Error initializing HuggingFaceEndpoint: {e}")
        return
    
    # Define the prompt template
    template = """You are a home automation device. Respond to the user's request in a helpful and concise way and even if its a one statement line just generate that only and dont answer anything which is not related to home automation .

    User: {user_input}
    Device: """
    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # User input
    user_input = st.text_input("User Input:", placeholder="Type your request here...")
    
    # Generate response on button click
    if st.button("Submit"):
        if user_input.strip():
            try:
                # Generate a response from the model
                response = llm_chain.invoke({"user_input": user_input})
                st.text_area("Device Response:", value=response['text'], height=200)
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
