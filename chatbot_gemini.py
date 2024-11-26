import os  
import streamlit as st  
import google.generativeai as genai  
# from langchain_openai import OpenAI /
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_google_genai import ChatGoogleGenerativeAI  
# from langchain_openai import OpenAIEmbeddings  
from langchain_community.document_loaders import Docx2txtLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from dotenv import load_dotenv  
from langchain.embeddings import HuggingFaceEmbeddings
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

# Retrieve OpenAI API key from the .env file
GOOGLE_API_KEY = "AIzaSyAytkzRS0Xp0pCyo6WqKJ4m1o330bF-gPk"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key not found. Please set it in the .env file.")

# Set OpenAI API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Streamlit app configuration
st.set_page_config(page_title="College Data Chatbot", layout="centered")
st.title("PreCollege Chatbot GEMINI+ HuggingFace Embeddings")

# Initialize OpenAI LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,  # Slightly higher for varied responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize embeddings using OpenAI
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_preprocessed_vectorstore():
    try:
        loader = Docx2txtLoader("./Updated_structred_aman.docx")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000, 
            chunk_overlap=1000)
        
        document_chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data32"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    """Creates a history-aware retriever chain."""
    retriever = vector_store.as_retriever()

    # Define the prompt for the retriever chain
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", """Given the chat history and the latest user question, which might reference context in the chat history, 
formulate a standalone question that can be understood without the chat history.
If the question is directly addressed within the provided document, provide a relevant answer. 
If the question is not explicitly addressed in the document, return the following message: 
'This question is beyond the scope of the available information. Please contact your mentor for further assistance.'
Do NOT answer the question directly, just reformulate it if needed and otherwise return it as is.""")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    """Creates a conversational chain using the retriever chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """ Hello! I'm your PreCollege AI assistant. I'll guide you through your JEE Mains journey, providing personalized advice and support.
        To get started, please share your JEE Mains rank and preferred engineering branches or colleges. 
        I'll provide information and suggestions based on our database.Please note that I'll only provide information available within our database, ensuring accuracy and relevance. Let's get started!"""
         "\n\n"
         "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_chain(retriever_chain)
    
    formatted_chat_history = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            formatted_chat_history.append({"author": "user", "content": message.content})
        elif isinstance(message, SystemMessage):
            formatted_chat_history.append({"author": "assistant", "content": message.content})
    
    response = conversation_rag_chain.invoke({
        "chat_history": formatted_chat_history,
        "input": user_query
    })
    
    return response['answer']

# Load the preprocessed vector store from the local directory
st.session_state.vector_store = load_preprocessed_vectorstore()

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"author": "assistant", "content": "Hello, I am Precollege. How can I help you?"}
    ]

# Main app logic
if st.session_state.get("vector_store") is None:
    st.error("Failed to load preprocessed data. Please ensure the data exists in './data' directory.")
else:
    # Display chat history
    with st.container():
        for message in st.session_state.chat_history:
            if message["author"] == "assistant":
                with st.chat_message("system"):
                    st.write(message["content"])
            elif message["author"] == "user":
                with st.chat_message("human"):
                    st.write(message["content"])

    # Add user input box below the chat
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your message here...", key="user_input")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_query:
            # Get bot response
            response = get_response(user_query)
            st.session_state.chat_history.append({"author": "user", "content": user_query})
            st.session_state.chat_history.append({"author": "assistant", "content": response})

            # Rerun the app to refresh the chat display
            st.rerun()


""""""
