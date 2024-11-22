import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Retrieve OpenAI API key from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit app configuration
st.set_page_config(page_title="College Data Chatbot", layout="centered")
st.title("PreCollege Chatbot")

# Initialize OpenAI LLM
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
)

# Initialize embeddings using OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_preprocessed_vectorstore():
    try:
        loader = Docx2txtLoader("./Updated_structred_aman.docx")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000, 
            chunk_overlap=200)
        
        document_chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data11"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

import logging

# Function to create the retriever and prompt chain
def get_context_retriever_chain(vector_store):
    """Creates a context-aware retriever and prompt chain."""
    retriever = vector_store.as_retriever(k=3)  # Hybrid retrieval for better results

    rag_prompt = PromptTemplate(
        template="""
        Act as a PreCollege AI assistant dedicated to guiding students through their JEE Mains journey. Your goal is to provide personalized, accurate, and interactive advice for students seeking college admissions guidance. Tailor your responses to address students' individual needs, including:

        1. College Selection and Counseling: Help students identify colleges they qualify for based on their JEE Mains rank and preferences, including IIITs institutions. Consider factors like location, course offerings, placement records, and fees.

        2. Admission Process Guidance: Clarify the college admission procedures, including JoSAA counseling, spot rounds, document verification, and category-specific quotas (if applicable).

        3. Career and Branch Selection Advice: Assist students in making informed decisions about their preferred engineering branches based on interest, market trends, and scope of opportunities.

        Interactive Sessions: Engage students in Q&A sessions to answer their doubts related to preparation, counseling, and career choices.

        Maintain a professional and friendly tone. Use your expertise to ensure students receive relevant and clear information. Provide examples, stats, and other insights to support your advice wherever needed.

        QUESTION: {question} 
        CONTEXT: {context} 
        Answer in a detailed yet concise manner, also highlight relevant information and do not give unnecessary information or negative responses:
        """,
        input_variables=["question", "context"],
    )

    rag_prompt_chain = rag_prompt | llm | StrOutputParser()

    return retriever, rag_prompt_chain


def get_response(user_query):
    """Processes the user query and generates a response."""
    # Define a set of common greetings
    greetings = ["hi", "hello", "hey", "greetings", "hi there"]

    # Check if the user query is a greeting
    if user_query.lower().strip() in greetings:
        return "Hello! How can I assist you with your college search today?"

    # Ensure the vector store is initialized
    if "vector_store" not in st.session_state:
        logging.error("Vector store is not initialized in session state.")
        return "Vector store is not initialized. Please preprocess the document first."

    retriever, rag_prompt_chain = get_context_retriever_chain(st.session_state.vector_store)

    # Format chat history from session state
    formatted_chat_history = []
    for message in st.session_state.chat_history:
        if message["author"] == "user":
            formatted_chat_history.append({"author": "user", "content": message["content"]})
        elif message["author"] == "assistant":
            formatted_chat_history.append({"author": "assistant", "content": message["content"]})

    try:
        # Retrieve context
        context = retriever.invoke(user_query)
        logging.info(f"Retrieved context: {context}")

        if not context:
            logging.error("No relevant context retrieved.")
            return "I couldn't retrieve relevant information. Please try a different query."

        # Generate response
        response = rag_prompt_chain.invoke({
            "chat_history": formatted_chat_history,
            "question": user_query,
            "context": context
        })
        logging.info(f"Generated response: {response}")

        # Check the response format
        if isinstance(response, dict) and "answer" in response:
            return response["answer"]
        elif isinstance(response, str):  # Handle raw string outputs
            return response
        else:
            logging.error(f"Unexpected response format: {response}")
            return "Unexpected error occurred. Please try again later."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Sorry, I encountered an issue while processing your request. Please try again later."



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
















# import os
# import streamlit as st
# from langchain_openai import OpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import Docx2txtLoader 
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from dotenv import load_dotenv


# # Retrieve OpenAI API key from the .env file
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# # Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# # Streamlit app configuration
# st.set_page_config(page_title="College Data Chatbot", layout="centered")
# st.title("PreCollege Chatbot")

# # Initialize OpenAI LLM
# llm = OpenAI(
#     model="gpt-3.5-turbo-instruct",
#     temperature=0,
# )

# # Initialize embeddings using OpenAI
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# def load_preprocessed_vectorstore():
#     try:
#         loader = Docx2txtLoader("./Updated_structred_aman.docx")
#         documents = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", ". ", " ", ""],
#             chunk_size=3000, 
#             chunk_overlap=200)
        
#         document_chunks = text_splitter.split_documents(documents)

#         vector_store = Chroma.from_documents(
            
#             embedding=embeddings,
#             documents=document_chunks,
#             persist_directory="./data11"
#         )
#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         return None

# def get_context_retriever_chain(vector_store):
#     """Creates a history-aware retriever chain."""
#     retriever = vector_store.as_retriever()

#     # Define the prompt for the retriever chain
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("system", "You are a PreCollege AI assistant helping students with JEE Mains college guidance. Answer interactively and provide relevant, accurate information.")
#     ])

#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#     return retriever_chain

# def get_conversational_chain(retriever_chain):
#     """Creates a conversational chain using the retriever chain."""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions based on the context below:\n\n{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}")
#     ])

#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_query):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_chain(retriever_chain)
    
#     formatted_chat_history = []
#     for message in st.session_state.chat_history:
#         if isinstance(message, HumanMessage):
#             formatted_chat_history.append({"author": "user", "content": message.content})
#         elif isinstance(message, SystemMessage):
#             formatted_chat_history.append({"author": "assistant", "content": message.content})
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": formatted_chat_history,
#         "input": user_query
#     })
    
#     return response['answer']

# # Load the preprocessed vector store from the local directory
# st.session_state.vector_store = load_preprocessed_vectorstore()

# # Initialize chat history if not present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         {"author": "assistant", "content": "Hello, I am Precollege. How can I help you?"}
#     ]

# # Main app logic
# if st.session_state.get("vector_store") is None:
#     st.error("Failed to load preprocessed data. Please ensure the data exists in './data' directory.")
# else:
#     # Display chat history
#     with st.container():
#         for message in st.session_state.chat_history:
#             if message["author"] == "assistant":
#                 with st.chat_message("system"):
#                     st.write(message["content"])
#             elif message["author"] == "user":
#                 with st.chat_message("human"):
#                     st.write(message["content"])

#     # Add user input box below the chat
#     with st.container():
#         with st.form(key="chat_form", clear_on_submit=True):
#             user_query = st.text_input("Type your message here...", key="user_input")
#             submit_button = st.form_submit_button("Send")

#         if submit_button and user_query:
#             # Get bot response
#             response = get_response(user_query)
#             st.session_state.chat_history.append({"author": "user", "content": user_query})
#             st.session_state.chat_history.append({"author": "assistant", "content": response})

#             # Rerun the app to refresh the chat display
#             st.rerun()




















# import os 
# import tempfile 
# import streamlit as st 
# from langchain_openai import OpenAI 
# from langchain_openai import OpenAIEmbeddings 
# from langchain_community.vectorstores import Chroma 
# from langchain_community.document_loaders import Docx2txtLoader 
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
# from langchain_core.messages import HumanMessage, SystemMessage 
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain  
# from langchain.chains.combine_documents import create_stuff_documents_chain 

# # Load environment variables for API keys 
# # load_dotenv() 
# import os 
# os.environ["OPENAI_API_KEY"]="sk-HQoHO1UganCjwF-tK2Hs-0wmwUHmVdiZIVwa_2SYBuT3BlbkFJSiebrtoqIo83LPDi-LaPHeLqndbP3I9tguwSnw3AMA" 

# # Initialize OpenAI LLM 
# llm = OpenAI( 
#     model="gpt-3.5-turbo-instruct",
#     temperature=0, 
# )

# # Initialize embeddings using OpenAI 
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# def get_vectorstore_from_docx(docx_file):
#     """Processes a .docx file to create a vector store.""" 
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file: 
#             temp_file.write(docx_file.read())
#             temp_file_path = temp_file.name

#         loader = Docx2txtLoader(temp_file_path)
#         documents = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
#         document_chunks = text_splitter.split_documents(documents)

#         vector_store = Chroma.from_documents(
#             embedding=embeddings,
#             documents=document_chunks,
#             persist_directory="./data1"
#         )
#         os.remove(temp_file_path)
#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         return None

# def get_context_retriever_chain(vector_store):
#     """Creates a history-aware retriever chain."""
#     retriever = vector_store.as_retriever()
    
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("system", "You are a PreCollege AI assistant helping students with JEE Mains college guidance. Answer interactively and provide relevant, accurate information.")
#     ])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#     return retriever_chain

# def get_conversational_chain(retriever_chain):
#     """Creates a conversational chain using the retriever chain."""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions based on the context below:\n\n{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}")
#     ])
    
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_query):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_chain(retriever_chain)
    
#     formatted_chat_history = []
#     for message in st.session_state.chat_history:
#         if isinstance(message, HumanMessage):
#             formatted_chat_history.append({"author": "user", "content": message.content})
#         elif isinstance(message, SystemMessage):
#             formatted_chat_history.append({"author": "assistant", "content": message.content})
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": formatted_chat_history,
#         "input": user_query
#     })
    
#     return response['answer']

# # Streamlit app configuration
# st.set_page_config(page_title="College Data Chatbot")
# st.title("College Data Chatbot")

# # Sidebar for document upload and automatic processing
# with st.sidebar:
#     st.header("Upload College Data Document")
#     docx_file = st.file_uploader("Upload a .docx file")

#     if docx_file:
#         # Automatically process the uploaded file
#         st.session_state.vector_store = get_vectorstore_from_docx(docx_file)
#         if st.session_state.vector_store:
#             st.session_state.docx_name = docx_file.name
#             st.success("Document processed successfully!")

# # Initialize chat history if not present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         {"author": "assistant", "content": "Hello, I am precollege. How can I help you?"}
#     ]

# # Main chat section
# if st.session_state.get("vector_store") is None:
#     st.info("Please upload and process a .docx file to get started.")
# else:
#     # Display the chat history first
#     with st.container():
#         for message in st.session_state.chat_history:
#             if message["author"] == "assistant":
#                 with st.chat_message("system"):
#                     st.write(message["content"])
#             elif message["author"] == "user":
#                 with st.chat_message("human"):
#                     st.write(message["content"])

#     # User input at the bottom of the chat
#     with st.container():
#         with st.form(key="chat_form", clear_on_submit=True):
#             user_query = st.text_input("Type your message here...", key="user_input")
#             submit_button = st.form_submit_button("Send")

#         if submit_button and user_query:
#             # Process the user query and get the bot's response
#             response = get_response(user_query)
#             st.session_state.chat_history.append({"author": "user", "content": user_query})
#             st.session_state.chat_history.append({"author": "assistant", "content": response})

#             # Scroll to the bottom of the chat
#             # st.experimental_rerun()
