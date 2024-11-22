import os
import tempfile
import pathlib
import getpass
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma 
from langchain_community.document_loaders import Docx2txtLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 

os.environ["GOOGLE_API_KEY"] = "AIzaSyCOVPvbV9NEg2dYAsP5i98bQnsGQW_qWMc" 

from langchain_google_genai import ChatGoogleGenerativeAI 

llm = ChatGoogleGenerativeAI( 
    model="gemini-1.5-pro-latest", 
    temperature=0, 
    max_tokens=None, 
    timeout=None, 
    max_retries=2, 
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vectorstore_from_docx(docx_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(docx_file.read())
            temp_file_path = temp_file.name

        loader = Docx2txtLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data"
        )
        os.remove(temp_file_path)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),

        ("system", """Act as a PreCollege AI assistant dedicated to guiding students through their JEE Mains journey. Your goal is to provide personalized, accurate, and interactive advice for students seeking college admissions guidance. Tailor your responses to address students' individual needs, including:

1. College Selection and Counseling: Help students identify colleges they qualify for based on their JEE Mains rank and preferences, including NITs, IIITs, GFTIs, and private institutions. Consider factors like location, course offerings, placement records, and fees.

2. Admission Process Guidance: Clarify the college admission procedures, including JoSAA counseling, spot rounds, document verification, and category-specific quotas (if applicable).

3. Career and Branch Selection Advice: Assist students in making informed decisions about their preferred engineering branches based on interest, market trends, and scope of opportunities.

Interactive Sessions: Engage students in Q&A sessions to answer their doubts related to preparation, counseling, and career choices.

Maintain a professional and friendly tone. Use your expertise to ensure students receive relevant and clear information. Provide examples, stats, and other insights to support your advice wherever needed""")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
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

st.set_page_config(page_title="College Data Chatbot")
st.title("College Data Chatbot")

with st.sidebar:
    st.header("Settings")
    docx_files = st.file_uploader("Upload College Data Document", accept_multiple_files=True)

    if not docx_files:
        st.info("Please upload a .docx file")
    else:
        docx_file = docx_files[0]

        if "docx_name" in st.session_state and st.session_state.docx_name != docx_file.name:
            st.session_state.pop("vector_store", None)
            st.session_state.pop("chat_history", None)
        
        if st.button("Preprocess"):
            st.session_state.vector_store = get_vectorstore_from_docx(docx_file)
            if st.session_state.vector_store:
                st.session_state.docx_name = docx_file.name
                st.success("Document processed successfully!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"author": "assistant", "content": "Hello, I am a bot. How can I help you?"}
    ]

if st.session_state.get("vector_store") is None:
    st.info("Please preprocess the document by clicking the 'Preprocess' button in the sidebar.")
else:
    for message in st.session_state.chat_history:
        if message["author"] == "assistant":
            with st.chat_message("system"):
                st.write(message["content"])
        elif message["author"] == "user":
            with st.chat_message("human"):
                st.write(message["content"])

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