import os  
import streamlit as st  
import google.generativeai as genai  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  
from langchain_community.document_loaders import Docx2txtLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.embeddings import HuggingFaceEmbeddings
from bert_score import score
from sklearn.metrics import f1_score
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

# Retrieve Google API Key
GOOGLE_API_KEY = "AIzaSyAytkzRS0Xp0pCyo6WqKJ4m1o330bF-gPk"
if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key not found. Please set it in the .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit configuration
st.set_page_config(page_title="College Data Chatbot", layout="centered")
st.title("PreCollege Chatbot GEMINI+ HuggingFace Embeddings")

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load vector store
def load_preprocessed_vectorstore():
    try:
        loader = Docx2txtLoader("./Updated_structred_aman.docx")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000, 
            chunk_overlap=1000
        )
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

# Evaluation Metrics
def calculate_recall_at_k(retrieved_docs, relevant_docs, k=5):
    retrieved_top_k = retrieved_docs[:k]
    relevant_in_top_k = len(set(retrieved_top_k).intersection(set(relevant_docs)))
    total_relevant = len(relevant_docs)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

def calculate_bertscore(generated_responses, reference_responses):
    P, R, F1 = score(generated_responses, reference_responses, lang="en", rescale_with_baseline=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

def calculate_f1_score(generated_response, relevant_text):
    generated_tokens = set(generated_response.split())
    relevant_tokens = set(relevant_text.split())
    intersection = generated_tokens.intersection(relevant_tokens)
    
    precision = len(intersection) / len(generated_tokens) if len(generated_tokens) > 0 else 0
    recall = len(intersection) / len(relevant_tokens) if len(relevant_tokens) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1

# Context Retriever Chain
def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", """Given a chat history and the latest user question, 
        reformulate it as a standalone question without using chat history. 
        Do NOT answer it, just reformulate.""")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Hello! I'm your PreCollege AI assistant. I'll guide you through your JEE Mains journey.
        To get started, share your JEE Mains rank and preferred engineering branches or colleges."""),
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

# Initialize vector store and metrics
st.session_state.vector_store = load_preprocessed_vectorstore()
if "metrics" not in st.session_state:
    st.session_state.metrics = {"recall_at_5": [], "bert_scores": [], "f1_scores": []}

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"author": "assistant", "content": "Hello, I am Precollege. How can I help you?"}
    ]

# Main app logic
if st.session_state.get("vector_store") is None:
    st.error("Failed to load preprocessed data. Ensure the data exists in './data' directory.")
else:
    with st.container():
        for message in st.session_state.chat_history:
            if message["author"] == "assistant":
                with st.chat_message("system"):
                    st.write(message["content"])
            elif message["author"] == "user":
                with st.chat_message("human"):
                    st.write(message["content"])

    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your message here...", key="user_input")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_query:
            # Get response
            response = get_response(user_query)
            st.session_state.chat_history.append({"author": "user", "content": user_query})
            st.session_state.chat_history.append({"author": "assistant", "content": response})

            # Dummy relevant docs for metrics demonstration
            retrieved_docs = ["doc1", "doc2", "doc3"]  # Replace with actual IDs from retriever
            relevant_docs = ["doc1", "doc4"]  # Replace with ground truth IDs
            recall_at_5 = calculate_recall_at_k(retrieved_docs, relevant_docs)
            st.session_state.metrics["recall_at_5"].append(recall_at_5)

            # Dummy reference and relevant text
            reference_response = "Gold-standard answer here."
            bert_scores = calculate_bertscore([response], [reference_response])
            st.session_state.metrics["bert_scores"].append(bert_scores["f1"])

            f1_score_value = calculate_f1_score(response, "Relevant text here")
            st.session_state.metrics["f1_scores"].append(f1_score_value)

            # Display evaluation metrics
            st.write("Evaluation Metrics:")
            st.write(f"Recall@5: {recall_at_5:.2f}")
            st.write(f"BERTScore F1: {bert_scores['f1']:.2f}")
            st.write(f"Faithfulness F1: {f1_score_value:.2f}")

            st.rerun()
