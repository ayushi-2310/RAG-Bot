import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key from .env or Streamlit secrets
load_dotenv()
api_key = st.secrets["API_KEY"]

if not api_key:
    st.error("❌ GEMINI_API_KEY not found! Please set it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Function to load database (without cache decorator)
def load_db(pdf_path="hotel-rules-en.pdf"):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")
        return None, None

# Function to get available models and select the best one
def get_best_available_model():
    """Try different Gemini models in order of preference"""
    model_priority = [
        "gemini-2.0-flash-exp",  # Latest experimental
        "gemini-2.0-flash",      # Current stable
        "gemini-1.5-flash",      # Fallback
        "gemini-1.5-pro",        # Fallback
        "gemini-pro"             # Legacy fallback
    ]
    
    for model_name in model_priority:
        try:
            model = genai.GenerativeModel(model_name)
            # Test with a simple prompt
            test_response = model.generate_content("Hello")
            st.success(f"✅ Using model: {model_name}")
            return model
        except Exception as e:
            st.warning(f"⚠️ Model {model_name} not available: {str(e)}")
            continue
    
    st.error("❌ No available Gemini models found!")
    return None

# Build UI
st.title("🏨 Hotel Policy RAGBot")
st.write("Ask any question based on the uploaded hotel policy document.")

# Load database using session state for caching
if 'vectorstore' not in st.session_state:
    with st.spinner("📦 Loading and indexing hotel policy..."):
        vectorstore, embeddings = load_db()
        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            st.session_state.embeddings = embeddings
            st.success("✅ Hotel policy loaded and ready!")
        else:
            st.error("❌ Failed to load hotel policy document!")
            st.stop()
    st.rerun()
else:
    st.success("✅ Hotel policy loaded and ready!")

# Initialize model if not already done
if 'model' not in st.session_state:
    with st.spinner("🤖 Initializing AI model..."):
        model = get_best_available_model()
        if model is not None:
            st.session_state.model = model
        else:
            st.error("❌ Failed to initialize AI model!")
            st.stop()

# Input box
user_question = st.text_input("❓ Enter your question here:")

if user_question:
    with st.spinner("Searching and generating answer..."):
        try:
            # Retrieve relevant documents
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", k=3)
            relevant_docs = retriever.get_relevant_documents(user_question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Enhanced prompt for better responses
            prompt = f"""
You are a helpful hotel policy assistant. Use the following document content to answer the question accurately and concisely.

Document Context:
{context}

Question: {user_question}

Instructions:
- Answer based only on the provided document content
- If the information is not in the document, say "I don't have information about that in the hotel policy document"
- Be specific and cite relevant policy details
- Keep your response clear and helpful

Answer:
"""

            # Generate response
            response = st.session_state.model.generate_content(prompt)
            
            # Display results
            st.markdown("### 🧠 Answer:")
            st.write(response.text)

            # Show source documents
            with st.expander("📄 Source Text"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("💡 Try rephrasing your question or check your API key.")

# Add footer with model info
if 'model' in st.session_state:
    st.markdown("---")
    st.caption("🤖 Powered by Google Gemini AI")



