# 🏨 Hotel Policy RAGBot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on hotel policy documents using Google Gemini AI and Streamlit.

## 🚀 Features

- **PDF Document Processing**: Automatically loads and processes hotel policy PDFs
- **Intelligent Chunking**: Splits documents into optimal chunks for better retrieval
- **Semantic Search**: Uses HuggingFace embeddings with FAISS for fast similarity search
- **Smart Model Selection**: Automatically detects and uses the best available Gemini model
- **Interactive UI**: Clean Streamlit interface with real-time responses
- **Source Transparency**: Shows the exact document sections used to generate answers
- **Error Handling**: Robust error handling with helpful user feedback

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini AI (2.0 Flash / 1.5 Flash)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector Database**: FAISS
- **Document Processing**: PyMuPDF via LangChain
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter

## 📋 Prerequisites

- Python 3.8+ (for local development)
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- Hotel policy PDF document
- GitHub account (for Streamlit Cloud deployment)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hotel-policy-ragbot.git
   cd hotel-policy-ragbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   **For Local Development:**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```
   
   **For Streamlit Cloud Deployment:**
   ```toml
   # Create .streamlit/secrets.toml file
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

5. **Add your hotel policy PDF**
   - Place your PDF file in the project directory
   - Name it `hotel-rules-en.pdf` or update the path in the code

## 🚀 Usage

### Local Development

1. **Start the application**
   ```bash
   streamlit run ragbot.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - Wait for the PDF to be processed and indexed
   - Start asking questions about the hotel policy!

### Streamlit Cloud Deployment

1. **Deploy to Streamlit Cloud**
   - Fork this repository to your GitHub account
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Connect your GitHub repository
   - Set up your secrets in the Streamlit Cloud dashboard
   - Your app will be deployed automatically!

2. **Live Demo**
   - 🚀 **[Try the Live Demo](https://hotel-rag-bot.streamlit.app/)**
   - Experience the RAGBot in action without any setup!

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
python-dotenv>=1.0.0
google-generativeai>=0.3.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
faiss-cpu>=1.7.4
PyMuPDF>=1.23.0
sentence-transformers>=2.2.2
```

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your configuration:
   - **Local**: Add to `.env` file
   - **Streamlit Cloud**: Add to secrets in your Streamlit Cloud dashboard

### Setting up Streamlit Cloud Secrets

1. Go to your [Streamlit Cloud dashboard](https://share.streamlit.io/)
2. Select your deployed app
3. Go to "Settings" → "Secrets"
4. Add your secrets in TOML format:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

## 📖 How It Works

### 1. Document Processing Pipeline
```
PDF → Text Extraction → Chunking → Embeddings → Vector Store
```

### 2. Query Processing Pipeline
```
User Question → Embedding → Similarity Search → Context Retrieval → LLM Generation → Response
```

### 3. Key Components

- **Document Loader**: PyMuPDF extracts text from PDF files
- **Text Splitter**: Creates 500-character chunks with 50-character overlap
- **Embeddings**: all-MiniLM-L6-v2 converts text to 384-dimensional vectors
- **Vector Store**: FAISS provides fast similarity search
- **LLM**: Gemini generates contextual responses

## ⚙️ Configuration

### Chunking Parameters
```python
chunk_size=500      # Characters per chunk
chunk_overlap=50    # Overlap between chunks
```

### Retrieval Parameters
```python
k=3  # Number of relevant chunks to retrieve
search_type="similarity"  # Search strategy
```

### Model Priority
The system tries models in this order:
1. `gemini-2.0-flash-exp` (Latest experimental)
2. `gemini-2.0-flash` (Current stable)
3. `gemini-1.5-flash` (Fallback)
4. `gemini-1.5-pro` (Fallback)
5. `gemini-pro` (Legacy fallback)

## 🎨 Customization

### Change PDF File
```python
# In load_db function
def load_db(pdf_path="your-policy-document.pdf"):
```

### Adjust Chunk Size
```python
# For longer contexts
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# For shorter, more precise chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
```

### Modify Retrieval Count
```python
# Retrieve more context
retriever = vectorstore.as_retriever(search_type="similarity", k=5)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure you have a valid Gemini API key
   - Check if your Google Cloud project has Gemini API enabled
   - Try using a different model name

2. **PDF loading fails**
   - Verify the PDF file exists and is readable
   - Check file permissions
   - Try with a different PDF file

3. **Slow performance**
   - Reduce chunk_size for faster processing
   - Use fewer retrieved chunks (k=2 instead of k=3)
   - Consider using a smaller embedding model

4. **Empty responses**
   - Check if the PDF content is extractable
   - Verify the question is related to the document content
   - Try rephrasing the question

5. **Deployment Issues**
   - **Local**: Ensure `.env` file exists and contains valid API key
   - **Streamlit Cloud**: Check secrets are properly configured in dashboard
   - **File paths**: Ensure PDF file is in the correct location in your repository

## 🚀 Performance Tips

1. **Optimize Chunking**
   - Balance chunk size vs. retrieval accuracy
   - Adjust overlap based on document structure

2. **Improve Retrieval**
   - Use more specific questions
   - Include relevant keywords from the document

3. **Model Selection**
   - Use `gemini-2.0-flash` for best performance
   - Consider `gemini-1.5-flash` for cost optimization

## 📝 Example Questions

- "What is the check-in time?"
- "What are the cancellation policies?"
- "Are pets allowed in the hotel?"
- "What amenities are included?"
- "What is the smoking policy?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for the RAG pipeline components
- [Google AI](https://ai.google/) for the Gemini API
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the [troubleshooting section](#-troubleshooting)
- Review the [Google AI documentation](https://ai.google.dev/docs)

---
