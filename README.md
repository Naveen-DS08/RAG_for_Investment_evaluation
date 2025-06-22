# **RAG Application for Investment Evaluation**
## Problem Statement
 To build a Retrieval-Augmented Generation (RAG)-based application that
enables Investor to upload pitch deck and ask questions related to the content of the uploaded
PDFs.

## 📃 Project Overview

The "Pitch Deck Analyst" is a Streamlit-based web application that leverages Retrieval Augmented Generation (RAG) to enable users to interactively query and extract key investor information from uploaded PDF pitch decks. This tool empowers users, particularly venture capitalists, analysts, or founders, to quickly grasp the essence of a pitch deck without manually sifting through pages.

The application processes PDF documents by chunking their content, creating vector embeddings, and storing them in a vector database. When a user asks a question, the system retrieves the most relevant information from the PDF and uses a Large Language Model (LLM) to generate a concise and accurate answer, grounded in the document's content. It also features a specialized extraction tool to automatically pull out critical investor-centric details like business model, team highlights, funding ask, and more.

## ✨ Features

* **PDF Upload & Processing:** Easily upload pitch deck PDFs for analysis.
* **Intelligent Q&A:** Ask natural language questions about the pitch deck content and get context-aware answers.
* **Investor Information Extraction:** A dedicated feature to automatically extract and summarize crucial investment-related details (Company Name, Problem, Solution, Market, Business Model, Traction, Team, Financials, Funding Ask, Vision, Risks, Investment Thesis) in a structured format.
* **Chat History:** Maintains a conversational history for follow-up questions.
* **Source Document Visibility:** View the specific document chunks used by the LLM to generate answers, ensuring transparency and trustworthiness.
* **Streamlit UI:** Intuitive and user-friendly interface for seamless interaction.
* **API Key Integration:** Securely input your Groq API key (and optional HuggingFace Token) via the sidebar.

## 🚀 How it Works (RAG Architecture & Workflow)

The application implements a standard Retrieval Augmented Generation (RAG) pattern, which involves several key steps:

### 1. **Document Ingestion & Indexing Pipeline (Offline/Pre-processing)**

* **PDF Loading:** When a PDF is uploaded, `PyPDFLoader` reads the content of each page.
* **Text Splitting (Chunking):** The raw text from the PDF is divided into smaller, manageable `chunks` using `RecursiveCharacterTextSplitter`. This is crucial because LLMs have token limits, and smaller, coherent chunks ensure that relevant information can be retrieved effectively without exceeding these limits.
* **Embedding Generation:** Each text chunk is converted into a high-dimensional numerical vector (an "embedding") using a pre-trained embedding model (`HuggingFaceEmbeddings` - `all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
* **Vector Store Storage:** The generated embeddings (along with their original text chunks and metadata like page numbers) are stored in a `Chroma` vector database. This database allows for efficient similarity search.

    ```mermaid
    graph TD
        A[User Uploads PDF] --> B[PyPDFLoader: Load Document]
        B --> C{RecursiveCharacterTextSplitter: Chunk Text}
        C --> D[HuggingFaceEmbeddings: Generate Embeddings]
        D --> E[Chroma: Store Embeddings & Chunks]
        E --> F[Vector Store Ready for Retrieval]
    ```

### 2. **Query Processing & Response Generation (Online/Real-time)**

When a user submits a query:

* **History-Aware Retrieval:**
    * The user's current query and the existing `chat_history` are fed into an LLM (the "history-aware retriever prompt").
    * This LLM generates an optimized search query, taking into account the conversational context. This helps in retrieving more relevant documents for follow-up questions.
    * This search query is then used to perform a similarity search in the `Chroma` vector store.
    * The `retriever` fetches the `k` most semantically similar text chunks (relevant context) from the stored embeddings.
* **Augmented Generation:**
    * The original user query, the `chat_history`, and the `retrieved context` (relevant text chunks from the PDF) are combined into a single prompt for the main LLM (`ChatGroq`).
    * The LLM then generates a comprehensive answer based *only* on the provided context, minimizing hallucinations.
    * For the "Extract Key Investor Information" feature, a specialized `investor_extraction_prompt` is used with the retrieved (or all relevant) context to guide the LLM to output a structured summary.
* **Response to User:** The LLM's generated answer is displayed to the user, and the relevant source documents (chunks) are optionally shown for transparency.

    ```mermaid
    graph TD
        A[User Query] --> B[Chat History]
        A & B --> C[History-Aware Retriever Prompt (LLM)]
        C --> D[Optimized Search Query]
        D --> E[Chroma: Similarity Search (Retrieval)]
        E --> F[Relevant Context Chunks]
        F & A & B --> G[QA Prompt (Main LLM - Groq)]
        G --> H[Generated Answer]
        H --> I[Display to User]
        I --> J[Update Chat History]
    ```

### **Investor Information Extraction Workflow**

This process is a specialized version of the general RAG workflow:

* When the "Extract Key Investor Information" button is clicked, a broad query (e.g., "summarize the entire pitch deck for an investor") is implicitly used to retrieve a wide range of documents from the vector store.
* These retrieved documents, representing the full context of the pitch deck, are then passed to a dedicated `extraction_chain` with the `investor_extraction_prompt`.
* The LLM, guided by this prompt, processes the entire available context to identify and structure the specific investor-centric information, providing a comprehensive summary.

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application UI.
* **LangChain:** A framework for developing applications powered by language models, used for:
    * `PyPDFLoader`: Document loading.
    * `RecursiveCharacterTextSplitter`: Text chunking.
    * `Chroma`: In-memory vector store.
    * `HuggingFaceEmbeddings`: For generating text embeddings.
    * `ChatGroq`: Integration with Groq's LLM API.
    * Core chain components (`create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain`, `ChatPromptTemplate`, `MessagesPlaceholder`).
* **Groq API:** Provides fast inference for Large Language Models (LLMs).
* **HuggingFace Embeddings:** Provides a readily available model for generating embeddings.
* **`python-dotenv`:** For managing environment variables.
* **`tempfile`, `os`, `pathlib`:** For file handling and temporary file management.

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.9+
* Groq API Key (You can get one from [Groq Console](https://console.groq.com/))
* (Optional but recommended for `HuggingFaceEmbeddings`) Hugging Face Token (You can get one from [HuggingFace Settings](https://huggingface.co/settings/tokens))

### Steps

1.  **Clone the repository (or download the code):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, create one with the following content and then run `pip install -r requirements.txt`*:
    ```
    streamlit
    langchain
    langchain-community
    langchain-chroma
    langchain-huggingface
    langchain-groq
    pypdf
    python-dotenv
    ```

5.  **Create a `.env` file:**
    In the root directory of your project, create a file named `.env` and add your Groq API key to it:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    # Optionally, if you face issues with HuggingFace embeddings, add your HF token:
    # HF_TOKEN="your_huggingface_token_here"
    ```
    *Alternatively, you can directly paste your API key into the Streamlit sidebar when the app runs.*

## 🏃 Running the Application

1.  **Activate your virtual environment (if not already active):**
    * **On Windows:** `.\venv\Scripts\activate`
    * **On macOS/Linux:** `source venv/bin/activate`

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Open in Browser:** The application will open in your default web browser (usually at `http://localhost:8501`).

## ❓ Usage

1.  **Enter API Keys:** In the sidebar, enter your Groq API Key (and optionally your Hugging Face Token).
2.  **Upload PDF:** Use the "Upload Pitch Deck (PDF)" button in the sidebar to upload your pitch deck file.
3.  **Wait for Processing:** The application will indicate when the PDF is processed (chunks created, embeddings generated, vector store initialized). This might take a few moments depending on the PDF size.
4.  **Extract Information:** Click the "Extract Key Investor Information from Pitch Deck" button to get a structured summary.
5.  **Ask Questions:** Use the chat input box at the bottom to ask specific questions about the pitch deck content.
6.  **Review Responses:** The answers will appear in the chat interface. You can expand "Source Documents" to see the retrieved text chunks that informed the answer.

## ⚠️ Important Notes & Troubleshooting

* **Groq API Key:** Ensure your Groq API key is correct and has access to the `llama3-8b-8192` (or `llama3-70b-8192`) model. If you are getting generic or "safe" responses, double-check that you are using a general-purpose LLM model, not a safety classifier model (like `llama-guard`).
* **HuggingFace Token:** While `HuggingFaceEmbeddings` can often work without a direct token for public models, providing one (either via `.env` or sidebar) can prevent rate limiting or authentication issues.
* **PDF Content:** The quality and specificity of answers are directly dependent on the content of the uploaded PDF. If information is not present in the deck, the model will politely state it doesn't know.
* **LLM Context Limits:** For very large PDFs, even with chunking, the LLM might struggle to process all retrieved context for certain complex queries or the full extraction. LangChain's `create_stuff_documents_chain` stuffs all relevant documents into the prompt; extremely large combined contexts can lead to truncated responses or errors.
* **Temporary Files:** The application creates a temporary PDF file for processing, which is deleted after successful initialization.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

**Built with ❤️ by Naveen Babu S**
