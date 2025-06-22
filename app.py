import os
from pathlib import Path
from dotenv import load_dotenv 
import tempfile 
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Load Environmental variable 
load_dotenv()

# Streamlit UI 
st.set_page_config(page_icon="📃", page_title="Pitch Deck Analyst from PDF by RAG",
                   layout= "wide")
st.title("Pitch Deck Analyst")
st.markdown("Upload a pitch deck PDF, ask questions,or extract key investor information")

# Initialize chat history in session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for inputs: API Key and File Uploader
with st.sidebar:
    # st.title("Menu")
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password", placeholder="API Key here")
    hf_token = st.text_input("Enter your HuggingFace Token", type="password", placeholder="HuggingFace Token")
    uploaded_file = st.file_uploader("Upload Pitch Deck (PDF)", type="pdf", )

    # Store API key in session state for later use
    if groq_api_key:    # From user UI input
        st.session_state.groq_api_key = groq_api_key
    elif "GROQ_API_KEY" in os.environ:  # From user .env file
        st.session_state.groq_api_key = os.environ["GROQ_API_KEY"]
    else:
        st.session_state.groq_api_key = None


    st.markdown("---") # Separator

# Initialize components in session state
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "extraction_chain" not in st.session_state: # Chain for investor extraction
    st.session_state.extraction_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:   # Used to stop reprocessing our file everytime
    st.session_state.pdf_processed = False

    # --- Langchain Processing (only run if file is uploaded and not yet processed) ---
if uploaded_file is not None and not st.session_state.pdf_processed:
    if not st.session_state.groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar before uploading a PDF.")
        st.stop()

    # Save the uploaded PDF to a temporary file since PyPDF loader doesnot accept uploader file directly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    st.sidebar.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
    st.sidebar.markdown("Processing PDF... This might take a moment.")

    # 1. Load PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.sidebar.write(f"Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {e}")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        st.stop()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    st.sidebar.write(f"Split into {len(texts)} chunks.")

    # 3. Create embeddings and vector store
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(texts, embeddings)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.sidebar.success("Vector store created successfully!")
    except Exception as e:
        st.sidebar.error(f"Error creating embeddings or vector store: {e}")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        st.stop()

    # 4. Initialize Groq LLM
    try:
        st.session_state.llm = ChatGroq(
            temperature=0,
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama3-70b-8192"
        )
        st.sidebar.success("Groq LLM initialized!")
    except Exception as e:
        st.sidebar.error(f"Error initializing Groq LLM: {e}")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        st.stop()

    # 5. Define History-Aware Retriever Prompt
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
        ]
    )

    # 6. Create History-Aware Retriever
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, st.session_state.retriever, history_aware_retriever_prompt
    )

    # 7. Define QA Prompt (for general questions)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant. Answer the user's question truthfully and directly based only on the provided context.\nIf the answer is not in the context, politely state that you don't know.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )

    # 8. Create Document Combining Chain for QA
    document_chain_qa = create_stuff_documents_chain(st.session_state.llm, qa_prompt)

    # 9. Create the main Retrieval Chain (for general questions)
    st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, document_chain_qa)

    # --- New Chain for Investor-Centric Extraction ---

    # 10. Define Investor-Centric Key Information Extraction Prompt (UPDATED!)
    investor_extraction_prompt_template = """
    You are a highly experienced Venture Capitalist (VC) analyst evaluating a pitch deck.
    Your task is to extract and summarize the most critical information an investor needs to quickly understand this opportunity.
    Focus on providing precise, concise, and structured answers for the following categories.
    If information for a category is genuinely not present or applicable in the provided pitch deck content, state 'N/A'.

    Extracted Investment Summary:

    - **1. Company/Startup Name:**
    - **2. What they do (Product/Service Description):**
    - **3. Problem being Solved:**
    - **4. Solution Overview:**
    - **5. Target Market & Size (TAM/SAM/SOM if available):**
    - **6. Business Model & Revenue Streams:**
    - **7. Competitive Landscape & Advantage (Moat):**
    - **8. Traction & Key Milestones:**
    - **9. Team Highlights:**
    - **10. Financial Summary (Key Projections/Currents if available):**
    - **11. Funding Ask & Use of Funds:**
    - **12. Vision & Long-term Potential:**
    - **13. Key Risks & Challenges:**
    - **14. Overall Investment Thesis (Why invest?):**

    Pitch Deck Content (Context):
    {context}

    Please provide the extracted summary strictly following the numbered bulleted list format above.
    Be objective and and avoid speculative language.
    If certain information is not provided in the context, state 'N/A' for that specific point.
    """
    investor_extraction_prompt = ChatPromptTemplate.from_template(investor_extraction_prompt_template)

    # 11. Create a simple chain for investor information extraction
    # This chain will use the LLM and the new investor_extraction_prompt
    st.session_state.extraction_chain = create_stuff_documents_chain(st.session_state.llm, investor_extraction_prompt)

    st.session_state.pdf_processed = True
    st.sidebar.success("All RAG Chains initialized!")
    
    # Clean up the temporary file after processing
    if os.path.exists(pdf_path):
        os.unlink(pdf_path) 

elif uploaded_file is None and not st.session_state.pdf_processed:
    st.info("Please enter your Groq API Key and upload a PDF file to get started. Use the sidebar for inputs.")
elif uploaded_file is None and st.session_state.pdf_processed:
    st.info("PDF processed. You can now ask questions or use the extraction tools!")

# --- Main Content Area Buttons and Chat ---
st.subheader("Actions:")
# Only one button now for extraction
extract_info_button = st.button("Extract Key Investor Information from Pitch Deck")

if extract_info_button:
    if st.session_state.extraction_chain and st.session_state.vectorstore:
        with st.spinner("Extracting key investor information..."):
            try:
                # Retrieve all documents to provide the fullest context for extraction
                # For this demo, we'll retrieve a large number of top documents
                # If your PDF is very large, this might still hit context limits for LLM.
                context_docs = st.session_state.retriever.invoke("summarize the entire pitch deck for an investor") # Broad investor-focused query
                
                if not context_docs:
                    st.warning("No relevant documents retrieved for extraction. The PDF might be empty or content is not extractable.")
                    st.session_state.messages.append({"role": "assistant", "content": "Could not retrieve documents for extraction."})
                    extracted_content_to_display = "N/A - No context retrieved."
                else:
                    print("\n--- Context for Extraction Chain ---")
                    for i, doc in enumerate(context_docs):
                        print(f"Doc {i+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content[:200]}...")
                    print("-----------------------------------\n")

                    extraction_response = st.session_state.extraction_chain.invoke({"context": context_docs})
                    
                    # print("\n--- Full Response from Extraction Chain ---")
                    # print(extraction_response)
                    # print(f"Type of extraction_response: {type(extraction_response)}")
                    # print("-------------------------------------------\n")

                    # Robust extraction of content
                    if isinstance(extraction_response, dict) and "answer" in extraction_response:
                        extracted_content_to_display = extraction_response["answer"]
                    elif isinstance(extraction_response, dict) and "output" in extraction_response:
                        extracted_content_to_display = extraction_response["output"]
                    elif hasattr(extraction_response, "content"):
                        extracted_content_to_display = extraction_response.content
                    elif isinstance(extraction_response, str):
                        extracted_content_to_display = extraction_response
                    else:
                        extracted_content_to_display = "Could not parse extraction result."
                        st.error(f"Unexpected extraction response type: {type(extraction_response)}")

                st.subheader("Extracted Key Investor Information:")
                st.markdown(extracted_content_to_display)
                st.success("Key investor information extracted!")
                st.session_state.messages.append({"role": "assistant", "content": "Key investor information extracted."})
                st.session_state.messages.append({"role": "assistant", "content": extracted_content_to_display})
            except Exception as e:
                st.error(f"Error during information extraction: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred during extraction: {e}"})
    else:
        st.warning("Please upload a PDF and ensure it's processed first.")

# Chat input and response generation
st.markdown("---")
st.subheader("Ask Questions about the Pitch Deck:")
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain and st.session_state.groq_api_key:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    langchain_messages = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=msg["content"]))

                    # print(f"\n--- Invoking RAG Chain with Input ---")
                    # print(f"Input: {prompt}")
                    # print(f"Chat History: {langchain_messages}")
                    # print("------------------------------------\n")

                    response = st.session_state.rag_chain.invoke({
                        "input": prompt,
                        "chat_history": langchain_messages
                    })

                    # # --- CRITICAL DEBUGGING: Display full response object ---
                    # st.subheader("Chain Debugging Output:")
                    # st.write("Full Chain Response Object:")
                    # st.json(response) # This will display the full JSON structure in the UI!
                    # st.write(f"Type of response object: {type(response)}")
                    
                    # print("\n--- Full Response from RAG Chain (Terminal) ---")
                    # print(response)
                    # print(f"Type of response: {type(response)}")
                    # print("----------------------------------\n")

                    # --- FIX: Extract the answer robustly ---
                    answer_content = None
                    if isinstance(response, dict) and "answer" in response:
                        answer_content = response["answer"]
                    elif isinstance(response, dict) and "output" in response: # Langchain output_parser
                        answer_content = response["output"]
                    elif hasattr(response, "content"): # Direct LLM response (AIMessage)
                        answer_content = response.content
                    elif isinstance(response, str): # Direct string output from simple chain
                        answer_content = response
                    
                    # print(f"Extracted answer_content (type {type(answer_content)}):")
                    # print(f"'{answer_content}'") # Print with quotes to see if it's empty string or whitespace

                    if answer_content and str(answer_content).strip(): # Check if not empty/whitespace
                        with st.chat_message("assistant"): # Make sure it's inside the assistant's message bubble
                            st.markdown(answer_content)
                    else:
                        st.warning("The LLM did not provide a visible answer. This might be due to context limits, an inability to find relevant information, or a prompt issue causing an empty/whitespace response.")
                        with st.chat_message("assistant"): # Show message in bubble
                            st.markdown("**(No answer generated, but relevant documents are below if available)**")

                    if response.get("context"):
                        with st.expander("Source Documents (Retrieved Context)"):
                            for i, doc in enumerate(response["context"]):
                                st.write(f"**Document {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                                st.code(doc.page_content[:500] + "...")
                                st.write("---")
                    else:
                        st.info("No source documents were retrieved for this query.")

                    # Add assistant response to chat history (even if empty, for continuity)
                    st.session_state.messages.append({"role": "assistant", "content": answer_content if answer_content else "(No answer generated)"})

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
    else:
        st.warning("Please ensure the API key is entered, PDF is uploaded, and processed before asking questions.")
        st.session_state.messages.append({"role": "assistant", "content": "Please ensure the API key is entered and PDF is processed."})
