import os
from pathlib import Path
from dotenv import load_dotenv 
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter   
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate 

# os.chdir("../")

#  Load environmental variable 
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Function for getting PDF 
def get_pdf(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        loader = PyPDFLoader(pdf_docs)
        doc = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500))
        docs.extend(doc)

    # docs = []
    # for doc_file in pdf_docs:       
    #     temp_pdf = f"./temp.pdf"
    #     with open(temp_pdf, "wb") as f:
    #         f.write(doc_file.getvalue())
    #         fine_name=doc_file.name
    #     loader = PyPDFLoader(temp_pdf)
    #     doc = loader.load()
    #     docs.extend(doc)

    return docs

# Function to create Embeddings
def get_embedding(model_name = "sentence-transformers/all-mpnet-base-v2", 
                  model_kwargs = {'device': 'cpu'},
                  encode_kwargs = {'normalize_embeddings': False}):


    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
                    )
    return embedding

# Function for processing data 
def embed_docs(chunks, chunk_size:int=5000, 
                 chunk_overlap:int = 1000, embedding = get_embedding(),
                   ):
    # # Splitting the document
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # chunks = text_splitter.split_documents()

    # Storing it in vector store
    chroma_db = Chroma.from_documents(documents=chunks, embedding=embedding)

    return chroma_db
    
# Function to get conversional chain 
def get_conversional_chain():
    # Defining the template for the prompt
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provide context just say, "Can't answer with the provided context",
    dont provide the the worng response.
    Context: \n{context}\n
    Question: \n{question}?\n
    
    Answer:

    """
    # Initilizing mode
    chat_model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

    # Prompt generation
    prompt = PromptTemplate(template=prompt_template,
                   input_variables=["context", "question"])

    # Create chain 
    chain = load_qa_chain(llm = chat_model,
                          chain_type="stuff",
                          prompt = prompt)

    return chain 

def user_query(query):

    embedding = get_embedding()
    similar_docs = process_docs().similarity_search(query)
    chain = get_conversional_chain()

    response = chain({"inputs":similar_docs,
                      "quention":query },
                      return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])


# Create main function 
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLAMA-4")

    user_query = st.text_input("Ask your Question from the PDF:")
    if user_query:
        user_query(user_query)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF fies and click on summit")
        if st.button("Submit"):
            with st.spinner("Processing......"):
                docs = get_pdf(pdf_docs)
                db = embed_docs(docs)
                st.success("Document loaded and stored as vector")

if __name__== "__main__":
    main()