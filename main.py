import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2

# Set the page configuration
st.set_page_config(
    page_title="TravGPT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Set your OpenAI API key
openai_api_key = 'sk-proj-sHu3KT60m5Q51Ivtc5KCT3BlbkFJEoqpkJl19iEpAZN1Ttsp'

# Path to the PDF file (relative path)
PDF_FILE_PATH = "data.pdf"

def pdf_file_to_text(pdf_file):
    print("Extracting text from PDF...")
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    print(f"Extracted text length: {len(text)}")
    return text

def text_splitter(raw_text):
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,  # This overlap helps maintain context between chunks
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = text_splitter.split_text(text=raw_text)
    print(f"Number of text chunks: {len(chunks)}")
    return chunks

def get_vector_store(text_chunks):
    print("Generating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    print("Vector store created successfully.")
    return vectorstore

def format_docs(docs):
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    print(f"Formatted docs length: {len(formatted_docs)}")
    return formatted_docs

def generate_answer(question, retriever):
    print("Generating answer...")
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"Number of relevant documents: {len(relevant_docs)}")

        context = format_docs(relevant_docs)

        # Initialize the OpenAI client
        print("Initializing OpenAI client...")
        client = OpenAI(api_key=openai_api_key)

        # Create the prompt
        prompt = f"Context: {context}\n\nQuestion: {question}"
        print(f"Prompt length: {len(prompt)}")

        # Get response from OpenAI API
        print("Sending request to OpenAI API...")
        response = client.Completions.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )

        # Extract the answer from the response
        answer = response.choices[0].text.strip()
        print(f"Answer generated: {answer}")
        return answer

    except Exception as e:
        print(f"Error: {e}")
        return "Error: An unexpected error occurred."

def main():
    st.header("TravGPT")

    question = st.text_input("Ask a question:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            # Open and process the PDF file
            raw_text = pdf_file_to_text(PDF_FILE_PATH)
            text_chunks = text_splitter(raw_text)
            vectorstore = get_vector_store(text_chunks)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            if question:
                answer = generate_answer(question, retriever)
                st.write(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
