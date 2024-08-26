import streamlit as st
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
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
openai.api_key = openai_api_key

# Path to the PDF file (relative path)
PDF_FILE_PATH = "data.pdf"

def pdf_file_to_text(pdf_file):
    # Extract text from the PDF file
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def text_splitter(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,  # This overlap helps maintain context between chunks
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, retriever):
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(question)
    context = format_docs(relevant_docs)

    # Create the messages list to send to the ChatGPT API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
    ]

    # Call the OpenAI API to get the response
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",  # Use the correct model name
        messages=messages,
        max_tokens=150,
        temperature=0.1
    )

    # Extract and return the answer from the response
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def main():
    st.header("TravGPT")

    question = st.text_input("Ask a question:")

    if st.button("Ask"):
        with st.spinner("Kaha se late ho itne Ache Question :)"):
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
