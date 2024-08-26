import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import time

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
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        context = format_docs(relevant_docs)

        # Initialize the OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Create the prompt
        prompt = f"Context: {context}\n\nQuestion: {question}"

        # Get response from OpenAI API
        response = client.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate and concise answers based on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the answer from the response
        answer = response.choices[0].message['content']
        return answer

    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        return "Error: Invalid request."
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        return "Error: Authentication failed."
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        return "Error: Rate limit exceeded."
    except openai.error.OpenAIError as e:  # Catch all other OpenAI errors
        print(f"OpenAI error: {e}")
        return "Error: An error occurred with the OpenAI API."
    except Exception as e:
        print(f"Unexpected error: {e}")
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
