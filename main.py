import streamlit as st
from openai import OpenAI
import PyPDF2

# Set the page configuration
st.set_page_config(
    page_title="TravGPT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Path to the PDF file
PDF_FILE_PATH = "data.pdf"

# OpenAI API key
openai_api_key = ""

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def pdf_file_to_text(pdf_file):
    # Extract text from the PDF file
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def upload_and_index_file(pdf_file_path):
    # Upload the PDF to OpenAI and create a vector store
    with open(pdf_file_path, "rb") as file_stream:
        vector_store = client.beta.vector_stores.create(name="TravGPT Documents")
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=[file_stream]
        )
    return vector_store

def create_assistant_with_vector_store(vector_store):
    # Create an assistant and link it with the vector store
    assistant = client.beta.assistants.create(
        name="TravGPT Assistant",
        instructions="You are an expert in answering travel-related questions using the provided documents.",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )
    return assistant

def generate_answer(assistant_id, question):
    # Create a thread, attach the question, and get the answer
    thread = client.beta.threads.create(
        assistant_id=assistant_id,
        messages=[{"role": "user", "content": question}]
    )

    # Stream the response from the assistant
    with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
        for event in stream:
            if event.type == "text_created":
                st.write(event.text)
                break  # Stop after getting the first response

def main():
    st.header("TravGPT🤖")

    question = st.text_input("Ask a question:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            # Open and process the PDF file
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            assistant = create_assistant_with_vector_store(vector_store)

            if question:
                generate_answer(assistant.id, question)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
