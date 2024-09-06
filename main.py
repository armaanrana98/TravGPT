import streamlit as st
from openai import OpenAI
import PyPDF2
import mysql.connector
from mysql.connector import Error
import time 

st.set_page_config(
    page_title="TravGPT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)

# Database connection
def get_db_connection():
    try:
        st.write("Attempting to connect to the database...")
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        st.write("Database connection successful!")
        return connection
    except Error as e:
        st.error(f"Error while connecting to database: {e}")
        print(f"Error while connecting to database: {e}")

# Logging question and response
def log_question_and_response(question, response, response_time_seconds):
    try:
        st.write("Logging question and response to the database...")
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO question_logs (question, response, response_time_seconds)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (question, response, response_time_seconds))
        connection.commit()
        st.write("Question and response logged successfully!")
        cursor.close()
        connection.close()
    except Error as e:
        st.error(f"Error during logging: {e}")
        print(f"Error during logging: {e}")

# Convert PDF to text
def pdf_file_to_text(pdf_file):
    text = ""
    st.write(f"Reading PDF file: {pdf_file}")
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text += page.extract_text()
                st.write(f"Extracted text from page {page_num + 1}")
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        print(f"Error reading PDF file: {e}")

# Upload and index PDF file
def upload_and_index_file(pdf_file_path):
    try:
        st.write(f"Uploading and indexing PDF file: {pdf_file_path}")
        with open(pdf_file_path, "rb") as file_stream:
            vector_store = client.beta.vector_stores.create(name="TravGPT Documents")
            st.write(f"Vector store created: {vector_store.id}")
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=[file_stream]
            )
            st.write("File uploaded and indexed successfully.")
        return vector_store
    except Exception as e:
        st.error(f"Error during file upload and indexing: {e}")
        print(f"Error during file upload and indexing: {e}")

# Create assistant with vector store
def create_assistant_with_vector_store(vector_store):
    try:
        st.write(f"Creating assistant with vector store: {vector_store.id}")
        assistant = client.beta.assistants.create(
            name="TravGPT Assistant",
            instructions="Answer the question as precisely as possible using the provided context. Answer it in a proper and detailed manner. If the answer is not contained in the context, say 'answer not available in context'.",
            model="gpt-4o",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
        st.write(f"Assistant created successfully with ID: {assistant.id}")
        return assistant
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        print(f"Error creating assistant: {e}")

# Generate answer using assistant
def generate_answer(assistant_id, question):
    st.write(f"Generating answer for assistant ID: {assistant_id}")
    try:
        thread = client.beta.threads.create(
            messages=[{"role": "user", "content": question}]
        )
        st.write(f"Thread created with ID: {thread.id}")

        answer = ""
        start_time = time.time()

        st.write("Starting to stream the answer...")
        with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
            for event in stream:
                st.write(f"Event received: {event.event}")
                if event.event == 'thread.message.delta':
                    for delta_block in event.data.delta.content:
                        st.write(f"Delta block received: {delta_block}")
                        if delta_block.type == 'text':
                            st.write(f"Adding text: {delta_block.text.value}")
                            answer += delta_block.text.value  # Log each part of the answer as it arrives
                            st.write(f"Current answer: {answer}")

        end_time = time.time()
        response_time_seconds = end_time - start_time
        st.write(f"Answer generation completed in {response_time_seconds} seconds")

        log_question_and_response(question, answer, response_time_seconds)
        
        # Final answer display
        if answer.strip():
            st.write(f"Final answer: {answer}")
        else:
            st.write("No valid answer received from the assistant.")
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        print(f"Error generating answer: {e}")

# Main function
def main():
    st.header("TravGPT🤖")

    question = st.text_input("Ask a question:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            st.write("Processing started.")
            try:
                vector_store = upload_and_index_file(PDF_FILE_PATH)
                st.write("Vector store ready.")
                assistant = create_assistant_with_vector_store(vector_store)
                st.write("Assistant ready.")

                if question:
                    st.write("Generating answer for question...")
                    generate_answer(assistant.id, question)
                else:
                    st.warning("Please enter a question.")
            except Exception as e:
                st.error(f"Error in main processing: {e}")
                print(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()
