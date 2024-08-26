import streamlit as st
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set the page configuration
st.set_page_config(
    page_title="TravGPT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Path to the PDF file
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
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key='yZ6ffLIC0qNWJrIWefcrmPs7BWIkOlD5ZpQV1b9Y')

    prompt_template = """Answer the question as precisely as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context." \n\n
                    Context: \n {context} \n\n
                    Question: \n {question} \n
                    Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

def main():
    st.header("TravGPT 🤖")

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
