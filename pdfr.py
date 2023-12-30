import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from googletrans import Translator


# Function to perform Google Translate
def google_translate(messages):
    translator = Translator()
    result = translator.translate(messages, dest="ko")
    return result.text

# Main function
def main():
    st.set_page_config(page_title="PDF Analyzer", layout="wide")

    # Sidebar
    with st.sidebar:
        open_apikey = st.text_input(label='OpenAI API Key', placeholder='Enter your API key')
        if open_apikey:
            st.session_state["openai_api"] = open_apikey
        st.markdown('---')

    # Main space
    st.header("PDF 내용질문 프로그램")
    st.markdown('---')
    st.subheader("Upload a PDF File")
    
    # Upload PDF file
    pdf = st.file_uploader(" ", type="pdf")
    # After extracting text from the PDF
    if pdf is not None:
        st.write(f"PDF file: {pdf}")  # Display information about the uploaded PDF file

        pdf_reader = PdfReader(pdf)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        st.markdown('---')
        st.subheader("질문입력")
        
        # Get user question
        user_question = st.text_input("Ask a question:")
        
        if user_question:
            # Embeddings/Semantic Index
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api"])
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            docs = knowledge_base.similarity_search(user_question)

            # Questioning
            lim = ChatOpenAI(
                temperature=0,
                openai_api_key=st.session_state["openai_api"],
                max_tokens=2000,
                model_name='gpt-3.5-turbo',
                request_timeout=120
            )
            chain = load_qa_chain(lim, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the answer
            st.info(response)
            
            # Translation
            if st.button(label="Translate"):
                translation = google_translate(response)
                st.success(translation)

if __name__ == '__main__':
    main()
