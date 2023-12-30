#기본정보

import streamlit as st
#pdf리더
from PyPDF2 import PdfReader
#랭체인
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
#구글번역
from googletrans import Translator


# 기능구현
def google_trans(messages):
    google = Translator()
    resurt = google.Translate(messages, dest="ko")
    return ressult.text

# 메인함수
def main():
    st.set_page_config(page_title="PDF analyzer", layout="wide")

    #사이드바
    with st.sidenar:
        open_apikey=st.text_input(label='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',placeholder='enter your ap')
        if open_apikey:
            st.session_state["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"]=open_apikey
        st.markdown('---')
    
    #메인공간
    st.header("pdf 내용 질문 프로그램")
    st.markdown('---')
    st.subheader("pdf 파일을 넣으세요")
    
    #pdf파일받기
    pdf=st.file_uploader(" ", type="pdf")
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)

        st.markdown('---')
        st.subheader("질문을 입력하세요")
        #질문받기
        user_question=st.text_input("Ask question :")
        if user_question:
            #임베딩/시멘틱 인덱스
            embeddings=OpenAIEmbeddings(openai_api_key=st.session_state["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"])
            knowledge_base=FAISS.from_texts(chunks,embeddings)
            docs=knowledge_base.similarity_search(user_question)

            #질문
            lim=ChatOpenAI(temperature=0,
                    openai_api_key=st.session_state["open_api"],
                    max_tokens=2000,
                    model_name='gpt-3.5-turbo',
                    request_timeout=120
                    )
            chain=load_qa_chain(lim,chain_type="stuff")
            response=chain.run(input_documents=docs,question=user_question)
                #답변결과
            st.info(response)
            #번역
            if st.button(label="번역하기"):
                trans=google_trans(response)
                st.success(trans)

    if _name_=='__main__':
        main()