import os
import streamlit as st
import time

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

FAISS_INDEX_PATH = "faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# ================== PROCESS URL ==================
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Text Splitter Done...✅")

    embeddings = OpenAIEmbeddings()

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding + Index Created...✅")

    vectorstore_openai.save_local(FAISS_INDEX_PATH)

    time.sleep(2)

# ================== QUERY ==================
query = st.text_input("Question: ")

if query:
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)