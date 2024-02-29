import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os
os.environ["OPENAI_API_KEY"] = "KEY"

# Create Streamlit UI
st.title("YouTube Video QA")

# Radio button to select YouTube Video QA
selected_option = st.sidebar.radio("Select Option", ["YouTube Video QA"])

# YouTube Video QA option selected
if selected_option == "YouTube Video QA":
    st.subheader("Upload YouTube Video")
    video_url = st.text_input("Enter YouTube Video URL")


    # Check if video URL is provided
    if video_url:
        try:
            # Load transcripts from YouTube video
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            data = loader.load()
            st.success("Video uploaded successfully!")

            # Split transcripts into smaller documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
            docs = text_splitter.split_documents(data)

            # Embed documents
            embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_documents(docs, embeddings)
            retriever = docsearch.as_retriever()

            # Initialize chat model
            chat_model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
            qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

            # Question and answer section
            st.subheader("Ask Questions")
            query = st.text_input("Enter your question")
            if st.button("Submit"):
                if query:
                    response = qa.run(query)
                    st.info("Answer: " + response)
                else:
                    st.warning("Please enter a question.")
        except Exception as e:
            st.error("Error occurred: " + str(e))