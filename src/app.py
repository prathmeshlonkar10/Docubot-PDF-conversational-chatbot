import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from htmlTemplates import bot_template, user_template, css

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

# Modify this line to the name of your PDF file
pdf_file_name = "Ads_cookbook.pdf"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
    )
    return conversation_chain

def display_chat():
    if "chat_history" in st.session_state:
        pairs = [st.session_state.chat_history[i : i + 2] for i in range(0, len(st.session_state.chat_history), 2)]
        reversed_pairs = reversed(pairs)

        # Iterate over reversed pairs to maintain order of conversation
        for pair in reversed_pairs:
            st.write(user_template.replace("{{MSG}}", pair[0].content), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", pair[1].content), unsafe_allow_html=True)

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
   
def build_interface():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :robot_face:")

    # Text input for user questions
    user_question = st.text_input("Ask questions about your documents:")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload PDFs and click on 'Process' first.")
        else:
            handle_userinput(user_question)
            display_chat()

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")


if __name__ == "__main__":

    build_interface()
