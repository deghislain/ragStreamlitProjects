import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import Hash_Map as map

label = "Upload a PDF document for your Q&A session"
is_doc_uploaded = False
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'))
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()

file_name = ""


def get_the_pdf_file():
    global retriever, file_name, is_doc_uploaded
    file = st.file_uploader(label, type=['pdf'], accept_multiple_files=False, key=None, help=None, on_change=None,
                            args=None, kwargs=None, disabled=False, label_visibility="visible")
    if file is not None:
        is_doc_uploaded = True
        file_name = file.name
        document = []
        reader = PdfReader(file)
        i = 1
        for page in reader.pages:
            document.append(Document(page_content=page.extract_text(), metadata={'page': i}))
            i += 1
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(document)
        # define embedding
        embeddings = OpenAIEmbeddings()
        # create vector database from data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        # define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever


def get_the_model(retriever):
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )
    return qa


questions_list = []
qa_map = map.HashMap(10)


def update_chat_session_QA(question, response):
    if 'qa_map' not in st.session_state:
        st.session_state['qa_map'] = qa_map
        st.session_state.qa_map.put(question, response["answer"])
    else:
        st.session_state.qa_map.put(question, response["answer"])
    if 'q_list' not in st.session_state:
        st.session_state['q_list'] = questions_list
        st.session_state.q_list.append(question)
    else:
        st.session_state.q_list.append(question)


q_list_session = questions_list
qa_map_session = qa_map


def print_conversation():
    global q_list_session, qa_map_session
    if 'qa_map' in st.session_state:
        qa_map_session = st.session_state['qa_map']

    if 'q_list' in st.session_state:
        q_list_session = st.session_state['q_list']

    count = 0
    while count < q_list_session.__len__():
        messages = st.container(height=None, border=None)
        q = q_list_session[count]
        messages.chat_message("user").write(q)
        messages.chat_message("assistant").write(qa_map_session.get_val(q))
        count = count + 1


retriever = get_the_pdf_file()

if is_doc_uploaded:
    st.write("Welcome" + "," + " the document " + file_name + " Is ready for a Q&A session")
    question = st.chat_input("Type your questions here")

    if 'qa_map' in st.session_state:
        qa_map_session = st.session_state['qa_map']

    if 'q_list' in st.session_state:
        q_list_session = st.session_state['q_list']
    count = 0
    messages = st.container(height=None, border=None)
    while count < q_list_session.__len__():
        q = q_list_session[count]
        messages.chat_message("user").write(q)
        messages.chat_message("assistant").write(qa_map_session.get_val(q))
        count = count + 1

    if question:
        messages.chat_message("user").write(question)
        qa = get_the_model(retriever)
        response = qa({"question": question})
        messages.chat_message("assistant").write(response["answer"])
        update_chat_session_QA(question, response)
