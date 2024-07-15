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

def update_chat_session_QA(response, question):
    if 'qa_map' not in st.session_state:
        st.session_state['qa_map'] = qa_map
    else:
        st.session_state.qa_map.put(question, response["answer"])
    if 'q_list' not in st.session_state:
        st.session_state['q_list'] = questions_list
    else:
        st.session_state.q_list.append(question)

def print_conversation(qa_map_session, q_list_session, isWelcomeMsg):
    for q in q_list_session:
        if isWelcomeMsg:
            isWelcomeMsg = False
            continue
        else:
            st.write(q)
            st.write(qa_map_session.get_val(q))


questions_list = []
qa_map = map.HashMap(10)

retriever = get_the_pdf_file()

if is_doc_uploaded:
    qa_map.put("Welcome", " the document " + file_name + " Is ready for a Q&A session")
    questions_list.append("Welcome")
    st.write(questions_list[0] + "," + qa_map.get_val(questions_list[0]))
    question = st.text_input("Type your questions", "here")
    qa = get_the_model(retriever)
    response = qa({"question": question})
    update_chat_session_QA(response, question)

isWelcomeMsg = True
if 'qa_map' in st.session_state:
    qa_map_session = st.session_state['qa_map']
else:
    qa_map_session = qa_map

if 'q_list' in st.session_state:
    q_list_session = st.session_state['q_list']
else:
    q_list_session = questions_list

print_conversation(qa_map_session, q_list_session, isWelcomeMsg)


