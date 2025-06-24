import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# import os
# from dotenv import load_dotenv
# load_dotenv()
# # Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Advanced RAG Document Q&A Chatbot"
# hf_api = os.getenv("HF_TOKEN")
hf_api = st.secrets['HF_TOKEN']
embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_api
)

api_key = st.sidebar.text_input("Enter your Groq API Key: ", type='password')

if api_key:
    def model_prep(api_key,model,temperature):
        llm = ChatGroq(model=model,api_key=api_key,temperature=temperature)
        return llm
    st.sidebar.title("Conversational RAG with PDF Upload and chat history")
    st.sidebar.write("Upload pdf's and chat with their content")
    temperature = st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0,value=0.6)
    model = st.sidebar.selectbox("Select an Open-Source Model : ",["mistral-saba-24b","llama-3.3-70b-versatile","gemma2-9b-it","meta-llama/llama-4-scout-17b-16e-instruct"])
    llm = model_prep(api_key,model,temperature)
    session_id = st.text_input("Session ID", value="Session_1")
    uploaded_files = st.file_uploader("Choose a Pdf File", type="pdf", accept_multiple_files=True)

    # Ensure store is initialized
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f'./temp.pdf'
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
            splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,  chunk_overlap = 200)
            splits = splitter.split_documents(documents)
            vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectordb.as_retriever()
            contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )
            history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            system_prompt = (
                "You are a rude ai assistant who gives all answers in as rude way as possible."
                "Use the following pieces of retrieved context to answer in detailed and pointwise manner "
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )
            qa_chain = create_stuff_documents_chain(llm,qa_prompt)
            rag_chain = create_retrieval_chain(history_retriever,qa_chain)
            def session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]
            conversation_rag_chain = RunnableWithMessageHistory(
                rag_chain,session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Your Question:")
            if user_input:
                session_history = session_history(session_id)
                response = conversation_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id":session_id}
                    },
                )
                st.write(st.session_state.store)
                st.success(f"Assistant: {response['answer']}")
                st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API!!")
