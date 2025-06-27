import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Conversational RAG with PDF",
    layout="wide"
)

st.title(" Conversational RAG with PDF Upload")

# Sidebar for API keys and settings
st.sidebar.title(" Configuration")
st.sidebar.write("Upload PDFs and chat with their content")

# API Keys
groq_api_key = st.sidebar.text_input(" Enter your Groq API Key:", type='password')
cohere_api_key = st.sidebar.text_input(" Enter your Cohere API Key:", type='password')

# Model settings
if groq_api_key and cohere_api_key:
    pdf_summ = st.button(label="PDF Summarize")
    if pdf_summ:
        temperature = st.sidebar.slider(" Temperature", min_value=0.0, max_value=1.0, value=0.6)
        model = st.sidebar.selectbox(
            " Select Model:", 
            ["meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen-qwq-32b","mistral-saba-24b"]
        )

    # Session ID
    session_id = st.text_input(" Session ID", value="Session_1")

    # Initialize session state
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'conversation_rag_chain' not in st.session_state:
        st.session_state.conversation_rag_chain = None

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    def initialize_embeddings(api_key):
        """Initialize Cohere embeddings"""
        try:
            embeddings = CohereEmbeddings(
                cohere_api_key=api_key,
                model="embed-english-light-v3.0"  # Free tier model
            )
            return embeddings
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return None

    def setup_rag_chain(llm, retriever):
        """Setup the RAG chain with history"""
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history-aware retriever
        history_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # QA prompt
        system_prompt = (
            "You are a rude ai assistant who gives all answers in as rude way as possible."
            "Use the following pieces of retrieved context to answer in detailed and pointwise manner "
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create chains
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)
        
        # Create conversation chain with history
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        return conversation_rag_chain

    # Main application logic
    if groq_api_key and cohere_api_key:
        # Initialize LLM
        llm = ChatGroq(
            model=model,
            api_key=groq_api_key,
            temperature=temperature
        )
        
        # Initialize embeddings
        embeddings = initialize_embeddings(cohere_api_key)
        
        if embeddings:
            st.success(" Embeddings initialized successfully!")
            
            # File upload
            uploaded_files = st.file_uploader(
                "📄 Choose PDF files", 
                type="pdf", 
                accept_multiple_files=True,
                help="Upload one or more PDF files to chat with their content"
            )
            
            if uploaded_files:
                # Process PDFs
                with st.spinner(" Processing PDFs..."):
                    documents = []
                    
                    for uploaded_file in uploaded_files:
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Load and process PDF
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                            documents.extend(docs)
                            
                            st.success(f" Processed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f" Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_path)
                    
                    if documents:
                        try:
                            # Split documents
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1500,
                                chunk_overlap=200
                            )
                            splits = splitter.split_documents(documents)
                            
                            # Create vector database
                            st.session_state.vectordb = FAISS.from_documents(
                                documents=splits,
                                embedding=embeddings
                            )
                            
                            # Create retriever
                            retriever = st.session_state.vectordb.as_retriever(
                                search_kwargs={"k": 3}
                            )
                            
                            # Setup RAG chain
                            st.session_state.conversation_rag_chain = setup_rag_chain(llm, retriever)
                            
                            st.success(f" Vector database created with {len(splits)} chunks!")
                            
                        except Exception as e:
                            st.error(f" Error creating vector database: {str(e)}")
            
            # Chat interface
            if st.session_state.conversation_rag_chain:
                st.subheader(" Chat with your documents")
                
                # Display chat history
                if session_id in st.session_state.store:
                    chat_history = st.session_state.store[session_id]
                    for message in chat_history.messages:
                        if hasattr(message, 'content'):
                            if message.type == "human":
                                st.chat_message("user").write(message.content)
                            else:
                                st.chat_message("assistant").write(message.content)
                
                # Chat input
                user_input = st.chat_input("Ask a question about your documents...")
                
                if user_input:
                    # Display user message
                    st.chat_message("user").write(user_input)
                    
                    try:
                        # Get response
                        with st.spinner(" Thinking..."):
                            response = st.session_state.conversation_rag_chain.invoke(
                                {"input": user_input},
                                config={"configurable": {"session_id": session_id}}
                            )
                        
                        # Display assistant response
                        st.chat_message("assistant").write(response['answer'])
                        
                    except Exception as e:
                        st.error(f" Error getting response: {str(e)}")
            
            else:
                st.info(" Please upload PDF files to start chatting!")
        
        else:
            st.error(" Failed to initialize embeddings. Please check your Cohere API key.")
    else:
        st.warning("Please choose one of the following option to go ahead!")
else:
    st.warning(" Please enter both Groq and Cohere API keys to continue.")
