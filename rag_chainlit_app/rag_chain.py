# rag_chain.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import AIMessage

from chroma import ChromaUtils
from local_llm import load_qwen_llm

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise.
Question: {question} 
Context: {context} 
Helpful Answer:
"""

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="./model/all-MiniLM-L6-v2")
    vectorstore = ChromaUtils(embeddings, './chromadb').create_chroma_db(file_path)

    return vectorstore.as_retriever()

def create_conversational_qa_chain(pdf_path: str):
    retriever = process_pdf(pdf_path)
    llm = load_qwen_llm()

    print('创建Memory')
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    rag_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    print('创建qa_chain')
    # 通过retriever查询到的doc会自动放入Prompt中的context字段
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
        verbose=False
    )
    print(' qa_chain创建完成')
    return qa_chain

def format_chat_history(chat_history):
    """Format chat history for llm chain"""

    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def invoke_qa_chain(qa_chain, message, history):
    """Invoke question-answering chain"""

    formatted_chat_history = format_chat_history(history)
    # print("formatted_chat_history",formatted_chat_history)

    print('消息送入llm开始处理')
    # Generate response using QA chain
    response = qa_chain.invoke(
        {"question": message, "chat_history": formatted_chat_history}
    )
    print('llm处理完成')
    print(response)

    response_sources = response["source_documents"]

    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]

    # Append user message and response to chat history
    new_history = history + [AIMessage(content=response_answer)]

    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)

    return response_answer, new_history, response_sources