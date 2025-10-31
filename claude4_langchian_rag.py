from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage
import os


class ModernRAGApp:
    def __init__(self, openai_api_key: str):
        """Initialize the RAG application with modern LangChain components."""
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize components
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.chat_history = []

        # Create the conversational RAG chain
        self._setup_chain()

    def _setup_chain(self):
        """Set up the conversational RAG chain using LCEL."""

        # Contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Main QA prompt
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

{context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Store prompts for later use
        self.contextualize_q_prompt = contextualize_q_prompt
        self.qa_prompt = qa_prompt

    def load_documents(self, file_paths: list):
        """Load and process documents for retrieval."""
        documents = []

        for file_path in file_paths:
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

        print(f"Loaded {len(splits)} document chunks into vectorstore")

    def _contextualize_question(self, input_dict):
        """Contextualize the question based on chat history."""
        if self.chat_history:
            contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()
            return contextualize_q_chain.invoke({
                "chat_history": self.chat_history,
                "input": input_dict["input"]
            })
        else:
            return input_dict["input"]

    def _format_docs(self, docs):
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask_question(self, question: str) -> str:
        """Ask a question and get a response using the RAG chain."""
        if not self.retriever:
            return "Please load documents first using load_documents()"

        # Create the RAG chain using LCEL
        rag_chain = (
                RunnablePassthrough.assign(
                    context=RunnableLambda(self._contextualize_question) | self.retriever | self._format_docs
                )
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
        )

        # Get response
        response = rag_chain.invoke({
            "input": question,
            "chat_history": self.chat_history
        })

        # Update chat history
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response)
        ])

        return response

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []
        print("Chat history cleared")

    def get_relevant_docs(self, question: str, k: int = 4):
        """Get relevant documents for a question."""
        if not self.retriever:
            return []

        return self.retriever.get_relevant_documents(question, k=k)


# Example usage
def main():
    # Initialize the RAG app
    rag_app = ModernRAGApp(openai_api_key="your-openai-api-key-here")

    # Load documents (replace with your file paths)
    # rag_app.load_documents(["document1.txt", "document2.txt"])

    # Example conversation
    print("RAG App initialized! Load documents and start asking questions.")

    # Example questions (uncomment after loading documents)
    # response1 = rag_app.ask_question("What is the main topic of the documents?")
    # print(f"Response: {response1}")

    # response2 = rag_app.ask_question("Can you elaborate on that topic?")
    # print(f"Follow-up Response: {response2}")


# Alternative: Simple function-based approach
def create_simple_rag_chain(vectorstore, llm):
    """Create a simple RAG chain without conversation history."""

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below:

    Context: {context}

    Question: {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    main()