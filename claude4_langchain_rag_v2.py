from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


class ModernRAGApp:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """Initialize the RAG application with DeepSeek-R1 1.5B."""

        # Initialize DeepSeek model
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print("Loading DeepSeek-R1 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # Create HuggingFace pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Initialize LangChain components
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.retriever = None
        self.chat_history = []

        # Create the conversational RAG chain
        self._setup_chain()
        print("RAG application initialized successfully!")

    def _setup_chain(self):
        """Set up the conversational RAG chain using LCEL."""

        # Contextualize question prompt (simplified for smaller model)
        contextualize_q_system_prompt = """Given the conversation history and new question, rephrase the question to be standalone if needed. Otherwise return it unchanged.

Conversation: {chat_history}
Question: {input}
Standalone question:"""

        self.contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_system_prompt)

        # Main QA prompt (optimized for smaller model)
        qa_system_prompt = """Use the following context to answer the question. Be concise and accurate.

Context: {context}

Question: {input}
Answer:"""

        self.qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)

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
            # Format chat history for the prompt
            history_str = "\n".join([
                f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
                for msg in self.chat_history[-4:]  # Keep last 4 messages
            ])

            contextualize_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()
            result = contextualize_chain.invoke({
                "chat_history": history_str,
                "input": input_dict["input"]
            })

            # Clean up the response to get just the question
            lines = result.strip().split('\n')
            return lines[-1] if lines else input_dict["input"]
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
        })

        # Clean up response (remove any prompt artifacts)
        cleaned_response = response.split("Answer:")[-1].strip()
        if not cleaned_response:
            cleaned_response = response.strip()

        # Update chat history
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=cleaned_response)
        ])

        return cleaned_response

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
    # Initialize the RAG app (no API key needed!)
    print("Initializing DeepSeek RAG App...")
    rag_app = ModernRAGApp()

    # Load documents (replace with your file paths)
    # rag_app.load_documents(["document1.txt", "document2.txt"])

    # Example conversation
    print("RAG App initialized! Load documents and start asking questions.")

    # Example questions (uncomment after loading documents)
    # response1 = rag_app.ask_question("What is the main topic of the documents?")
    # print(f"Response: {response1}")

    # response2 = rag_app.ask_question("Can you elaborate on that topic?")
    # print(f"Follow-up Response: {response2}")


# Alternative: Simple function-based approach with DeepSeek
def create_simple_deepseek_rag_chain(vectorstore, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Create a simple RAG chain with DeepSeek model."""

    # Initialize DeepSeek model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template("""
    Context: {context}

    Question: {question}

    Answer based on the context above:
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