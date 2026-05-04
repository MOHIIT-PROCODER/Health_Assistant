from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from src.prompt import system_prompt

# ------------------ INIT ------------------
app = Flask(__name__)
load_dotenv()

# ENV VARIABLES
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ GLOBALS (LAZY LOAD) ------------------
retriever = None
rag_chain = None


# ------------------ LOAD RETRIEVER ------------------
def get_retriever():
    global retriever

    if retriever is None:
        embeddings = download_hugging_face_embeddings()

        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medical-chatbot",
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    return retriever


# ------------------ LOAD CHAIN ------------------
def get_chain():
    global rag_chain

    if rag_chain is None:
        retriever_instance = get_retriever()

        chatModel = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever_instance, qa_chain)

    return rag_chain


# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "Please enter a message."

    try:
        chain = get_chain()
        response = chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't understand.")
        return str(answer)

    except Exception as e:
        print("Error:", e)
        return "Something went wrong. Try again."


# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic port
    app.run(host="0.0.0.0", port=port)
