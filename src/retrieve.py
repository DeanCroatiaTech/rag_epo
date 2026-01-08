from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
import os

MODEL = "gpt-4.1-nano"
db_name = "vector_db"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
llm = ChatOpenAI(temperature=0, model_name=MODEL)

num_vectors = vectorstore._collection.count()
print("Number of vectors in the store:", num_vectors)

RETRIEVAL_K = 5
retriever = vectorstore.as_retriever()    


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company EPO.
You are chatting with a user about EP1001 form documentation.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""    

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    History format: list of dicts with 'role' and 'content' keys from Gradio.
    """
    # Extract user messages from dict format
    prior_messages = [
        str(msg["content"]) for msg in history 
        if msg.get("role") == "user" and msg.get("content")
    ]
    prior = "\n".join(prior_messages) if prior_messages else ""
    return prior + "\n" + question if prior else question

def normalize_content(content) -> str:
    """
    Convert Gradio message content into a plain string.
    """
    if content is None:
        return ""

    # Already a string
    if isinstance(content, str):
        return content

    # Gradio rich-text format: [{"text": "..."}]
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return " ".join(texts)

    # Fallback
    return str(content)


def answer_question(question: str, history) -> str:
    """
    Fully Gradio-compatible RAG chat function.
    Handles tuple history, dict history, and rich content.
    """

    messages = []
    user_questions = []

    # Normalize current question
    question = normalize_content(question)

    for msg in history:
        # ---- Dict-based history (new Gradio) ----
        if isinstance(msg, dict):
            role = msg.get("role")
            content = normalize_content(msg.get("content"))

            if not content:
                continue

            if role == "user":
                user_questions.append(content)
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # ---- Tuple-based history (old Gradio) ----
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user, assistant = msg
            user = normalize_content(user)
            assistant = normalize_content(assistant)

            if user:
                user_questions.append(user)
                messages.append(HumanMessage(content=user))
            if assistant:
                messages.append(AIMessage(content=assistant))

    # Combine for retrieval (NOW SAFE)
    combined = "\n".join(user_questions + [question])

    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT.format(context=context)

    # Prepend system message
    messages.insert(0, SystemMessage(content=system_prompt))

    # Add current question
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    # ✅ Always return plain string
    return str(response.content) 