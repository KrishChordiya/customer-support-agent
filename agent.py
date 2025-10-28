from typing import List, Tuple, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import config


# --- State ---
class GraphState(TypedDict):
    question: str
    chat_history: List[Tuple[str, str]]
    context_documents: List[Document]
    messages: List
    retriever: object
    api_key: str


# --- Node 1: Retrieve context ---
def retrieve_context(state: GraphState):
    docs = state["retriever"].invoke(state["question"])
    return {"context_documents": docs}


# --- Node 2: Generate answer (streaming) ---
def generate_answer(state: GraphState):
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        temperature=0.3,
        google_api_key=state["api_key"],
        streaming=True,  # Enable streaming
    )
    context = "\n".join([d.page_content for d in state["context_documents"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant. Use this context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    # Convert chat history
    history = [HumanMessage(c) if r == "user" else AIMessage(c)
               for r, c in state["chat_history"]]

    # Return a streaming generator
    return chain.stream({
        "chat_history": history,
        "question": state["question"],
    })


# --- Graph Builder ---
def create_agent_graph(api_key: str, retriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    graph = workflow.compile()

    def stream(inputs):
        """Streams tokens from the final LLM call."""
        state = {
            "api_key": api_key,
            "retriever": retriever,
            "question": inputs["question"],
            "chat_history": inputs.get("chat_history", []),
            "messages": [],
        }

        # Step 1: retrieve context
        retrieved = retrieve_context(state)
        state.update(retrieved)

        # Step 2: stream LLM output
        for chunk in generate_answer(state):
            yield chunk

    graph.stream = stream
    return graph


# --- Helper function (non-streaming fallback) ---
def run_agent(graph, api_key, retriever, question, chat_history):
    """Runs the agent graph synchronously (no stream)."""
    state = {
        "api_key": api_key,
        "retriever": retriever,
        "question": question,
        "chat_history": chat_history,
        "messages": [],
    }
    retrieved = retrieve_context(state)
    state.update(retrieved)

    final_chunks = []
    for chunk in generate_answer(state):
        if hasattr(chunk, "content"):
            final_chunks.append(chunk.content)
        elif isinstance(chunk, dict) and "content" in chunk:
            final_chunks.append(chunk["content"])
    return "".join(final_chunks)
