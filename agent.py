# ==============================================================================
# Capstone Project: Building Agentic AI Systems
# Track Chosen: Track A - Personal Assistant with Subagents
# Student Name: Osamah Alsuhaibani
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Environment Setup & LangSmith Observability
# ------------------------------------------------------------------------------
import os
from typing import Dict, List, TypedDict, Annotated
import operator

# Configure LangSmith Observability
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key_here"
os.environ["LANGCHAIN_PROJECT"] = "Capstone-Project-Osamah"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# ------------------------------------------------------------------------------
# 2. RAG Pipeline Setup (Documents, Splitting, Embeddings, ChromaDB)
# ------------------------------------------------------------------------------
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

# Load sample corporate documents
docs = [
    Document(page_content="Meeting Policy: Strategy meetings are held on Tuesdays at 10 AM in Room 4B."),
    Document(page_content="Email Policy: High-priority client inquiries require explicit manager approval before sending responses."),
    Document(page_content="Calendar Support: Recurring team syncs should always be scheduled with a 15-minute buffer.")
]

# Text Splitting & Embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Define Custom Tools
@tool
def search_policy_kb(query: str) -> str:
    """Retrieves corporate calendar and email policies from the knowledge base."""
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

@tool
def create_calendar_event(event_details: str) -> str:
    """Schedules a new calendar event."""
    return f"[SUCCESS] Event successfully scheduled: '{event_details}'"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an outbound email."""
    return f"[SUCCESS] Email sent to {to} | Subject: {subject} | Body: {body}"

# ------------------------------------------------------------------------------
# 3. State Management, LangGraph Construction & Human-in-the-Loop
# ------------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# Define State Structure
class AgentState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    next_agent: str
    memory: Dict[str, str]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Supervisor Router Node
def supervisor_node(state: AgentState):
    messages = state["messages"]
    prompt = f"""You are a Supervisor Agent overseeing two specialized assistants: 'calendar' and 'email'.
Based on the conversation history, decide the appropriate subagent to handle the request.
Respond ONLY with one word: 'calendar', 'email', or 'FINISH'.

History: {messages}"""
    
    response = llm.invoke(prompt)
    content = response.content.strip().lower()
    
    if "calendar" in content:
        return {"next_agent": "calendar"}
    elif "email" in content:
        return {"next_agent": "email"}
    else:
        return {"next_agent": "FINISH"}

# Email Subagent Node with Human-in-the-Loop Interrupt
def email_agent_node(state: AgentState):
    last_user_msg = state["messages"][-1]["content"]
    
    # Human-In-The-Loop Checkpoint
    approval = interrupt({
        "question": f"HUMAN APPROVAL REQUIRED: Approve sending email regarding: '{last_user_msg}'?",
        "action": "send_email"
    })
    
    if approval.get("approved", False):
        res = send_email.invoke({"to": "team@company.com", "subject": "Action Item Update", "body": last_user_msg})
        return {"messages": [{"role": "assistant", "content": res}], "next_agent": "FINISH"}
    else:
        return {"messages": [{"role": "assistant", "content": "[CANCELLED] Action rejected by human supervisor."}], "next_agent": "FINISH"}

# Calendar Subagent Node
def calendar_agent_node(state: AgentState):
    # Perform RAG lookup first
    kb_info = search_policy_kb.invoke("meeting strategy")
    res = create_calendar_event.invoke(f"Strategy Meeting - Ref Policy: {kb_info}")
    return {"messages": [{"role": "assistant", "content": res}], "next_agent": "FINISH"}

# Build Workflow Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("calendar", calendar_agent_node)
workflow.add_node("email", email_agent_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "calendar": "calendar",
        "email": "email",
        "FINISH": END
    }
)

workflow.add_edge("calendar", END)
workflow.add_edge("email", END)

# Compile Graph with Checkpointer for Short/Long term memory state persistence
memory_checkpointer = MemorySaver()
app = workflow.compile(checkpointer=memory_checkpointer)

# ------------------------------------------------------------------------------
# 4. End-to-End Test & Execution Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "osamah_session_001"}}

    print("=== 1. Testing Calendar Subagent Path ===")
    events = app.stream(
        {"messages": [{"role": "user", "content": "Schedule the strategy meeting based on policy."}]},
        thread_config
    )
    for event in events:
        print(event)

    print("\n=== 2. Testing Email Subagent Path (Triggers Human-in-the-Loop Interrupt) ===")
    events = app.stream(
        {"messages": [{"role": "user", "content": "Send email update to the team about the launch."}]},
        thread_config
    )
    for event in events:
        print(event)

    print("\n=== 3. Simulating Human Approval Resume ===")
    # Resuming execution after interrupt
    events = app.stream(Command(resume={"approved": True}), thread_config)
    for event in events:
        print(event)
