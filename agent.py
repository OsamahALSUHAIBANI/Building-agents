# ==============================================================================
# Capstone Project: Building Agentic AI Systems (Functional API Refactor)
# Track Chosen: Track A - Personal Assistant with Subagents
# Student Name: Osamah Alsuhaibani
# ==============================================================================

import os
from typing import Dict, List, TypedDict, Annotated
import operator

# ------------------------------------------------------------------------------
# 1. Environment Setup & LangSmith Observability
# ------------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key_here"
os.environ["LANGCHAIN_PROJECT"] = "Capstone-Project-Osamah"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# ------------------------------------------------------------------------------
# 2. Expanded RAG Pipeline Setup (Vectorstore & Retries)
# ------------------------------------------------------------------------------
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

# Expanded Corporate Knowledge Base
docs = [
    Document(page_content="Meeting Policy: Strategy and executive syncs are held on Tuesdays at 10 AM in Room 4B. Mandatory 15-min buffer required between meetings.", metadata={"source": "calendar_policy"}),
    Document(page_content="Calendar Support: Recurring team syncs should always be scheduled with a 15-minute buffer and clear agenda sent 24h prior.", metadata={"source": "calendar_policy"}),
    Document(page_content="Email Policy: High-priority client inquiries require explicit manager approval before sending responses.", metadata={"source": "email_policy"}),
    Document(page_content="Outbound Communication: External emails containing contract details or status updates must pass human review.", metadata={"source": "email_policy"}),
    Document(page_content="Working Hours & Escalations: Escalations outside 9 AM - 5 PM PST require approval from the duty manager.", metadata={"source": "general_policy"})
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def search_policy_kb(query: str) -> str:
    """Retrieves corporate calendar and email policies from the expanded knowledge base."""
    results = retriever.invoke(query)
    return "\n\n".join([f"[{doc.metadata.get('source', 'policy')}]: {doc.page_content}" for doc in results])

@tool
def create_calendar_event(event_details: str) -> str:
    """Schedules a new calendar event."""
    return f"[SUCCESS] Event successfully scheduled: '{event_details}'"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an outbound email."""
    return f"[SUCCESS] Email sent to {to} | Subject: {subject} | Body: {body}"

# ------------------------------------------------------------------------------
# 3. Functional API, Retry Policies & Error Handling Strategies
# ------------------------------------------------------------------------------
from langgraph.func import task, entrypoint
from langgraph.types import interrupt, Command, RetryPolicy

# Strategy 1: Transient Retry Policy for LLM & Network Calls
llm_retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=(Exception,)
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Subagent Task: Calendar
@task(retry=llm_retry_policy)
def calendar_subagent_task(user_query: str) -> str:
    kb_info = search_policy_kb.invoke(user_query)
    res = create_calendar_event.invoke(f"Scheduled Event - KB Reference: {kb_info}")
    return res

# Subagent Task: Email (with User-Fixable Interrupt)
@task(retry=llm_retry_policy)
def email_subagent_task(user_query: str) -> str:
    # Human-In-The-Loop Interrupt Point
    approval = interrupt({
        "question": f"HUMAN APPROVAL REQUIRED: Approve sending email regarding: '{user_query}'?",
        "action": "send_email"
    })
    
    if approval.get("approved", False):
        res = send_email.invoke({"to": "team@company.com", "subject": "Action Item Update", "body": user_query})
        return res
    else:
        return "[CANCELLED] Action rejected by human supervisor."

# Supervisor Task (with Strategy 2: LLM-Recoverable Loopback Error Handling)
@task(retry=llm_retry_policy)
def supervisor_router_task(messages: List[Dict]) -> str:
    max_retries = 3
    attempt = 0
    feedback = ""
    
    while attempt < max_retries:
        prompt = f"""You are a Supervisor Agent overseeing two specialized assistants: 'calendar' and 'email'.
Based on the conversation history, decide the appropriate subagent to handle the request.
Respond ONLY with one word: 'calendar', 'email', or 'FINISH'.

{feedback}
History: {messages}"""
        
        response = llm.invoke(prompt)
        content = response.content.strip().lower()
        
        if any(target in content for target in ["calendar", "email", "finish"]):
            if "calendar" in content: return "calendar"
            if "email" in content: return "email"
            return "FINISH"
        
        # Loopback Recovery: Feed back the error to the LLM
        attempt += 1
        feedback = f"ERROR: Your previous response '{content}' was invalid. You must reply strictly with 'calendar', 'email', or 'FINISH'."
    
    return "FINISH" # Safe fallback

# ------------------------------------------------------------------------------
# 4. Entrypoint Definition (LangGraph Functional API)
# ------------------------------------------------------------------------------
@entrypoint()
def personal_assistant_workflow(inputs: Dict) -> Dict:
    messages = inputs.get("messages", [])
    last_user_msg = messages[-1]["content"] if messages else ""
    
    # Execute Supervisor Task
    route = supervisor_router_task(messages).result()
    
    if route == "calendar":
        result = calendar_subagent_task(last_user_msg).result()
        return {"response": result, "route_taken": "calendar"}
    elif route == "email":
        result = email_subagent_task(last_user_msg).result()
        return {"response": result, "route_taken": "email"}
    else:
        return {"response": "Workflow finished without subagent invocation.", "route_taken": "FINISH"}

# ------------------------------------------------------------------------------
# 5. End-to-End Test Execution Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "osamah_functional_session_001"}}

    print("=== 1. Testing Calendar Subagent Path (Functional API) ===")
    res_1 = personal_assistant_workflow.invoke(
        {"messages": [{"role": "user", "content": "Schedule the strategy meeting based on policy."}]},
        thread_config
    )
    print("Result:", res_1)

    print("\n=== 2. Testing Email Subagent Path (Triggers Interrupt) ===")
    res_2 = personal_assistant_workflow.invoke(
        {"messages": [{"role": "user", "content": "Send email update to the team about the launch."}]},
        thread_config
    )
    print("Result (Paused at Interrupt):", res_2)

    print("\n=== 3. Resuming Workflow with Human Approval ===")
    res_3 = personal_assistant_workflow.invoke(
        Command(resume={"approved": True}),
        thread_config
    )
    print("Result (Post-Approval):", res_3)
