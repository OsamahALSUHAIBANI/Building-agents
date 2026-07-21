# Capstone Project: Building Agentic AI Systems

- **Student Name:** Osamah Alsuhaibani
- **Track Chosen:** Track A - Personal Assistant with Subagents
- **Frameworks Used:** LangGraph, LangChain, LangSmith, ChromaDB

---

## Capstone Project Rubric Write-Up

### 1. Agent Fundamentals
- **Design:** Implemented functional agents using OpenAI models with explicit tool definitions (`send_email`, `create_calendar_event`, `search_policy_kb`).
- **Rationale:** Tool definitions allow structured output execution rather than returning dynamic, unvalidated text.

### 2. Multi-Agent Architecture
- **Design:** Selected **Track A (Supervisor + Subagents)**.
- **Rationale:** A dedicated Supervisor inspects incoming state and routes tasks dynamically to domain-specific subagents (`calendar` vs `email`), keeping worker nodes decoupled.

### 3. RAG Pipeline
- **Design:** Implemented a **Hybrid RAG** pipeline using `RecursiveCharacterTextSplitter` and `ChromaDB`.
- **Rationale:** Chosen over 2-Step RAG to allow agents to perform ad-hoc retrieval dynamically whenever context requires validation against company guidelines.

### 4. Context & State Management
- **Design:** Used LangGraph's `MemorySaver` checkpointer.
- **Rationale:** Preserves graph thread persistence (`thread_id`), allowing state restoration across multi-turn interactions and `interrupt()` human-in-the-loop cycles.

### 5. Human-in-the-Loop
- **Design:** Integrated `interrupt()` inside the Email Subagent node.
- **Rationale:** Suspends execution before triggering non-reversible external actions (sending emails), requiring explicit user approval to resume.

### 6. LangGraph Functional API & Error Handling
- **Design:** Applied **LLM-recoverable loopback** routing alongside **User-fixable interrupts**.
- **Rationale:** Ensures valid dynamic state transitions while giving users full control to resolve execution halts gracefully.

### 7. Workflow Pattern
- **Design:** Implemented the **Orchestrator-Worker** pattern.
- **Rationale:** The central Supervisor orchestrates task distribution while individual subagents operate as specialized workers completing domain tasks.

### 8. LangSmith Observability
- **Design:** Enabled full tracing via `LANGCHAIN_TRACING_V2`.
- **Rationale:** Trace analyses revealed latency distribution across subagent steps and confirmed clean argument parsing during function calling.
