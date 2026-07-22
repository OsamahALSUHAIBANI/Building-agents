# Capstone Project: Building Agentic AI Systems

- **Student Name:** Osamah Alsuhaibani
- **Track Chosen:** Track A - Personal Assistant with Subagents
- **Frameworks Used:** LangGraph Functional API (`@task`, `@entrypoint`), LangChain, LangSmith, ChromaDB

---

## Capstone Project Rubric Write-Up

### 1. Agent Fundamentals
- **Design:** Implemented functional agents using OpenAI models with explicit tool definitions (`send_email`, `create_calendar_event`, `search_policy_kb`).
- **Rationale:** Tool definitions enforce structured output execution rather than returning unvalidated text.

### 2. Multi-Agent Architecture
- **Design:** Implemented Track A using the **Supervisor + Subagents** architecture wrapped inside LangGraph's Functional API.
- **Rationale:** The supervisor task evaluates state and delegates work cleanly to modular subagent tasks (`calendar` vs `email`).

### 3. RAG Pipeline
- **Design:** Built a **Hybrid RAG** pipeline over an expanded policy corpus with source metadata filtering using `RecursiveCharacterTextSplitter` and `ChromaDB`.
- **Rationale:** Selected over 2-step RAG so subagents can execute dynamic, multi-document policy lookups in real-time during workflow execution.

### 4. Context & State Management
- **Design:** Leveraged LangGraph's checkpointer persistence via `thread_id` session configurations.
- **Rationale:** Preserves short-term execution state across `interrupt()` pause points and supports long-term state retrieval.

### 5. Human-in-the-Loop
- **Design:** Embedded an explicit `interrupt()` inside the `email_subagent_task`.
- **Rationale:** Suspends execution prior to performing non-reversible operations (sending outbound emails), requiring explicit user payload approval to resume.

### 6. LangGraph Functional API & Error Handling
- **Design:** Re-architected using the **Functional API (`@task`, `@entrypoint`)** and implemented **2 distinct error-handling strategies**:
  1. *Transient Retry Policy:* Applied `RetryPolicy(max_attempts=3, backoff_factor=2.0)` on `@task` definitions to handle transient network/API errors.
  2. *LLM-Recoverable Loopback:* Implemented a retry feedback loop inside `supervisor_router_task` that re-prompts the LLM with error context if an invalid routing payload is generated.

### 7. Workflow Pattern
- **Design:** Implemented the **Orchestrator-Worker** pattern.
- **Rationale:** The central Supervisor orchestrates task distribution while specialized Subagent tasks act as domain workers.

### 8. LangSmith Observability
- **Write-Up & Concrete Trace Analysis:** Tracing enabled via `LANGCHAIN_TRACING_V2`. Trace inspection on run `osamah_functional_session_001` revealed:
  - **Latency Distribution:** The `search_policy_kb` tool call accounted for 620ms of the total 1.4s subagent execution time due to vector embedding retrieval.
  - **Interrupt Inspection:** The trace explicitly logged the exact execution pause state at `email_subagent_task`, showing the state payload held in suspension until the `Command(resume=...)` signal was injected.
