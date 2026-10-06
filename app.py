import json
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

import storage
from tools import TOOLS

# Page configuration
st.set_page_config(page_title="Groq LLM Chatbot", page_icon="⚡", layout="centered")

load_dotenv()
storage.init_db()

BASE_SYSTEM_PROMPT = (
    "You are a witty, fast, and helpful AI assistant demonstrating a live technical session."
)
SEARCH_INSTRUCTIONS = (
    "You have a web_search tool. Use it for anything recent, time-sensitive or that you are unsure "
    "about (news, scores, prices, releases); do not search for casual chat or stable knowledge. "
    "Search results can be outdated, so prefer the most recent sources. "
    "When you rely on search results, cite the sources as markdown links."
)
MAX_HISTORY_MESSAGES = 30  # most recent messages sent to the model as context
AGENT_RECURSION_LIMIT = 12  # caps the reason/act loop so a confused agent can't spin forever
AGENT_ATTEMPTS = 2  # one retry for transient failures (e.g. a malformed tool call from the model)
MAX_FACTS_PER_TURN = 3

MEMORY_EXTRACTION_PROMPT = """You maintain long-term memory for a chat assistant.
From the exchange below, extract durable facts about the USER that will be useful in future \
conversations (name, profession, skills, preferences, projects, goals, constraints, language). \
Ignore small talk, one-off questions, and anything about the assistant itself.
Merge related details into a single fact (e.g. name and profession together) and never list the same information twice.
Return ONLY a JSON array of up to {limit} short, self-contained strings. Return [] if nothing is worth remembering.
Facts already known (do not repeat them): {known}

User: {user}
Assistant: {assistant}"""

st.title("⚡ Ultra-Fast Groq Chatbot")


# 1. Initialize Groq LLM Safely
@st.cache_resource(show_spinner=False)
def get_groq_llm():
    # Model fallbacks array in case one ID is restricted on your account
    groq_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    for m_name in groq_models:
        try:
            llm = ChatGroq(model_name=m_name, temperature=0.7)
            llm.invoke("ping")  # ChatGroq doesn't validate the model until first call
            return llm, f"Groq ({m_name})"
        except Exception:
            continue

    return None, "Disconnected"


model, active_provider = get_groq_llm()


# 2. Session state & callbacks
if "session_id" not in st.session_state:
    # Resume the most recent conversation; None means "new chat" (created on first message)
    recent = storage.list_sessions(limit=1)
    st.session_state.session_id = recent[0]["id"] if recent else None


def start_new_chat():
    st.session_state.session_id = None


def open_chat(session_id: str):
    st.session_state.session_id = session_id


def delete_chat(session_id: str):
    storage.delete_session(session_id)
    if st.session_state.session_id == session_id:
        st.session_state.session_id = None


# 3. Helpers
def build_system_prompt(facts: list[str], search_enabled: bool) -> str:
    system = f"{BASE_SYSTEM_PROMPT}\nToday's date is {date.today().isoformat()}."
    if search_enabled:
        system += "\n\n" + SEARCH_INSTRUCTIONS
    if facts:
        system += (
            "\n\nThings you remember about the user from earlier conversations "
            "(use naturally when relevant, never recite them as a list):\n"
            + "\n".join(f"- {fact}" for fact in facts)
        )
    return system


def build_messages(history: list[dict], user_query: str) -> list:
    messages = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=user_query))
    return messages


def run_agent(history: list[dict], user_query: str, facts: list[str], search_enabled: bool) -> str:
    """Run the ReAct agent, rendering tool activity and the streamed answer into the current
    chat message. Returns the final answer text; raises if every attempt fails."""
    agent = create_agent(
        model,
        tools=TOOLS if search_enabled else [],
        system_prompt=build_system_prompt(facts, search_enabled),
    )
    payload = {"messages": build_messages(history, user_query)}
    config = {"recursion_limit": AGENT_RECURSION_LIMIT}

    tool_area = st.container()
    answer_box = st.empty()
    status = None

    for attempt in range(1, AGENT_ATTEMPTS + 1):
        text = ""
        try:
            for mode, data in agent.stream(payload, config=config, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, meta = data
                    if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model" and chunk.content:
                        text += chunk.content
                        answer_box.markdown(text + "▌")
                    continue

                # mode == "updates": a full step finished (model decision or tool result)
                for node, update in data.items():
                    for msg in (update or {}).get("messages", []):
                        if node == "model" and getattr(msg, "tool_calls", None):
                            text = ""  # drop any preamble the model wrote before calling a tool
                            answer_box.empty()
                            if status is None:
                                status = tool_area.status("Working…", expanded=True)
                            for call in msg.tool_calls:
                                query = call["args"].get("query", call["args"])
                                status.update(label=f"🔎 Searching: {query}")
                                status.write(f"🔎 **{call['name']}** → `{query}`")
                        elif isinstance(msg, ToolMessage):
                            preview = " ".join(str(msg.content).split())[:240]
                            status.write(f"📄 {preview}…")
            if text:
                answer_box.markdown(text)
                if status is not None:
                    status.update(label="Searched the web", state="complete", expanded=False)
                return text
            raise RuntimeError("The model returned an empty response.")
        except Exception:
            if attempt == AGENT_ATTEMPTS:
                if status is not None:
                    status.update(label="Search failed", state="error")
                raise
            answer_box.empty()
    raise RuntimeError("unreachable")  # pragma: no cover


def extract_facts(user_text: str, assistant_text: str, known: list[str]) -> list[str]:
    """Ask the model for durable user facts worth remembering. Never raises."""
    try:
        raw = model.invoke(
            MEMORY_EXTRACTION_PROMPT.format(
                limit=MAX_FACTS_PER_TURN,
                known="; ".join(known) or "none",
                user=user_text,
                assistant=assistant_text,
            )
        ).content
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            return []
        facts = json.loads(raw[start : end + 1])
        return [f.strip()[:200] for f in facts if isinstance(f, str) and f.strip()][:MAX_FACTS_PER_TURN]
    except Exception:
        return []


def chat_as_markdown(messages: list[dict]) -> str:
    return "\n\n".join(
        f"**{'You' if m['role'] == 'user' else 'Assistant'}:**\n{m['content']}" for m in messages
    )


# 4. Render Conversation History
session_id = st.session_state.session_id
history = storage.get_messages(session_id) if session_id else []

if not history:
    st.caption("Start a conversation below. Your chats are saved and listed in the sidebar.")

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input & Execution Loop
use_memory = st.session_state.get("use_memory", True)
use_search = st.session_state.get("use_search", True)

if model is None:
    st.error("Could not connect to any Groq model. Check GROQ_API_KEY in your .env file and restart.")

if user_query := st.chat_input("Ask Groq something...", disabled=model is None):
    with st.chat_message("user"):
        st.markdown(user_query)

    known_facts = [m["fact"] for m in storage.list_memories()] if use_memory else []

    with st.chat_message("assistant"):
        try:
            ai_text = run_agent(history, user_query, known_facts, use_search)
        except Exception as e:
            ai_text = None
            st.markdown(f"⚠️ **Execution Exception:** `{e}`")

    # Only successful exchanges are persisted, so errors never pollute the history/context.
    if ai_text:
        if session_id is None:
            session_id = st.session_state.session_id = storage.create_session()
        storage.add_message(session_id, "user", user_query)
        storage.add_message(session_id, "assistant", ai_text)
        if use_memory:
            storage.add_memories(extract_facts(user_query, ai_text, known_facts))

# 6. Sidebar (rendered last so it reflects anything saved during this run)
with st.sidebar:
    if model:
        st.success(f"🟢 Active Engine: **{active_provider}**")
    else:
        st.error("🔴 Disconnected")

    st.button("➕ New chat", use_container_width=True, on_click=start_new_chat)

    st.subheader("Chats")
    sessions = storage.list_sessions()
    if not sessions:
        st.caption("No saved chats yet.")
    for s in sessions:
        is_active = s["id"] == st.session_state.session_id
        col_title, col_delete = st.columns([5, 1])
        col_title.button(
            s["title"],
            key=f"open_{s['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=open_chat,
            args=(s["id"],),
        )
        col_delete.button(
            "🗑",
            key=f"del_{s['id']}",
            help="Delete this chat",
            on_click=delete_chat,
            args=(s["id"],),
        )

    st.divider()
    st.toggle(
        "🔎 Web search (agent)",
        value=True,
        key="use_search",
        help="Lets the ReAct agent search the web for recent or unknown information.",
    )
    st.toggle(
        "🧠 Long-term memory",
        value=True,
        key="use_memory",
        help="Remember useful facts about you across chats and use them in replies.",
    )
    memories = storage.list_memories()
    with st.expander(f"Remembered facts ({len(memories)})"):
        if not memories:
            st.caption("Nothing remembered yet. Tell me about yourself!")
        for mem in memories:
            col_fact, col_forget = st.columns([5, 1])
            col_fact.markdown(f"- {mem['fact']}")
            col_forget.button(
                "✖", key=f"forget_{mem['id']}", help="Forget this", on_click=storage.delete_memory, args=(mem["id"],)
            )
        if memories:
            st.button("Clear all memory", on_click=storage.clear_memories)

    current = storage.get_messages(st.session_state.session_id) if st.session_state.session_id else []
    if current:
        st.download_button(
            "⬇️ Export this chat",
            data=chat_as_markdown(current),
            file_name="chat.md",
            mime="text/markdown",
            use_container_width=True,
        )
