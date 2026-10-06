# ⚡ AI Agent Chatbot

[![CI](https://github.com/Amankhan0087/ai_agent_chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Amankhan0087/ai_agent_chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A fast, conversational AI assistant built with **Streamlit**, **LangChain** and **Groq**. It streams answers, keeps
your chats across restarts, remembers facts about you over time, and uses a **ReAct agent with live web search**
when a question needs fresh information.

## Features

- **Streaming responses** powered by Groq's low-latency inference (`openai/gpt-oss-20b`, with `openai/gpt-oss-120b` as fallback).
- **ReAct agent with web search**: the model decides when to search, shows its search steps live, and cites sources.
- **Persistent chat sessions**: every conversation is stored in SQLite and listed in the sidebar; the latest chat reopens on load.
- **Long-term memory**: durable facts about the user (name, profession, preferences) are extracted after each reply,
  stored, and injected into future conversations. View, forget or disable them from the sidebar.
- **Export** any chat as Markdown, delete chats, start new ones.
- **Resilient by design**: startup model validation with fallback, capped agent loop, one automatic retry, and failed
  replies are never saved into history.

## Architecture

```
+-------------+     +----------------------------+     +-------------+
|  Streamlit  | --> |  ReAct agent (LangChain)   | --> |  Groq LLM   |
|   app.py    |     |  create_agent + tools.py   |     +-------------+
+------+------+     +-------------+--------------+
       |                          |
       |                          +--> web_search (DuckDuckGo via ddgs)
       v
+-------------------------------+
|  storage.py (SQLite)          |   sessions | messages | long-term memory
+-------------------------------+
```

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, agent runner, memory extraction, sidebar |
| `storage.py` | SQLite persistence for sessions, messages and memory facts |
| `tools.py` | Tools available to the agent (`web_search`) |
| `bot.py` | Minimal streaming CLI version (no agent, sessions or memory) |
| `tests/` | Pytest suite for storage and tools |

## Quick start

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/Amankhan0087/ai_agent_chatbot.git
cd ai_agent_chatbot

python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your GROQ_API_KEY in .env
streamlit run app.py
```

Open http://localhost:8501. For the CLI version run `python bot.py`.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Your Groq API key |
| `CHATBOT_DB_PATH` | no | SQLite file location (default `./chatbot.db`) |

Models are listed in `get_groq_llm()` in `app.py`; the first one that responds is used.

## Deploy for free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Pick this repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings → Secrets** add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**.

> **Important: read before sharing the link**
> - The SQLite database is **shared by everyone who opens the app** and is **wiped when the app restarts or
>   redeploys** (Community Cloud storage is ephemeral). Treat a public deployment as a demo, not a multi-user product.
> - Anyone with the link can spend your Groq quota. In the app's **Share** settings restrict viewing to specific
>   email addresses, or keep the app private.
> - For real multi-user use, swap `storage.py` for a hosted database (e.g. Postgres/Supabase) and add per-user
>   authentication.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

CI (GitHub Actions) runs lint and tests on Python 3.11 and 3.12 for every push and pull request.

## Roadmap

- Per-user accounts and a hosted database
- More agent tools (calculator, document/file search)
- Higher-quality search provider (e.g. Tavily) as an option
- Migrate `bot.py` to the agent and shared storage

## License

[MIT](LICENSE)
