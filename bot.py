from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

# Store session histories globally
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 1. Initialize Groq Engine Directly
model = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.7
)

# 2. Prompt & Runnable Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI chatbot running on Groq."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat(user_input: str, session_id: str = "cli_session"):
    try:
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("⚡ Groq Chatbot Initialized (Type 'exit' or 'quit' to stop)\n" + "-"*50)
    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        print("Bot: ", end="", flush=True)
        try:
            for chunk in conversational_chain.stream(
                {"input": user_msg},
                config={"configurable": {"session_id": "cli_session"}}
            ):
                print(chunk.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error: {e}")