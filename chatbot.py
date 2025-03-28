import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

# Set API keys (replace with your own or load from environment variables)
groq_api_key = os.getenv("GROQ_API_KEY","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")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "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")

os.environ["langchain_api_key"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LANGRAPH_CHATBOT"

# Model name
model_name = "Gemma2-9b-It"

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Define State for chatbot
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State) -> State:
    user_input = state["messages"][-1].content  # Extract latest user message
    formatted_messages = state["messages"]  # Pass full conversation history
    
    response = llm.invoke(formatted_messages)  # Proper invocation
    state["messages"].append(AIMessage(content=response.content))  # Append AI response
    
    return state

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

executor = graph_builder.compile()

# Streamlit UI
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")

# Sidebar for model information
with st.sidebar:
    st.header("🤖 Chatbot Info")
    st.write(f"**Model Name:** {model_name}")
    st.write("💬 **Welcome to the AI Chatbot!** Ask me anything and I'll try to help.")

st.title("LangGraph Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask me something:")
if st.button("Send") and user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    result = executor.invoke({"messages": st.session_state.messages})
    
    st.session_state.messages.append(result["messages"][-1])  # Append AI response

st.subheader("Chat History")
for msg in st.session_state.messages:
    role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 AI"
    st.write(f"**{role}:** {msg.content}")

