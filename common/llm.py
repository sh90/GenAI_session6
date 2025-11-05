import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Load API keys
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["TAVILY_API_KEY"] = "tvly-..."

# 1. Define the LLM
# We use gpt-4o-mini for its speed and intelligence
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Define the Agent Prompt
# This is crucial. It tells the agent how to behave and
# includes placeholders for history, input, and the ReAct "scratchpad".
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You must plan your work step-by-step."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])