import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import convert_to_messages
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor


# ================================
# LOAD ENV VARIABLES
# ================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not BRIGHT_DATA_API_TOKEN:
    raise ValueError("BRIGHT_DATA_API_TOKEN not found in .env")


# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="NSE Trading AI", layout="wide")

st.title("📈 Multi-Agent Intraday Trading System")

st.write("Ask for intraday trading recommendations for NSE stocks.")

query = st.text_input(
    "🔍 Enter stock or market query",
    placeholder="Example: Give intraday recommendation for RELIANCE"
)

run_btn = st.button("🚀 Run Trading Analysis")


# ================================
# PRETTY PRINT STREAM OUTPUT
# ================================
def collect_messages(update):
    output = ""

    # Handle subgraph updates
    if isinstance(update, tuple):
        ns, update = update

        if not update or len(ns) == 0:
            return ""

        graph_id = ns[-1].split(":")[0]
        output += f"\n### Update from subgraph {graph_id}\n"

    if not isinstance(update, dict):
        return ""

    for node_name, node_update in update.items():

        # ✅ skip empty updates
        if not node_update:
            continue

        # ✅ ensure dictionary
        if not isinstance(node_update, dict):
            continue

        # ✅ ensure messages exist
        if "messages" not in node_update:
            continue

        if node_update["messages"] is None:
            continue

        output += f"\n**Node:** {node_name}\n"

        try:
            messages = convert_to_messages(node_update["messages"])
        except Exception:
            continue

        for m in messages:
            if hasattr(m, "content") and m.content:
                output += f"\n{m.content}\n"

    return output



# ================================
# MAIN AGENT
# ================================
async def run_agent(query):

    # ---------- MCP CLIENT ----------
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": BRIGHT_DATA_API_TOKEN,
                    "PRO_MODE": "false",
                    "RATE_LIMIT": "10/1h",
                    "POLLING_TIMEOUT": "600",
                },
                "transport": "stdio",
            },
        }
    )

    await client.get_tools()

    # ---------- MODEL ----------
    model = init_chat_model(
        model="llama-3.1-8b-instant",
        model_provider="groq",
        api_key=GROQ_API_KEY
    )

    # ---------- AGENTS ----------
    stock_finder_agent = create_agent(
        model,
        tools=[],
        name="stock_finder_agent",
        system_prompt="""You are an intraday trader specializing in NSE stocks.
Identify 2 high-probability intraday stocks with entry, target, stop loss, and reasoning."""
    )

    market_data_agent = create_agent(
        model,
        tools=[],
        name="market_data_agent",
        system_prompt="""Analyze NSE intraday market data including price, volume,
trend, RSI, VWAP, and support/resistance."""
    )

    news_analyst_agent = create_agent(
        model,
        tools=[],
        name="news_analyst_agent",
        system_prompt="""Analyze recent news and sentiment for NSE stocks."""
    )

    price_recommender_agent = create_agent(
        model,
        tools=[],
        name="price_recommender_agent",
        system_prompt="""Provide Buy/Sell/Hold recommendation with entry,
target price, and stop loss."""
    )

    # ---------- SUPERVISOR ----------
    supervisor_model = init_chat_model(
        "llama-3.1-8b-instant",
        model_provider="groq",
        api_key=GROQ_API_KEY
    )

    supervisor = create_supervisor(
        model=supervisor_model,
        agents=[
            stock_finder_agent,
            market_data_agent,
            news_analyst_agent,
            price_recommender_agent,
        ],
        system_prompt="""You manage trading agents.
Complete full workflow from stock selection to final recommendation.""",
        add_handoff_back_messages=True,
        output_mode="full_history",
        handoff_mode="router",
    ).compile()

    # ---------- STREAM RESULTS ----------
    full_output = ""

    async for chunk in supervisor.astream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        full_output += collect_messages(chunk)

    return full_output


# ================================
# RUN BUTTON
# ================================
if run_btn:

    if not query:
        st.warning("Enter a stock query.")
        st.stop()

    with st.spinner("Running multi-agent trading analysis..."):
        result = asyncio.run(run_agent(query))

    st.subheader("📊 Analysis Result")
    st.markdown(result)
