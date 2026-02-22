import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import convert_to_messages
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

load_dotenv()

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


async def run_agent(query):
    
    client = MultiServerMCPClient(
    {
        "bright_data": {
            "command": "npx",
            "args": ["@brightdata/mcp"],
            "env": {
                "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                "PRO_MODE": "false",        # disable premium tools
                "RATE_LIMIT": "10/1h",      # lower safe limit
                "POLLING_TIMEOUT": "600"
                },

            "transport": "stdio",
        },
    }
)
    tools = await client.get_tools()
    
    model=init_chat_model(model="llama-3.1-8b-instant",model_provider="groq",api_key=os.getenv("GROQ_API_KEY"))
    
    stock_finder_agent = create_agent(model, tools=[],system_prompt="""You are an intraday trader specializing in the Indian Stock Market, particularly NSE-listed stocks (National Stock Exchange of India). Your task is to identify 2 high-probability stocks suitable for intraday trading (same-day buy/sell opportunities).
    Selection criteria should include:
    High trading volume and liquidity
    - Strong price momentum or volatility
    - Breakout or breakdown patterns
    - Pre-market or recent news impact
    - Technical indicators (VWAP, RSI, moving averages, support/resistance levels)
    Avoid penny stocks and illiquid companies.
    Output should include:
    - Stock name
    -Ticker symbol
    - Trade direction (Buy/Sell)
    - Entry range
    Target price
    - Stop loss
    - Brief reasoning for the trade
    Respond in structured plain text format.""",name="stock_finder_agent")
    
    market_data_agent = create_agent(model, tools=[],system_prompt=""" You are an intraday market data analyst specializing in Indian stocks listed on the National Stock Exchange of India (NSE). Given a list of stock tickers (e.g., RELIANCE, INFY), your task is to gather and analyze real-time and intraday trading data for each stock.
    Focus on information relevant for same-day trading decisions, including:
    - Current market price (INR)
    - Previous closing price
    - Intraday price movement (open, high, low, current trend)
    - Today's trading volume and volume comparison with average volume
    - Intraday price momentum (bullish/bearish/sideways)
    - Short-term price trend (1-day and 5-day trend)
    - Key technical indicators relevant for intraday trading:
    - RSI (short timeframe)
    - VWAP
    - 20/50-day moving averages
    - Key support and resistance levels
    - Notable spikes in volume, volatility, or price action
    - Breakout or breakdown signals (if any)
    Return findings in a structured and readable format for each stock, suitable for further analysis by an intraday trading recommendation engine.
    Use INR as the currency. Be concise but complete.""",name="market_data_agent")
    
    news_analyst_agent = create_agent(model, tools=[],system_prompt="""You are an intraday financial news analyst specializing in Indian stocks listed on the National Stock Exchange of India (NSE).
    Given the names or ticker symbols of NSE-listed stocks, your task is to:
    - Search for the most recent news (last 24–72 hours or current trading day where possible)
    - Summarize key updates, announcements, or events affecting each stock
    - Classify each news item as Positive, Negative, or Neutral for intraday trading
    - Identify whether the news can trigger immediate price movement, volatility, or trading momentum
    - Highlight potential intraday impact (bullish momentum, bearish pressure, high volatility, breakout catalyst, etc.)
    - Note any event-driven triggers such as earnings announcements, management updates, regulatory actions, large orders, or sector news
    Present your response in a clear, structured format with one section per stock.
    Use bullet points where necessary. Keep the analysis concise, factual, and focused on same-day trading impact.""",name="news_analyst_agent")
    
    
    price_recommender_agent = create_agent(model, tools=[],system_prompt=""" You are an intraday trading strategy advisor for the Indian Stock Market, focusing on NSE-listed stocks.
    You are given:
    - Recent market data (current price, intraday volume, price movement, trend, technical indicators)
    - Latest news summaries and sentiment for each stock
    Based on this information, for each stock:
    1. Recommend an intraday action: Buy, Sell, or Hold
    2. Suggest a specific entry price range (INR)
    3. Suggest a target price (INR)
    4. Suggest a stop loss level (INR)
    5. Briefly explain the reasoning based on technical indicators, price momentum, volume, and news sentiment
    6. Highlight potential intraday risks or volatility factors (if any)
    Your goal is to provide practical, same-day trading advice for intraday execution.
    Keep the response concise and clearly structured.""",name="price_recommender_agent")
    
    
    supervisor_model = init_chat_model(
        "llama-3.1-8b-instant",
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY")
        )
    
    supervisor = create_supervisor(
        model=supervisor_model,
        agents=[stock_finder_agent, market_data_agent, news_analyst_agent, price_recommender_agent],
        system_prompt=("""
                You are a supervisor managing four specialized intraday trading agents for NSE-listed stocks.
                Agents under your control:

                - stock_finder_agent:
                Assign stock scanning and research tasks to this agent to identify 2 high-probability NSE stocks suitable for intraday trading based on liquidity, volume, volatility, and momentum.

                - market_data_agent:
                Assign tasks to fetch real-time and intraday market data (current price, intraday movement, volume, technical indicators, trends).

                - news_analyst_agent:
                Assign tasks to search and summarize recent news (last 24–72 hours) and analyze its intraday price impact and sentiment.

                -  price_recommender_agent:
                Assign tasks to generate intraday trading decisions (Buy/Sell/Hold) with entry range, target price, and stop loss.
    
                Rules:
                - Assign work to one agent at a time.
                - Do not call agents in parallel.
                - Do not perform any analysis or work yourself.
                - Ensure the full workflow is completed from stock selection to final recommendation.
                - Do not ask for user confirmation or pause between steps."""),
        add_handoff_back_messages=True,
        output_mode="full_history",
        handoff_mode="router" 
        ).compile()
    
    
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                    }
                ]
            },
        ):
        
        pretty_print_messages(chunk, last_message=False)
        
        for node in chunk.values():
            if isinstance(node, dict) and "messages" in node:
                final_message_history = node["messages"]

if __name__ == "__main__":
    asyncio.run(run_agent("Give me a good stock recommendation from NSE"))