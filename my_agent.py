from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated
import operator
import math
import wikipediaapi
import requests
import yfinance as yf
import json
import re
from datetime import datetime

# 🆕 Feedback 儲存檔案
FEEDBACK_FILE = "feedback_log.jsonl"


# ============ 0. 自訂 Guardrail ============
class SensitiveDataGuardrail(AgentMiddleware):
    """自訂 Guardrail：偵測敏感資料同危險內容"""

    def __init__(self):
        self.patterns = {
            # --- 敏感資料 ---
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "api_key": r"sk-[a-zA-Z0-9]{20,}",            
            "phone": r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
            "nz_ird": r"\b\d{2,3}-\d{3}-\d{3}\b",

            # --- 粗口 / 侮辱 ---
            "profanity": r"\b(fuck|shit|damn|bastard|asshole)\b",
            "chinese_profanity": r"(屌|仆街|戇鳩|死開|白痴)",

            # --- 危險 / 非法指令 ---
            "dangerous_commands": r"\b(kill|hack|bomb|attack|explode|poison|murder)\b",
            "illegal_activity": r"\b(cocaine|heroin|meth|weapon|gun)\b",

            # --- 自殺 / 自殘 ---
            "self_harm": r"\b(suicide|kill myself|self-harm|end my life)\b",
        }

        self.messages = {
            "email": "your email address",
            "credit_card": "a credit card number",
            "api_key": "an API key",
            "phone": "a phone number",
            "nz_ird": "an IRD number",
            "profanity": "inappropriate language",
            "chinese_profanity": "inappropriate language",
            "dangerous_commands": "a dangerous request",
            "illegal_activity": "an illegal request",
            "self_harm": "a sensitive topic",
        }

    def check(self, text: str):
        """檢查文字係咪 Safe。回傳 (is_safe, reason)"""
        for data_type, pattern in self.patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return False, data_type
        return True, ""

    def before_agent(self, state, runtime):
        """喺 Agent 執行之前檢查 Input"""
        last_message = state["messages"][-1].content

        for data_type, pattern in self.patterns.items():
            if re.search(pattern, last_message, re.IGNORECASE):
                reason = self.messages.get(data_type, data_type)
                print(f"\n🚫 [GUARDRAIL BLOCKED]")
                print(f"   ⚠️ Detected: {data_type}")
                print(f"   ⚠️ Request has been blocked.")
                return {"messages": [AIMessage(
                    content=f"🚫 I cannot process this request. "
                            f"It appears to contain {reason}, which is not allowed. "
                            f"Please rephrase your question."
                )]}
        return None


# ============ 1. Feedback Logger ============
def log_feedback(query: str, response: str, rating: int):
    """將 Feedback 寫入 JSONL 檔案"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "rating": rating
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Feedback recorded: {rating}/10")


# ============ 2. 定義 Tools ============
wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='MyAgent/1.0')


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given query and return a summary."""
    print(f"\n📚 Tool Called: search_wikipedia")
    print(f"   Query: {query}")
    try:
        page = wiki_wiki.page(query)
        if not page.exists():
            search_results = wiki_wiki.search(query, results=5)
            if search_results:
                for candidate in search_results:
                    candidate_page = wiki_wiki.page(candidate)
                    if candidate_page.exists():
                        page = candidate_page
                        break
        if not page.exists():
            return f"No Wikipedia article found for '{query}'."
        summary = page.summary[:500]
        if len(page.summary) > 500:
            summary += "..."
        return f"Wikipedia article for '{page.title}':\n\n{summary}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '25 * 4' or '(10+5)/3'."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Only numbers and + - * / ( ) are allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_count(text: str) -> str:
    """Count the number of words in a given text."""
    print(f"\n📝 Tool Called: word_count")
    print(f"   Text: {text}")
    if not text or not text.strip():
        return "0 words"
    count = len(text.strip().split())
    print(f"   Word count: {count}")
    return f"{count} words"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city using Open-Meteo API (free, no API key needed)."""
    print(f"\n🌤️ Tool Called: get_weather")
    print(f"   City: {city}")
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_response = requests.get(geo_url, timeout=10).json()
        if not geo_response.get("results"):
            return f"City '{city}' not found."
        lat = geo_response["results"][0]["latitude"]
        lon = geo_response["results"][0]["longitude"]
        city_name = geo_response["results"][0]["name"]
        country = geo_response["results"][0].get("country", "")
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        )
        weather_response = requests.get(weather_url, timeout=10).json()
        current = weather_response["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        code = current["weather_code"]
        weather_desc = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }.get(code, f"Unknown (code {code})")
        return f"""
Current weather in {city_name}, {country}:
- Temperature: {temp}°C
- Condition: {weather_desc}
- Humidity: {humidity}%
- Wind Speed: {wind} km/h
"""
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')."""
    print(f"\n📈 Tool Called: get_stock_price")
    print(f"   Symbol: {symbol}")
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        if not info or ("currentPrice" not in info and "regularMarketPrice" not in info):
            return f"Could not find stock data for '{symbol}'."
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency", "USD")
        name = info.get("longName", symbol.upper())
        change = info.get("regularMarketChange", 0)
        change_pct = info.get("regularMarketChangePercent", 0)
        return f"""
{name} ({symbol.upper()}):
- Current Price: {price} {currency}
- Change: {change:.2f} ({change_pct:.2f}%)
"""
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


# ============ 3. 建立 Sub-Agents ============
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Research Agent
research_agent = create_agent(
    model=llm,
    tools=[search_wikipedia],
    system_prompt="You are a research specialist. Use Wikipedia to answer questions about facts, history, and general knowledge."
)

# Search Agent
search_agent = create_agent(
    model=llm,
    tools=[get_weather, get_stock_price],
    system_prompt="""You are a real-time information specialist.

You have TWO tools:
1. `get_weather(city)` — Use for weather questions.
2. `get_stock_price(symbol)` — Use for stock price questions.

CRITICAL: ALWAYS call the appropriate tool. Do NOT answer from your own knowledge.

Examples:
- User: "Weather in Auckland?" → Call `get_weather("Auckland")`
- User: "MSFT stock price?" → Call `get_stock_price("MSFT")`
"""
)

# Math Agent
math_agent = create_agent(
    model=llm,
    tools=[calculate, word_count],
    system_prompt="""You are a math and text specialist.
You have TWO tools:
1. `calculate(expression)` — Use for math calculations like "25 * 4".
2. `word_count(text)` — Use to count words in a sentence.

Examples:
- User: "What is 25 * 4?" → Call `calculate("25 * 4")`
- User: "How many words in 'hello world foo'?" → Call `word_count("hello world foo")`
"""
)


# ============ 4. 建立 Supervisor State ============
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str
    reason: str


# ============ 5. Supervisor 決策邏輯（加咗 Guardrail） ============
def supervisor_node(state: AgentState):
    messages = state["messages"]

    # 🆕 攞 Query
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, tuple):
        last_message = last_msg[1]
    elif hasattr(last_msg, 'content'):
        last_message = last_msg.content
    else:
        last_message = str(last_msg)

    # 🆕 喺最前線做 Guardrail 檢查
    guardrail = SensitiveDataGuardrail()
    is_safe, reason = guardrail.check(last_message)

    if not is_safe:
        print(f"\n🚫 [GUARDRAIL BLOCKED at Supervisor]")
        print(f"   ⚠️ Detected: {reason}")
        print(f"   ⚠️ Request has been blocked.")
        return {"next": "BLOCKED", "reason": reason}

    # 如果已經有 Agent 答過，就 FINISH
    if len(messages) > 1:
        return {"next": "FINISH"}

    query = last_message.lower().strip()

    # 即時資訊
    realtime_keywords = [
        "weather", "temperature", "forecast", "rain", "sunny",
        "stock", "price", "share", "market", "msft", "aapl", "tsla",
        "news", "latest", "today", "now", "current",
    ]
    if any(kw in query for kw in realtime_keywords):
        print(f"🎯 [Supervisor] 決定派去 → search_agent（即時資訊）")
        return {"next": "search_agent"}

    # 數學問題
    math_keywords = ["calculate", "sum", "plus", "add", "minus", "subtract",
                     "multiply", "times", "divide", "*", "+", "-", "/", "=",
                     "word count", "count words", "how many words", "number of words"]
    if any(kw in query for kw in math_keywords):
        print(f"🎯 [Supervisor] 決定派去 → math_agent")
        return {"next": "math_agent"}

    # 知識性問題
    research_keywords = ["who", "what", "when", "where", "history", "tell me about"]
    if any(kw in query for kw in research_keywords):
        print(f"🎯 [Supervisor] 決定派去 → research_agent")
        return {"next": "research_agent"}

    print(f"🎯 [Supervisor] 決定 FINISH（直接答）")
    return {"next": "FINISH"}


# ============ 6. 建立 Sub-Agent Nodes ============
def research_node(state: AgentState):
    print("📚 [Research Agent] 開始處理...")
    result = research_agent.invoke({"messages": state["messages"]})
    print("📚 [Research Agent] 完成")
    return {"messages": [result["messages"][-1]]}


def search_node(state: AgentState):
    print("🔍 [Search Agent] 開始處理...")
    result = search_agent.invoke({"messages": state["messages"]})
    print("🔍 [Search Agent] 完成")
    return {"messages": [result["messages"][-1]]}


def math_node(state: AgentState):
    print("🧮 [Math Agent] 開始處理...")
    result = math_agent.invoke({"messages": state["messages"]})
    print("🧮 [Math Agent] 完成")
    return {"messages": [result["messages"][-1]]}


def finish_node(state: AgentState):
    """當 Supervisor 決定 FINISH，由 LLM 直接答"""
    print("✅ [Finish] 準備回傳答案")
    messages = state["messages"]

    if len(messages) > 1:
        return {"messages": [messages[-1]]}

    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, tuple):
        last_message = last_msg[1]
    elif hasattr(last_msg, 'content'):
        last_message = last_msg.content
    else:
        last_message = str(last_msg)

    response = llm.invoke(last_message)
    return {"messages": [response]}


def blocked_node(state: AgentState):
    """當 Guardrail 觸發，回傳拒答訊息"""
    print("🚫 [Blocked Node] 回傳拒答訊息")
    reason = state.get("reason", "sensitive content")

    # 將 Reason 轉做 Human-readable 訊息
    reason_map = {
        "email": "your email address",
        "credit_card": "a credit card number",
        "api_key": "an API key",
        "phone": "a phone number",
        "nz_ird": "an IRD number",
        "profanity": "inappropriate language",
        "chinese_profanity": "inappropriate language",
        "dangerous_commands": "a dangerous request",
        "illegal_activity": "an illegal request",
        "self_harm": "a sensitive topic",
    }
    readable_reason = reason_map.get(reason, reason)

    return {"messages": [AIMessage(
        content=f"🚫 I'm sorry, but I cannot process this request. "
                f"It appears to contain {readable_reason}, which is not allowed. "
                f"Please rephrase your question."
    )]}


# ============ 7. 建立 Graph ============
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("search_agent", search_node)
workflow.add_node("math_agent", math_node)
workflow.add_node("finish", finish_node)
workflow.add_node("blocked", blocked_node)  # 🆕

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "research_agent": "research_agent",
        "search_agent": "search_agent",
        "math_agent": "math_agent",
        "FINISH": "finish",
        "BLOCKED": "blocked",  # 🆕
    }
)

workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("search_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")
workflow.add_edge("finish", END)
workflow.add_edge("blocked", END)  # 🆕

graph = workflow.compile()


# ============ 8. 測試 ============
if __name__ == "__main__":
    print("="*50)
    print("Multi-Agent System 已啟動！")
    print("輸入 'bye' 或 'exit' 結束對話。")
    print("="*50)

    while True:
        user_input = input("\n你: ")

        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Agent: 再見！")
            break

        if not user_input.strip():
            continue

        result = graph.invoke(
            {"messages": [("user", user_input)]},
            config={"recursion_limit": 10}
        )

        last = result['messages'][-1]
        if isinstance(last, tuple):
            final_content = last[1]
        elif hasattr(last, 'content'):
            final_content = last.content
        else:
            final_content = str(last)

        print(f"\nAgent: {final_content}")

        print("\n📝 Rate this response (1-10), or press Enter to skip:")
        feedback = input("   Rating: ").strip()

        if feedback == "":
            continue

        try:
            rating = int(feedback)
            if 1 <= rating <= 10:
                log_feedback(user_input, final_content, rating)
            else:
                print("⚠️ Please enter a number between 1 and 10.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")