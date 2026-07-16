import os, warnings, logging
os.environ["OPENAI_API_KEY"] = "dummy"
warnings.filterwarnings('ignore')
for lg in ["httpx","openai","httpcore","chromadb","urllib3"]:
    logging.getLogger(lg).setLevel(logging.ERROR)

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["backend_url"] = "http://localhost:11434/v1"
config["deep_think_llm"] = "qwen2.5:72b"
config["quick_think_llm"] = "qwen2.5:14b"
config["max_debate_rounds"] = 1
config["poisoned_news"] = False
config["data_vendors"] = {
    "core_stock_apis": "local",
    "technical_indicators": "local",
    "fundamental_data": "local",
    "news_data": "local",
}

print("Testing BTC-USD | centralized | market+news only...", flush=True)
try:
    ta = TradingAgentsGraph(debug=False, config=config, graph_structure="centralized",
                            selected_analysts=["market", "news"])
    print("Graph created, running propagate...", flush=True)
    _, decision = ta.propagate("BTC-USD", "2025-01-02")
    print(f">>> Decision: {decision}", flush=True)
except Exception as e:
    import traceback
    print(f">>> ERROR: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
