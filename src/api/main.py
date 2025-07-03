from fastapi import FastAPI, Request
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from langgraph_engine.graph_builder import build_agent_graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
graph = build_agent_graph()

@app.post("/api/message")
async def process(request: Request):
    body = await request.json()
    message = body["message"]
    
    # Create initial state with query for the new structure
    initial_state = {
        "query": message,
        "messages": [],
        "current_agent": "",
        "agent_outputs": {},
        "next_agent": "",
        "dataframe_info": {},
        "has_data": False,
        "final_result": "",
        "metadata": {},
        "iteration_count": 0
    }
    
    # Invoke the graph with the initial state
    result = graph.invoke(initial_state)
    return result


