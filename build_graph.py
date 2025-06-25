# build_graph.py
from langgraph.graph import StateGraph, END

from src.nodes import InferenceNode, ConfidenceCheckNode, FallbackNode
from src.graph_state import GraphState

graph = StateGraph(GraphState)

graph.add_node("Inference", InferenceNode())
graph.add_node("ConfidenceCheck", ConfidenceCheckNode())
graph.add_node("Fallback", FallbackNode())

graph.set_entry_point("Inference")

graph.add_edge("Inference", "ConfidenceCheck")

graph.add_conditional_edges(
    "ConfidenceCheck",
    lambda x: x["path"],
    {
        "fallback": "Fallback",
        "accept": END  # Direct end if confident
    }
)


flow = graph.compile()