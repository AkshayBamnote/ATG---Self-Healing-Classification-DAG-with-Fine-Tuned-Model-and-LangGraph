from typing import TypedDict, Optional

class GraphState(TypedDict):
    text: str
    prediction: Optional[str]
    confidence: Optional[float]
    clarification_needed: bool
    user_clarification: Optional[str]
    final_label: Optional[str]
    log_entry: dict
