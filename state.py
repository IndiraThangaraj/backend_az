# state.py
from typing import TypedDict, List, Dict, Any

class WorkflowState(TypedDict):
    """Represents the state of our graph, holding data passed between nodes."""
    raw_input: str
    extracted_info: Dict[str, Any]
    demand_classification: str
    domain_classification: str
    application_list: List[str]
    application_details: str
    final_output: str
    team_lead_prompt: str