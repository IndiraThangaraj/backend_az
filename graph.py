# graph.py
from langgraph.graph import StateGraph, END
from state import WorkflowState
import nodes as nodes

# This file builds and compiles the app, which can then be imported
workflow = StateGraph(WorkflowState)

workflow.add_node("extract_information", nodes.extract_information)
workflow.add_node("classify_demand", nodes.classify_demand)
workflow.add_node("classify_domain", nodes.classify_domain)
workflow.add_node("extract_and_classify_applications", nodes.extract_and_classify_applications)
workflow.add_node("format_output", nodes.format_output)



workflow.set_entry_point("extract_information")
workflow.add_edge("extract_information", "classify_demand")
workflow.add_edge("classify_demand", "classify_domain")
workflow.add_edge("classify_domain", "extract_and_classify_applications")
workflow.add_edge("extract_and_classify_applications", "format_output")
workflow.add_edge("format_output",END)


app = workflow.compile()
print("LangGraph app compiled successfully.")