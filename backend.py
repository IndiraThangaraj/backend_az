# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from graph import app as langgraph_app # Import your compiled LangGraph app

# Initialize the FastAPI app
api = FastAPI(
    title="Demand Analysis Agent API",
    description="An API for interacting with the LangGraph classification agent.",
    version="1.0.0"
)

# Define the request body model
class AnalysisRequest(BaseModel):
    raw_input: str

@api.post("/analyze")
def analyze_demand(request: AnalysisRequest):
    """
    Receives raw text input and runs it through the LangGraph workflow.
    """
    try:
        inputs = {"raw_input": request.raw_input}
        final_state = langgraph_app.invoke(inputs)
        return final_state
    except Exception as e:
        # In production, you'd want more robust error logging
        return {"error": str(e)}, 500

@api.get("/")
def read_root():
    return {"status": "Demand Analysis Agent API is running"}