# backend/main.py
from fastapi import FastAPI, Request, status # Import Request and status
from fastapi.responses import JSONResponse # Import JSONResponse
from pydantic import BaseModel
from graph import app as langgraph_app # Import your compiled LangGraph app
import logging

# Configure basic logging to capture console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the FastAPI app
api = FastAPI(
    title="Demand Analysis Agent API",
    description="An API for interacting with the LangGraph classification agent.",
    version="1.0.0"
)

# Define the request body model
class AnalysisRequest(BaseModel):
    raw_input: str

# Add the middleware (from previous suggestion, helpful for general debugging)
@api.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming Request: {request.method} {request.url}")
    # logging.info(f"Headers: {request.headers}") # Can be very verbose, enable if needed
    try:
        response = await call_next(request)
        logging.info(f"Outgoing Response Status: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"Request processing error: {e}", exc_info=True)
        raise # Re-raise the exception after logging

@api.post("/analyze")
def analyze_demand(request: AnalysisRequest):
    logging.info(f"Inside /analyze endpoint. Raw input length: {len(request.raw_input)}")
    logging.info(f"Raw input starts with: '{request.raw_input[:50]}'") # Log first 50 chars

    try:
        inputs = {"raw_input": request.raw_input}
        logging.info("Attempting to invoke LangGraph app...")
        final_state = langgraph_app.invoke(inputs)
        logging.info("LangGraph app invoked successfully.")
        return final_state
    except Exception as e:
        error_message = f"Error during LangGraph invocation: {str(e)}"
        logging.error(error_message, exc_info=True) # Log exception with traceback

        # --- THE FIX IS HERE ---
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Use status.HTTP_500_INTERNAL_SERVER_ERROR
            content={"error": error_message} # Provide the error message as JSON content
        )

@api.get("/")
def read_root():
    logging.info("GET / endpoint accessed.")
    return {"status": "Demand Analysis Agent API is running"}
