# tools.py
import psycopg2
from typing import List, Dict, Union # Import Dict and Union
from pgvector.psycopg2 import register_vector
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage # Potentially useful if message objects are passed

# Import shared clients and credentials from the config file
from config import embeddings_model, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

# Add a logger for this module
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to INFO or DEBUG for debugging


def create_raw_sql_retriever(collection_name: str):
    """
    Creates a custom retriever function that executes a raw SQL query
    against a specific table in our PostgreSQL database.
    """
    # Changed the input type hint to Union[str, Dict[str, str], BaseMessage]
    # to handle various ways LLMs/Agents might pass data.
    def get_relevant_documents(input_data: Union[str, Dict[str, str], BaseMessage]) -> List[Document]:
        """
        This inner function is the actual retriever. It takes a query,
        embeds it, and runs the raw SQL search.
        """
        query_str = ""
        if isinstance(input_data, str):
            query_str = input_data
        elif isinstance(input_data, dict):
            # Agent often passes the input as a dict with "input" or "query" key
            query_str = input_data.get("query") or input_data.get("input")
            if query_str is None:
                raise ValueError("Dictionary input to tool must contain 'query' or 'input' key.")
        elif isinstance(input_data, BaseMessage):
            # If an Agent passes a message object directly
            query_str = input_data.content
        else:
            raise TypeError(f"Unsupported input type for tool: {type(input_data)}. Expected str, dict, or BaseMessage.")

        if not isinstance(query_str, str) or not query_str: # Ensure it's a non-empty string
            logger.error(f"Invalid or empty query string extracted for embeddings: '{query_str}' (type: {type(query_str)})")
            raise ValueError("Invalid or empty query string provided to embeddings model.")

        logger.info(f"Embedding query: '{query_str[:100]}...'") # Log the actual string being embedded
        query_vector = embeddings_model.embed_query(query_str) # Use the extracted string

        conn = None
        try:
            conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
            register_vector(conn)
            cur = conn.cursor()
            
            sql_query = f"""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) AS similarity_score
                FROM {collection_name}
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """
            cur.execute(sql_query, (query_vector, query_vector))
            results = cur.fetchall()
            
            documents = []
            for row in results:
                content, metadata, score = row
                doc = Document(page_content=content, metadata=metadata or {})
                doc.metadata['score'] = score
                documents.append(doc)
            
            logger.info(f"Custom retriever for '{collection_name}' found {len(documents)} documents for query: '{query_str[:50]}...'")
            return documents

        except Exception as e:
            logger.error(f"Error in custom retriever for table {collection_name} with query '{query_str[:50]}...': {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    return get_relevant_documents

# Create tools using the custom retriever
logger.info("Creating tools with custom raw SQL retrievers...")

rules_retriever_func = create_raw_sql_retriever(collection_name="rules_kb")
app_retriever_func = create_raw_sql_retriever(collection_name="app_kb")
domain_retriever_func = create_raw_sql_retriever(collection_name="domain_kb")

tool_rules_kb = Tool(
    name="rules_kb",
    func=rules_retriever_func,
    description="Use this tool to get knowledge about demand categorization rules. The input should be a descriptive query about the rules."
)
tool_application_kb = Tool(
    name="application_kb",
    func=app_retriever_func,
    description="Use this tool to find details about company systems and applications. The input should be a descriptive query."
)
tool_domain_kb = Tool(
    name="domain_kb",
    func=domain_retriever_func,
    description="Use this tool to get knowledge about business domain classifications. The input should be a descriptive query."
)
logger.info("Successfully created custom tools.")
