# tools.py
import psycopg2
from typing import List
from pgvector.psycopg2 import register_vector
from langchain.tools import Tool
from langchain_core.documents import Document

# Import shared clients and credentials from the config file
from config import embeddings_model, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

def create_raw_sql_retriever(collection_name: str):
    """
    Creates a custom retriever function that executes a raw SQL query
    against a specific table in our PostgreSQL database.
    """
    def get_relevant_documents(query: str) -> List[Document]:
        """
        This inner function is the actual retriever. It takes a query,
        embeds it, and runs the raw SQL search.
        """
        query_vector = embeddings_model.embed_query(query)
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
            
            print(f"Custom retriever for '{collection_name}' found {len(documents)} documents for query: '{query[:50]}...'")
            return documents

        except Exception as e:
            print(f"Error in custom retriever for table {collection_name}: {e}")
            return []
        finally:
            if conn:
                conn.close()

    return get_relevant_documents

# Create tools using the custom retriever
print("Creating tools with custom raw SQL retrievers...")

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
print("Successfully created custom tools.")
