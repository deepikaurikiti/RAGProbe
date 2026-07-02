import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import datetime
import config

def get_connection(db_name=None):
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=db_name or config.DB_NAME
    )

def init_db():
    # Step 1: Connect to default 'postgres' database to create the target database if it doesn't exist
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{config.DB_NAME}';")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{config.DB_NAME}' does not exist. Creating it...")
            cursor.execute(f"CREATE DATABASE {config.DB_NAME};")
            print(f"Database '{config.DB_NAME}' created successfully.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error checking/creating database '{config.DB_NAME}': {e}")
        # We'll try to proceed anyway in case they created it manually or user doesn't have privileges to postgres db

    # Step 2: Connect to the target database and create the table
    try:
        conn = get_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name VARCHAR(100),
            question TEXT,
            answer TEXT,
            contexts TEXT[],
            ground_truth TEXT,
            faithfulness REAL,
            answer_relevancy REAL,
            context_recall REAL,
            context_precision REAL
        );
        """
        cursor.execute(create_table_query)
        print("Table 'evaluation_runs' verified/created successfully.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error initializing table in '{config.DB_NAME}': {e}")

def log_evaluation(model_name, question, answer, contexts, ground_truth, metrics):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO evaluation_runs 
        (model_name, question, answer, contexts, ground_truth, faithfulness, answer_relevancy, context_recall, context_precision)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        cursor.execute(insert_query, (
            model_name,
            question,
            answer,
            contexts,
            ground_truth,
            metrics.get("faithfulness", 0.0),
            metrics.get("answer_relevancy", 0.0),
            metrics.get("context_recall", 0.0),
            metrics.get("context_precision", 0.0)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Evaluation logged successfully to PostgreSQL.")
    except Exception as e:
        print(f"Failed to log evaluation to PostgreSQL: {e}")

if __name__ == "__main__":
    init_db()
