LOCAL_SQL_TEMPLATE = """
import pandas as pd
import os
import sqlite3
import duckdb

def detect_db_type(file_path):
    if file_path.endswith('.db') or file_path.endswith('.sqlite') :
        return 'sqlite'
    elif file_path.endswith('.duckdb'):
        return 'duckdb'
    else:
        try:
            conn = duckdb.connect(database=file_path, read_only=True)
            conn.execute('SELECT 1')
            conn.close()
            return 'duckdb'
        except:
            return 'sqlite'

def execute_sql(file_path, command, output_path_path):
    db_type = detect_db_type(file_path)
    
    # Make sure the file path is correct
    if not os.path.exists(file_path) and db_type == 'sqlite':
        print(f"ERROR: Database file not found: {{file_path}}")
        return

    # Connect to the database
    if db_type == 'sqlite':
        conn = sqlite3.connect(file_path)
    elif db_type == 'duckdb':
        conn = duckdb.connect(database=file_path, read_only=False)
    else:
        print(f"ERROR: Unsupported database type {{db_type}}")
        return
    
    try:
        # Execute the SQL command and fetch the results
        df = pd.read_sql_query(command, conn)
        
        # Check if the output should be saved to a CSV file or printed directly
        if output_path_path.lower().endswith(".csv"):
            df.to_csv(output_path_path, index=False)
            print(f"Output saved to: {{output_path_path}}")
        else:
            print(df)
    except Exception as e:
        print(f"ERROR: {{e}}")
    finally:
        # Close the connection to the database
        conn.close()

# Example usage
file_path = "{file_path}"    # Path to your database file
command = f\"\"\"{sql_command}\"\"\"    # SQL command to be executed
output_path = "{output_path}"# Path to save the output as a CSV or "directly"

execute_sql(file_path, command, output_path)
"""
