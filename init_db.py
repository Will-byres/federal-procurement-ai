import duckdb

def setup_duckdb():
    print("Initializing DuckDB...")
    # Connect to a persistent file database inside the data folder
    conn = duckdb.connect('data/procurement.db')
    
    # Load the CSV directly into a DuckDB table using SQL
    print("Loading data from CSV into DuckDB...")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS contracts AS 
        SELECT * FROM read_csv_auto('data/cleaned_contracts.csv')
    ''')
    
    # Verify the data insertion
    count = conn.execute('SELECT COUNT(*) FROM contracts').fetchone()[0]
    print(f"Successfully loaded {count} contracts into DuckDB!")
    conn.close()

if __name__ == "__main__":
    setup_duckdb()