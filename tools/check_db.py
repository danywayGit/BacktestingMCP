import sqlite3
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conn = sqlite3.connect(os.path.join(_ROOT, 'data', 'crypto.db'))
cursor = conn.cursor()

# Get tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)

if tables:
    for table in tables:
        table_name = table[0]
        print(f"\n{table_name} structure:")
        cursor.execute(f"PRAGMA table_info({table_name})")
        print(cursor.fetchall())
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Row count: {count}")
        
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            print(f"Sample rows:")
            for row in cursor.fetchall():
                print(row)

conn.close()
