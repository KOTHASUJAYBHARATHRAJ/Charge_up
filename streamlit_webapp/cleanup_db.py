import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'chargeup_data.db')
print(f"Cleaning database: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Clear ALL stale swap requests
    c.execute("DELETE FROM swap_requests")
    deleted_swaps = c.rowcount
    
    # Also clear old bookings to start fresh
    c.execute("DELETE FROM bookings")
    deleted_bookings = c.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"✅ Cleared {deleted_swaps} swap requests and {deleted_bookings} bookings")
    print("Database cleaned! Restart Streamlit for fresh start.")
except Exception as e:
    print(f"Error: {e}")
