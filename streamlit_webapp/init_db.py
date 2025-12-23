"""Initialize missing database tables"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'chargeup_data.db')
print(f"Initializing database: {db_path}")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create bookings table if not exists
c.execute("""CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    user_id INTEGER,
    vehicle_id TEXT,
    vehicle_model TEXT,
    station_id TEXT,
    slot INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_mins INTEGER,
    status TEXT DEFAULT 'pending',
    price REAL,
    qr_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

# Create swap_requests table if not exists
c.execute("""CREATE TABLE IF NOT EXISTS swap_requests (
    id TEXT PRIMARY KEY,
    from_user_id INTEGER,
    from_vehicle TEXT,
    to_user_id INTEGER,
    to_vehicle TEXT,
    from_score REAL DEFAULT 50,
    to_score REAL DEFAULT 50,
    status TEXT DEFAULT 'pending_user',
    points_offered INTEGER DEFAULT 0,
    from_battery INTEGER DEFAULT 50,
    from_urgency INTEGER DEFAULT 5,
    user_accepted INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
)""")

conn.commit()
conn.close()

print("✅ Database initialized successfully!")
print("   - bookings table created/verified")
print("   - swap_requests table created/verified")
print("\nRestart Streamlit now for the changes to take effect.")
