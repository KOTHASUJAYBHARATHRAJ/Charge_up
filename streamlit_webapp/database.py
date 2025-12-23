"""
ChargeUp EV System - Database Manager
Centralized SQLite database operations for all modules.
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

# Database path - shared across all modules
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       "raspberry_pi_backend", "chargeup_system_v3.db")

class DatabaseManager:
    """Thread-safe SQLite database manager for ChargeUp system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.db_path = DB_PATH
        self.conn_lock = threading.Lock()
        self._init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a new database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize all database tables"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # ============ AUTH TABLES ============
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            vehicle_id TEXT,
            points INTEGER DEFAULT 0,
            cooperation_score REAL DEFAULT 50.0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS merchants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            station_id TEXT,
            contact_number TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            last_login TIMESTAMP
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_username TEXT,
            action TEXT,
            ip_address TEXT,
            device_info TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # ============ STATION TABLES ============
        c.execute("""CREATE TABLE IF NOT EXISTS stations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            lat REAL,
            lon REAL,
            address TEXT,
            power_kw INTEGER DEFAULT 50,
            total_slots INTEGER DEFAULT 4,
            available_slots INTEGER DEFAULT 4,
            pricing_base REAL DEFAULT 15.0,
            operational BOOLEAN DEFAULT 1,
            ports TEXT DEFAULT '["CCS2", "Type 2"]'
        )""")
        
        # ============ CHARGING STATIONS TABLE (Legacy Compatibility) ============
        c.execute("""CREATE TABLE IF NOT EXISTS charging_stations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            lat REAL,
            lon REAL,
            power_kw REAL DEFAULT 50,
            slots INTEGER DEFAULT 4,
            status TEXT DEFAULT 'online',
            address TEXT
        )""")
        
        # ============ STATION QUEUES TABLE ============
        c.execute("""CREATE TABLE IF NOT EXISTS station_queues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT,
            vehicle_id TEXT,
            position INTEGER,
            priority_score REAL,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'waiting',
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )""")
        
        # ============ BOOKING TABLES ============
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )""")
        
        # ============ SWAP TABLES ============
        c.execute("""CREATE TABLE IF NOT EXISTS swap_requests (
            id TEXT PRIMARY KEY,
            from_user_id INTEGER,
            from_vehicle TEXT,
            to_user_id INTEGER,
            to_vehicle TEXT,
            from_score REAL,
            to_score REAL,
            status TEXT DEFAULT 'pending',
            points_offered INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (from_user_id) REFERENCES users(id),
            FOREIGN KEY (to_user_id) REFERENCES users(id)
        )""")
        
        # ============ SIMULATION TABLES ============
        c.execute("""CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT,
            run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_users INTEGER,
            total_bookings INTEGER,
            total_swaps INTEGER,
            swap_success_rate REAL,
            avg_wait_time REAL,
            qlearning_iterations INTEGER,
            fuzzy_calculations INTEGER,
            status TEXT DEFAULT 'running'
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS fuzzy_calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_run_id INTEGER,
            vehicle_id TEXT,
            battery_level INTEGER,
            urgency INTEGER,
            distance_km REAL,
            wait_time_mins INTEGER DEFAULT 0,
            priority_score REAL,
            membership_values TEXT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (simulation_run_id) REFERENCES simulation_runs(id)
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS qlearning_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_run_id INTEGER,
            state_repr TEXT,
            action INTEGER,
            reward REAL,
            q_value REAL,
            iteration INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (simulation_run_id) REFERENCES simulation_runs(id)
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS station_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT,
            user_id INTEGER,
            booking_id TEXT,
            rating INTEGER,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES stations(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        
        # ============ VEHICLES TABLE ============
        c.execute("""CREATE TABLE IF NOT EXISTS vehicles (
            id TEXT PRIMARY KEY,
            owner_id INTEGER,
            model TEXT NOT NULL,
            battery_capacity_kwh REAL DEFAULT 40.0,
            current_battery_percent REAL DEFAULT 75.0,
            range_km REAL DEFAULT 300.0,
            port_type TEXT DEFAULT 'CCS2',
            lat REAL,
            lon REAL,
            status TEXT DEFAULT 'idle',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        )""")
        
        # ============ COOPERATION HISTORY TABLE ============
        c.execute("""CREATE TABLE IF NOT EXISTS cooperation_history (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT,
            user_id INTEGER,
            event_type TEXT NOT NULL,
            reward_points INTEGER DEFAULT 0,
            description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        
        # ============ RESERVATIONS TABLE ============
        c.execute("""CREATE TABLE IF NOT EXISTS reservations (
            reservation_id TEXT PRIMARY KEY,
            vehicle_id TEXT,
            station_id TEXT,
            slot INTEGER,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status TEXT DEFAULT 'pending',
            price REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(id),
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )""")
        
        # ============ STATION FEEDBACK TABLE ============
        c.execute("""CREATE TABLE IF NOT EXISTS station_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT,
            vehicle_id TEXT,
            user_id INTEGER,
            rating INTEGER DEFAULT 5,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT 0,
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )""")
        
        # ============ ANTI-MALPRACTICE TABLES ============
        c.execute("""CREATE TABLE IF NOT EXISTS malpractice_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            offense_type TEXT NOT NULL,
            description TEXT,
            penalty_points INTEGER DEFAULT 10,
            penalty_applied BOOLEAN DEFAULT 0,
            reported_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS user_bans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            ban_reason TEXT,
            ban_level TEXT DEFAULT 'temporary',
            ban_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ban_end TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS swap_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            swap_id TEXT,
            verification_code TEXT,
            verified_by_from BOOLEAN DEFAULT 0,
            verified_by_to BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP,
            FOREIGN KEY (swap_id) REFERENCES swap_requests(id)
        )""")
        
        conn.commit()
        self._seed_initial_data(conn)
        conn.close()
    
    def _seed_initial_data(self, conn):
        """Seed initial data if tables are empty"""
        c = conn.cursor()
        
        # Seed Admin
        c.execute("SELECT COUNT(*) FROM admins")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", 
                     ("admin", "admin123"))
        
        # Seed Users (20 users for comprehensive testing)
        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            users = [
                ("user1", "pass1", "user1@chargeup.com", "KL-01-AB-1234", 150, 75.0),
                ("user2", "pass2", "user2@chargeup.com", "KL-07-CD-5678", 120, 60.0),
                ("user3", "pass3", "user3@chargeup.com", "KL-03-EF-9012", 200, 80.0),
                ("user4", "pass4", "user4@chargeup.com", "KL-04-GH-3456", 80, 55.0),
                ("user5", "pass5", "user5@chargeup.com", "KL-05-IJ-7890", 175, 70.0),
                ("user6", "pass6", "user6@chargeup.com", "KL-06-KL-2345", 90, 65.0),
                ("user7", "pass7", "user7@chargeup.com", "KL-07-MN-6789", 300, 88.0),
                ("user8", "pass8", "user8@chargeup.com", "KL-08-OP-1234", 50, 45.0),
                ("user9", "pass9", "user9@chargeup.com", "KL-09-QR-5678", 220, 82.0),
                ("user10", "pass10", "user10@chargeup.com", "KL-10-ST-9012", 130, 68.0),
                ("user11", "pass11", "user11@chargeup.com", "KL-11-UV-3456", 180, 72.0),
                ("user12", "pass12", "user12@chargeup.com", "KL-12-WX-7890", 95, 58.0),
                ("user13", "pass13", "user13@chargeup.com", "KL-13-YZ-2345", 250, 85.0),
                ("user14", "pass14", "user14@chargeup.com", "KL-14-AB-6789", 70, 50.0),
                ("user15", "pass15", "user15@chargeup.com", "KL-15-CD-1234", 190, 78.0),
                ("rajan", "rajan123", "rajan@gmail.com", "KL-01-RJ-1001", 350, 90.0),
                ("arun", "arun123", "arun@gmail.com", "KL-02-AR-2002", 280, 83.0),
                ("priya", "priya123", "priya@gmail.com", "KL-03-PR-3003", 400, 92.0),
                ("chargeupuser", "chargeupuser", "demo@chargeup.com", "KL-01-AB-1234", 500, 85.0),
                ("demo", "demo", "demo2@chargeup.com", "KL-99-DM-9999", 1000, 95.0),
            ]
            c.executemany("""INSERT INTO users 
                (username, password, email, vehicle_id, points, cooperation_score) 
                VALUES (?, ?, ?, ?, ?, ?)""", users)
        
        # Seed Merchants (merchant1, merchant2, merchant3 for demo + others)
        c.execute("SELECT COUNT(*) FROM merchants")
        if c.fetchone()[0] == 0:
            merchants = [
                ("merchant1", "pass1", "STN01", "9876543210"),  # Demo Merchant 1
                ("merchant2", "pass2", "STN02", "9876543211"),  # Demo Merchant 2
                ("merchant3", "pass3", "STN03", "9876543212"),  # Demo Merchant 3
                ("merchant4", "pass4", "STN04", "9876543213"),
                ("merchant5", "pass5", "STN05", "9876543214"),
                ("merchant6", "pass6", "STN06", "9876543215"),
                ("merchant7", "pass7", "STN07", "9876543216"),
            ]
            c.executemany("""INSERT INTO merchants 
                (username, password, station_id, contact_number) 
                VALUES (?, ?, ?, ?)""", merchants)
        
        # Seed Stations (7 major stations across Kerala)
        c.execute("SELECT COUNT(*) FROM stations")
        if c.fetchone()[0] == 0:
            stations = [
                ("STN01", "Kochi Central Hub", 9.9312, 76.2673, "MG Road, Ernakulam", 50, 4, 4, 15.0, '["CCS2", "Type 2"]'),
                ("STN02", "Trivandrum Tech Park", 8.5241, 76.9366, "Technopark, Trivandrum", 60, 4, 4, 18.0, '["CCS2", "CHAdeMO"]'),
                ("STN03", "Calicut Highway", 11.2588, 75.7804, "NH66, Kozhikode", 25, 3, 3, 12.0, '["CCS2"]'),
                ("STN04", "Thrissur Mall", 10.5276, 76.2144, "Shakthan Nagar, Thrissur", 30, 4, 4, 14.0, '["CCS2", "Type 2"]'),
                ("STN05", "Kottayam Junction", 9.5916, 76.5222, "Baker Junction, Kottayam", 22, 2, 2, 10.0, '["Type 2"]'),
                ("STN06", "Alappuzha Beach Road", 9.4981, 76.3388, "Beach Road, Alappuzha", 35, 2, 2, 11.0, '["CCS2"]'),
                ("STN07", "Kannur Smart City", 11.8745, 75.3704, "Kannur IT Park", 50, 4, 4, 16.0, '["CCS2", "CHAdeMO"]'),
            ]
            c.executemany("""INSERT INTO stations 
                (id, name, lat, lon, address, power_kw, total_slots, available_slots, pricing_base, ports) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", stations)
        
        # Seed Vehicles (15 vehicles with different models)
        c.execute("SELECT COUNT(*) FROM vehicles")
        if c.fetchone()[0] == 0:
            vehicles = [
                ("KL-01-AB-1234", 1, "Tata Nexon EV Max", 40.5, 72, 350, "CCS2", 9.935, 76.270, "idle"),
                ("KL-07-CD-5678", 2, "MG ZS EV", 50.3, 13, 380, "CCS2", 8.520, 76.940, "idle"),
                ("KL-03-EF-9012", 3, "Hyundai Kona Electric", 39.2, 68, 340, "CCS2", 11.260, 75.780, "idle"),
                ("KL-04-GH-3456", 4, "Tata Tiago EV", 24.0, 55, 280, "CCS2", 10.530, 76.210, "idle"),
                ("KL-05-IJ-7890", 5, "Mahindra XUV400", 39.4, 82, 360, "CCS2", 9.590, 76.520, "idle"),
                ("KL-06-KL-2345", 6, "BYD Atto 3", 60.5, 78, 420, "CCS2", 10.790, 76.660, "idle"),
                ("KL-07-MN-6789", 7, "Hyundai Ioniq 5", 72.6, 65, 480, "CCS2", 9.500, 76.340, "charging"),
                ("KL-08-OP-1234", 8, "Kia EV6", 77.4, 58, 500, "CCS2", 11.880, 75.370, "idle"),
                ("KL-09-QR-5678", 9, "Tata Nexon EV Prime", 30.2, 42, 250, "CCS2", 8.890, 76.610, "idle"),
                ("KL-10-ST-9012", 10, "MG Comet EV", 17.3, 88, 180, "Type 2", 8.740, 76.720, "idle"),
                ("KL-01-RJ-1001", 16, "BMW iX", 105.2, 90, 550, "CCS2", 9.920, 76.260, "idle"),
                ("KL-02-AR-2002", 17, "Mercedes EQS", 107.8, 75, 600, "CCS2", 10.090, 77.060, "idle"),
                ("KL-03-PR-3003", 18, "Audi e-tron", 95.0, 60, 500, "CCS2", 11.690, 76.130, "idle"),
                ("KL-99-DM-9999", 20, "Tesla Model 3", 82.0, 95, 520, "CCS2", 9.931, 76.267, "idle"),
                ("KL-00-TEST-0000", None, "Test Vehicle", 50.0, 50, 300, "CCS2", 9.900, 76.200, "idle"),
            ]
            c.executemany("""INSERT INTO vehicles 
                (id, owner_id, model, battery_capacity_kwh, current_battery_percent, range_km, port_type, lat, lon, status) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", vehicles)
        
        conn.commit()
    
    # ============ QUERY METHODS ============
    
    def execute(self, query: str, params: tuple = (), fetch: bool = False) -> Any:
        """Execute a query with thread-safe locking"""
        with self.conn_lock:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute(query, params)
            if fetch:
                result = [dict(row) for row in c.fetchall()]
                conn.close()
                return result
            conn.commit()
            last_id = c.lastrowid
            conn.close()
            return last_id
    
    def get_all_stations(self) -> List[Dict]:
        """Get all stations"""
        return self.execute("SELECT * FROM stations WHERE operational = 1", fetch=True)
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        return self.execute("SELECT * FROM users WHERE is_active = 1", fetch=True)
    
    def get_user_by_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        result = self.execute(
            "SELECT * FROM users WHERE (username=? OR email=?) AND password=?",
            (username, username, password), fetch=True
        )
        return result[0] if result else None
    
    def get_merchant_by_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate merchant"""
        result = self.execute(
            "SELECT * FROM merchants WHERE username=? AND password=?",
            (username, password), fetch=True
        )
        return result[0] if result else None
    
    def get_admin_by_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate admin"""
        result = self.execute(
            "SELECT * FROM admins WHERE username=? AND password=?",
            (username, password), fetch=True
        )
        return result[0] if result else None
    
    def log_admin_action(self, username: str, action: str, ip: str = "127.0.0.1"):
        """Log admin action"""
        self.execute(
            "INSERT INTO admin_logs (admin_username, action, ip_address) VALUES (?, ?, ?)",
            (username, action, ip)
        )
    
    def get_admin_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent admin logs"""
        return self.execute(
            "SELECT * FROM admin_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,), fetch=True
        )
    
    def create_booking(self, booking_data: Dict) -> str:
        """Create a new booking"""
        self.execute("""
            INSERT INTO bookings (id, user_id, vehicle_id, vehicle_model, station_id, 
                                 slot, start_time, end_time, duration_mins, status, price, qr_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            booking_data['id'], booking_data.get('user_id'), booking_data['vehicle_id'],
            booking_data.get('vehicle_model', ''), booking_data['station_id'],
            booking_data['slot'], booking_data['start_time'], booking_data['end_time'],
            booking_data['duration_mins'], booking_data.get('status', 'confirmed'),
            booking_data.get('price', 0), booking_data.get('qr_code', '')
        ))
        return booking_data['id']
    
    def get_bookings(self, status: str = None, limit: int = 100) -> List[Dict]:
        """Get bookings with optional status filter"""
        if status:
            return self.execute(
                "SELECT * FROM bookings WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit), fetch=True
            )
        return self.execute(
            "SELECT * FROM bookings ORDER BY created_at DESC LIMIT ?",
            (limit,), fetch=True
        )
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile by ID"""
        result = self.execute(
            "SELECT * FROM users WHERE id=?",
            (user_id,), fetch=True
        )
        return result[0] if result else None
    
    def get_user_vehicles(self, user_id: int) -> List[Dict]:
        """Get all vehicles owned by a specific user - CRITICAL FOR ISOLATION"""
        return self.execute(
            "SELECT * FROM vehicles WHERE owner_id=?",
            (user_id,), fetch=True
        )
    def create_simulation_run(self, run_name: str, total_users: int) -> int:
        """Create a new simulation run and return its ID"""
        return self.execute("""
            INSERT INTO simulation_runs (run_name, total_users, status) VALUES (?, ?, 'running')
        """, (run_name, total_users))
    
    def update_simulation_run(self, run_id: int, data: Dict):
        """Update simulation run with results"""
        self.execute("""
            UPDATE simulation_runs SET 
                total_bookings=?, total_swaps=?, swap_success_rate=?, 
                avg_wait_time=?, qlearning_iterations=?, fuzzy_calculations=?, status='completed'
            WHERE id=?
        """, (
            data.get('total_bookings', 0), data.get('total_swaps', 0),
            data.get('swap_success_rate', 0), data.get('avg_wait_time', 0),
            data.get('qlearning_iterations', 0), data.get('fuzzy_calculations', 0),
            run_id
        ))
    
    def log_fuzzy_calculation(self, run_id: int, calc_data: Dict):
        """Log a fuzzy logic calculation"""
        self.execute("""
            INSERT INTO fuzzy_calculations 
                (simulation_run_id, vehicle_id, battery_level, urgency, distance_km, 
                 wait_time_mins, priority_score, membership_values)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, calc_data['vehicle_id'], calc_data['battery_level'],
            calc_data['urgency'], calc_data['distance_km'],
            calc_data.get('wait_time_mins', 0), calc_data['priority_score'],
            json.dumps(calc_data.get('membership_values', {}))
        ))
    
    def log_qlearning_state(self, run_id: int, state_data: Dict):
        """Log Q-Learning state update"""
        self.execute("""
            INSERT INTO qlearning_states 
                (simulation_run_id, state_repr, action, reward, q_value, iteration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id, state_data['state'], state_data['action'],
            state_data['reward'], state_data['q_value'], state_data['iteration']
        ))
    
    def get_simulation_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent simulation runs"""
        return self.execute(
            "SELECT * FROM simulation_runs ORDER BY run_timestamp DESC LIMIT ?",
            (limit,), fetch=True
        )
    
    def get_fuzzy_calculations(self, run_id: int) -> List[Dict]:
        """Get fuzzy calculations for a simulation run"""
        return self.execute(
            "SELECT * FROM fuzzy_calculations WHERE simulation_run_id=? ORDER BY calculated_at",
            (run_id,), fetch=True
        )
    
    def get_qlearning_states(self, run_id: int) -> List[Dict]:
        """Get Q-Learning states for a simulation run"""
        return self.execute(
            "SELECT * FROM qlearning_states WHERE simulation_run_id=? ORDER BY iteration",
            (run_id,), fetch=True
        )
    
    def get_station_stats(self) -> Dict:
        """Get station statistics"""
        total = self.execute("SELECT COUNT(*) as cnt FROM stations", fetch=True)[0]['cnt']
        operational = self.execute("SELECT COUNT(*) as cnt FROM stations WHERE operational=1", fetch=True)[0]['cnt']
        return {'total': total, 'operational': operational}
    
    def get_user_stats(self) -> Dict:
        """Get user statistics"""
        total = self.execute("SELECT COUNT(*) as cnt FROM users", fetch=True)[0]['cnt']
        active = self.execute("SELECT COUNT(*) as cnt FROM users WHERE is_active=1", fetch=True)[0]['cnt']
        return {'total': total, 'active': active}
    
    def get_booking_stats(self) -> Dict:
        """Get booking statistics"""
        total = self.execute("SELECT COUNT(*) as cnt FROM bookings", fetch=True)[0]['cnt']
        confirmed = self.execute("SELECT COUNT(*) as cnt FROM bookings WHERE status='confirmed'", fetch=True)[0]['cnt']
        return {'total': total, 'confirmed': confirmed}
    
    # ============ VEHICLE METHODS ============
    
    def get_all_vehicles(self) -> List[Dict]:
        """Get all active vehicles"""
        return self.execute("SELECT * FROM vehicles WHERE is_active=1", fetch=True)
    
    def get_user_vehicles(self, user_id: int) -> List[Dict]:
        """Get ALL vehicles owned by a user - supports multiple cars per user"""
        return self.execute(
            "SELECT * FROM vehicles WHERE owner_id=? AND is_active=1",
            (user_id,), fetch=True
        )
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile with points, cooperation score for personalized fuzzy logic"""
        result = self.execute(
            "SELECT id, username, email, points, cooperation_score FROM users WHERE id=?",
            (user_id,), fetch=True
        )
        return result[0] if result else None
    
    def update_user_points(self, user_id: int, points_delta: int):
        """Add or subtract points from user"""
        self.execute(
            "UPDATE users SET points = points + ? WHERE id=?",
            (points_delta, user_id)
        )
    
    def update_cooperation_score(self, user_id: int, score: float):
        """Update user's cooperation score (for fuzzy logic personalization)"""
        self.execute(
            "UPDATE users SET cooperation_score = ? WHERE id=?",
            (score, user_id)
        )
    
    def update_vehicle_battery(self, vehicle_id: str, battery_percent: float):
        """Update vehicle battery level"""
        self.execute(
            "UPDATE vehicles SET current_battery_percent=? WHERE id=?",
            (battery_percent, vehicle_id)
        )
    
    def update_vehicle_location(self, vehicle_id: str, lat: float, lon: float):
        """Update vehicle location"""
        self.execute(
            "UPDATE vehicles SET lat=?, lon=? WHERE id=?",
            (lat, lon, vehicle_id)
        )
    
    def register_vehicle(self, owner_id: int, vehicle_id: str, model: str, 
                        battery_capacity: float = 40.0, port_type: str = 'CCS2',
                        lat: float = 10.85, lon: float = 76.27) -> bool:
        """Register a new vehicle for a user - syncs to database"""
        try:
            self.execute(
                """INSERT INTO vehicles (id, owner_id, model, battery_capacity_kwh, 
                   current_battery_percent, range_km, port_type, lat, lon, status, is_active)
                   VALUES (?, ?, ?, ?, 75.0, 300.0, ?, ?, ?, 'idle', 1)""",
                (vehicle_id, owner_id, model, battery_capacity, port_type, lat, lon)
            )
            return True
        except Exception as e:
            print(f"Vehicle registration failed: {e}")
            return False
    
    def create_reservation_with_payment(self, vehicle_id: str, station_id: str, 
                                        slot: int, start_time, end_time, 
                                        advance_payment: float = 50.0) -> dict:
        """Create reservation with advance payment - returns QR data"""
        import uuid
        reservation_id = f"RES_{uuid.uuid4().hex[:8].upper()}"
        
        self.execute(
            """INSERT INTO reservations (reservation_id, vehicle_id, station_id, slot, 
               start_time, end_time, status, price, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending_payment', ?, CURRENT_TIMESTAMP)""",
            (reservation_id, vehicle_id, station_id, slot, start_time, end_time, advance_payment)
        )
        
        # Generate QR payment data
        qr_data = {
            'reservation_id': reservation_id,
            'station_id': station_id,
            'slot': slot,
            'amount': advance_payment,
            'vehicle_id': vehicle_id,
            'valid_until': end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time),
            'payment_status': 'pending'
        }
        
        return qr_data
    
    def confirm_reservation_payment(self, reservation_id: str) -> bool:
        """Confirm payment and activate reservation"""
        self.execute(
            "UPDATE reservations SET status='confirmed' WHERE reservation_id=?",
            (reservation_id,)
        )
        return True
    
    def log_station_issue(self, station_id: str, issue_type: str, 
                         description: str, reported_by: int = None):
        """Log real-time station issues (port problems, etc.)"""
        self.execute(
            """INSERT INTO station_feedback (station_id, user_id, rating, message, timestamp, resolved)
               VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP, 0)""",
            (station_id, reported_by, f"[{issue_type}] {description}")
        )
    
    def get_vehicle_stats(self) -> Dict:
        """Get vehicle statistics"""
        total = self.execute("SELECT COUNT(*) as cnt FROM vehicles", fetch=True)[0]['cnt']
        active = self.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE is_active=1", fetch=True)[0]['cnt']
        charging = self.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE status='charging'", fetch=True)[0]['cnt']
        return {'total': total, 'active': active, 'charging': charging}
    
    # ============ ANTI-MALPRACTICE METHODS ============
    
    def record_malpractice(self, user_id: int, offense_type: str, 
                           description: str, penalty_points: int = 10,
                           reported_by: str = "system") -> int:
        """Record a malpractice offense"""
        return self.execute("""
            INSERT INTO malpractice_records 
                (user_id, offense_type, description, penalty_points, reported_by)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, offense_type, description, penalty_points, reported_by))
    
    def get_user_offenses(self, user_id: int) -> List[Dict]:
        """Get all offenses for a user"""
        return self.execute(
            "SELECT * FROM malpractice_records WHERE user_id=? ORDER BY created_at DESC",
            (user_id,), fetch=True
        )
    
    def apply_penalty(self, user_id: int, penalty_points: int):
        """Apply penalty by reducing cooperation score"""
        self.execute("""
            UPDATE users SET cooperation_score = MAX(0, cooperation_score - ?)
            WHERE id=?
        """, (penalty_points, user_id))
    
    def ban_user(self, user_id: int, reason: str, 
                 ban_level: str = "temporary", duration_days: int = 7):
        """Ban a user temporarily or permanently"""
        ban_end = None
        if ban_level == "temporary":
            from datetime import datetime, timedelta
            ban_end = (datetime.now() + timedelta(days=duration_days)).isoformat()
        
        # Check if already banned
        existing = self.execute(
            "SELECT id FROM user_bans WHERE user_id=? AND is_active=1",
            (user_id,), fetch=True
        )
        
        if existing:
            self.execute(
                "UPDATE user_bans SET ban_reason=?, ban_level=?, ban_end=? WHERE user_id=? AND is_active=1",
                (reason, ban_level, ban_end, user_id)
            )
        else:
            self.execute("""
                INSERT INTO user_bans (user_id, ban_reason, ban_level, ban_end)
                VALUES (?, ?, ?, ?)
            """, (user_id, reason, ban_level, ban_end))
        
        # Deactivate user
        self.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
    
    def unban_user(self, user_id: int):
        """Remove user ban"""
        self.execute("UPDATE user_bans SET is_active=0 WHERE user_id=?", (user_id,))
        self.execute("UPDATE users SET is_active=1 WHERE id=?", (user_id,))
    
    def is_user_banned(self, user_id: int) -> bool:
        """Check if user is currently banned"""
        result = self.execute(
            "SELECT id FROM user_bans WHERE user_id=? AND is_active=1",
            (user_id,), fetch=True
        )
        return len(result) > 0
    
    def get_banned_users(self) -> List[Dict]:
        """Get all currently banned users"""
        return self.execute(
            "SELECT * FROM user_bans WHERE is_active=1 ORDER BY ban_start DESC",
            fetch=True
        )
    
    # ============ SWAP REQUEST METHODS ============
    
    def create_swap_request(self, req_data: Dict) -> bool:
        """Create a new swap request - simplified to avoid schema issues"""
        try:
            # Direct insert without subqueries that might fail
            self.execute("""
                INSERT INTO swap_requests (
                    id, from_user_id, from_vehicle, to_user_id, to_vehicle,
                    from_score, to_score, status, points_offered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                req_data['id'], 
                req_data.get('from_user_id', 1),  # Default to user 1 if not provided
                req_data.get('from_vehicle', req_data.get('from_user', '')),
                req_data.get('to_user_id', 2),    # Default to user 2
                req_data.get('to_vehicle', req_data.get('to_user', '')),
                req_data.get('from_score', 50),
                req_data.get('to_score', 50),
                req_data.get('status', 'pending_user'),
                req_data.get('points_offered', 0)
            ))
            return True
        except Exception as e:
            print(f"Swap creation failed: {e}")
            return False

    def get_swap_requests(self, to_vehicle_id: str = None, status: str = 'pending_user') -> List[Dict]:
        """Get swap requests for a specific target vehicle/user"""
        if to_vehicle_id:
            # Join with users to get usernames if needed, but schema stores basic data
            # Note: The table stores `to_vehicle` as text based on my create statement above
            return self.execute(
                "SELECT * FROM swap_requests WHERE to_vehicle=? AND status=? ORDER BY created_at DESC",
                (to_vehicle_id, status), fetch=True
            )
        return self.execute("SELECT * FROM swap_requests WHERE status=?", (status,), fetch=True)

    # ============ ROBUST SWAP VERIFICATION ============
    
    def create_swap_verification(self, swap_id: str) -> str:
        """Create verification record for swap with unique code"""
        import random
        import string
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
        self.execute("""
            INSERT INTO swap_verifications (swap_id, verification_code)
            VALUES (?, ?)
        """, (swap_id, code))
        
        return code
    
    def verify_swap_participant(self, swap_id: str, is_from_user: bool):
        """Mark swap as verified by one participant"""
        if is_from_user:
            self.execute(
                "UPDATE swap_verifications SET verified_by_from=1 WHERE swap_id=?",
                (swap_id,)
            )
        else:
            self.execute(
                "UPDATE swap_verifications SET verified_by_to=1 WHERE swap_id=?",
                (swap_id,)
            )
    
    def is_swap_fully_verified(self, swap_id: str) -> bool:
        """Check if both parties have verified the swap"""
        result = self.execute(
            "SELECT verified_by_from, verified_by_to FROM swap_verifications WHERE swap_id=?",
            (swap_id,), fetch=True
        )
        if result:
            return result[0]['verified_by_from'] and result[0]['verified_by_to']
        return False
    
    def complete_verified_swap(self, swap_id: str):
        """Complete a swap after both parties verify"""
        self.execute(
            "UPDATE swap_verifications SET verified_at=datetime('now') WHERE swap_id=?",
            (swap_id,)
        )
        self.execute(
            "UPDATE swap_requests SET status='completed', resolved_at=datetime('now') WHERE id=?",
            (swap_id,)
        )
    
    def get_malpractice_stats(self) -> Dict:
        """Get malpractice statistics"""
        total = self.execute("SELECT COUNT(*) as cnt FROM malpractice_records", fetch=True)[0]['cnt']
        pending = self.execute("SELECT COUNT(*) as cnt FROM malpractice_records WHERE penalty_applied=0", fetch=True)[0]['cnt']
        banned = self.execute("SELECT COUNT(*) as cnt FROM user_bans WHERE is_active=1", fetch=True)[0]['cnt']
        return {'total_offenses': total, 'pending_review': pending, 'active_bans': banned}


# Singleton instance
db = DatabaseManager()

