#!/usr/bin/env python3
"""
ChargeUp EV Management System v2.0 - Raspberry Pi Backend
Enhanced with: QR validation, AI battery prediction, fuzzy logic, hotel integration,
personal destinations, smart rerouting, and time-based billing
"""

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import math
import logging
import os
import random
import requests
from typing import Dict, List, Any, Optional, Tuple
import sys
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import socket
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.FileHandler('chargeup_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
@dataclass
class SystemConfig:
    SIMULATION_MODE: bool = True
    NUM_SIMULATED_CARS: int = 4
    NUM_SIMULATED_STATIONS: int = 3

    # MQTT Configuration
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883

    # Battery & Charging
    CRITICAL_BATTERY_THRESHOLD: int = 20
    LOW_BATTERY_THRESHOLD: int = 30
    MIN_CHARGING_TIME_MIN: int = 10
    MAX_CHARGING_TIME_MIN: int = 90

    # Billing Rates (INR)
    ELECTRICITY_RATE_PER_KWH: float = 12.0
    PARKING_RATE_PER_HOUR: float = 20.0
    OVERTIME_MULTIPLIER: float = 1.5

    # Fuzzy Logic Parameters
    COOPERATION_REWARD_BASE: float = 10.0
    URGENCY_WEIGHT: float = 0.4
    BATTERY_WEIGHT: float = 0.35
    DISTANCE_WEIGHT: float = 0.25

    # Rerouting
    REROUTING_SEARCH_RADIUS_KM: float = 50.0
    MAX_DETOUR_KM: float = 20.0

    # QR Code
    QR_TIMEOUT_SECONDS: int = 300  # 5 minutes

    # AI Battery Health
    BATTERY_HEALTH_WARNING_THRESHOLD: float = 80.0
    BATTERY_HEALTH_CRITICAL_THRESHOLD: float = 70.0

config = SystemConfig()

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================
class DatabaseManager:
    # Use relative path if the script is run from the directory where the DB should reside
    def __init__(self, db_path='chargeup_system.db'): 
        # Resolve path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(script_dir, db_path)
        
        self.lock = threading.Lock()
        self._initialize_database()

    def _initialize_database(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Vehicles table - enhanced with health tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    vehicle_id TEXT PRIMARY KEY,
                    battery_level INTEGER,
                    battery_health REAL DEFAULT 100.0,
                    cycle_count INTEGER DEFAULT 0,
                    current_lat REAL,
                    current_lon REAL,
                    status TEXT,
                    charging BOOLEAN,
                    range_km REAL,
                    cooperation_score REAL DEFAULT 0,
                    user_preferences TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Charging stations - enhanced with billing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS charging_stations (
                    station_id TEXT PRIMARY KEY,
                    name TEXT,
                    lat REAL,
                    lon REAL,
                    total_slots INTEGER,
                    available_slots INTEGER,
                    charging_rate_kwh REAL,
                    parking_rate_hour REAL,
                    operational BOOLEAN DEFAULT 1,
                    amenities TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # QR Codes tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qr_codes (
                    qr_id TEXT PRIMARY KEY,
                    vehicle_id TEXT,
                    station_id TEXT,
                    qr_data TEXT,
                    generated_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    scanned BOOLEAN DEFAULT 0,
                    scanned_at TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
                    FOREIGN KEY (station_id) REFERENCES charging_stations(station_id)
                )
            """)

            # Charging sessions - comprehensive billing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS charging_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    station_id TEXT,
                    qr_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    allocated_minutes INTEGER,
                    actual_minutes INTEGER,
                    overtime_minutes INTEGER DEFAULT 0,
                    start_battery INTEGER,
                    end_battery INTEGER,
                    energy_delivered_kwh REAL,
                    electricity_cost REAL,
                    parking_cost REAL,
                    overtime_cost REAL DEFAULT 0,
                    total_cost REAL,
                    cooperation_bonus REAL DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
                    FOREIGN KEY (station_id) REFERENCES charging_stations(station_id)
                )
            """)

            # Reservations - enhanced with hotel integration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    station_id TEXT,
                    reservation_type TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    estimated_charge_time INTEGER,
                    destination_type TEXT,
                    destination_name TEXT,
                    destination_lat REAL,
                    destination_lon REAL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
                    FOREIGN KEY (station_id) REFERENCES charging_stations(station_id)
                )
            """)

            # User destinations (home, friends, work, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_destinations (
                    dest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    dest_type TEXT,
                    dest_name TEXT,
                    lat REAL,
                    lon REAL,
                    has_charging BOOLEAN DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
                )
            """)

            # Battery health history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS battery_health_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    battery_health REAL,
                    cycle_count INTEGER,
                    voltage REAL,
                    temperature REAL,
                    ai_prediction TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
                )
            """)

            # Cooperation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cooperation_history (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    event_type TEXT,
                    urgency_score REAL,
                    battery_level INTEGER,
                    yielded_to TEXT,
                    reward_points REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
                )
            """)

            # Hotel partnerships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hotels (
                    hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    lat REAL,
                    lon REAL,
                    charging_stations INTEGER,
                    parking_slots INTEGER,
                    combined_rate_hour REAL,
                    amenities TEXT,
                    rating REAL,
                    phone TEXT
                )
            """)
            
            # Queue tracking (essential for swap logic)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS station_queues (
                    station_id TEXT PRIMARY KEY,
                    queue_json TEXT,
                    queue_length INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES charging_stations(station_id)
                )
            """)

            conn.commit()
            conn.close()
            logger.info("✅ Enhanced database initialized")

    def execute_query(self, query: str, params: tuple = (), fetch: bool = False):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)

            if fetch:
                result = cursor.fetchall()
                conn.close()
                return [dict(row) for row in result]
            else:
                conn.commit()
                conn.close()
                return cursor.lastrowid

# ============================================================================
# ENHANCED FUZZY LOGIC SYSTEM
# ============================================================================
class MamdaniFuzzyLogic:
    """
    Proper Mamdani-type Fuzzy Inference System
    Replaces simple weighted approach with full fuzzy pipeline
    """

    def __init__(self):
        self.battery_range = np.arange(0, 101, 1)
        self.distance_range = np.arange(0, 51, 1)
        self.urgency_range = np.arange(0, 11, 1)
        self.priority_range = np.arange(0, 101, 1)

    def battery_membership(self, battery_level: float) -> Dict[str, float]:
        """Triangular/Trapezoidal membership for battery"""
        memberships = {}

        # Critical: Trapezoidal [0, 0, 10, 20]
        if battery_level <= 10:
            memberships['critical'] = 1.0
        elif battery_level <= 20:
            memberships['critical'] = (20 - battery_level) / 10
        else:
            memberships['critical'] = 0.0

        # Low: Triangular [10, 30, 50]
        if battery_level <= 10:
            memberships['low'] = 0.0
        elif battery_level <= 30:
            memberships['low'] = (battery_level - 10) / 20
        elif battery_level <= 50:
            memberships['low'] = (50 - battery_level) / 20
        else:
            memberships['low'] = 0.0

        # Medium: Triangular [40, 60, 80]
        if battery_level <= 40:
            memberships['medium'] = 0.0
        elif battery_level <= 60:
            memberships['medium'] = (battery_level - 40) / 20
        elif battery_level <= 80:
            memberships['medium'] = (80 - battery_level) / 20
        else:
            memberships['medium'] = 0.0

        # High: Trapezoidal [70, 90, 100, 100]
        if battery_level <= 70:
            memberships['high'] = 0.0
        elif battery_level <= 90:
            memberships['high'] = (battery_level - 70) / 20
        else:
            memberships['high'] = 1.0

        return memberships

    def distance_membership(self, distance_km: float) -> Dict[str, float]:
        """Gaussian membership for distance"""
        memberships = {}
        memberships['very_near'] = np.exp(-0.5 * ((distance_km - 0) / 3) ** 2)
        memberships['near'] = np.exp(-0.5 * ((distance_km - 10) / 5) ** 2)
        memberships['far'] = np.exp(-0.5 * ((distance_km - 25) / 7) ** 2)
        memberships['very_far'] = np.exp(-0.5 * ((distance_km - 45) / 8) ** 2)
        return memberships

    def urgency_membership(self, urgency: float) -> Dict[str, float]:
        """Triangular membership for urgency"""
        memberships = {}

        # Low: [0, 0, 5]
        if urgency <= 0:
            memberships['low'] = 1.0
        elif urgency <= 5:
            memberships['low'] = (5 - urgency) / 5
        else:
            memberships['low'] = 0.0

        # Medium: [3, 5, 7]
        if urgency <= 3:
            memberships['medium'] = 0.0
        elif urgency <= 5:
            memberships['medium'] = (urgency - 3) / 2
        elif urgency <= 7:
            memberships['medium'] = (7 - urgency) / 2
        else:
            memberships['medium'] = 0.0

        # High: [6, 10, 10]
        if urgency <= 6:
            memberships['high'] = 0.0
        elif urgency <= 10:
            memberships['high'] = (urgency - 6) / 4
        else:
            memberships['high'] = 1.0

        return memberships

    def fuzzy_rules(self, battery_mf: Dict, distance_mf: Dict, 
                    urgency_mf: Dict) -> np.ndarray:
        """Fuzzy rule base with min-max inference"""
        output_fuzzy = np.zeros(len(self.priority_range))

        # Rule 1: Critical battery + very near + high urgency = Very High Priority
        rule1 = min(battery_mf.get('critical', 0), 
                   distance_mf.get('very_near', 0), 
                   urgency_mf.get('high', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule1 * self._priority_mf('very_high'))

        # Rule 2: Critical + near + high = Very High
        rule2 = min(battery_mf.get('critical', 0), 
                   distance_mf.get('near', 0), 
                   urgency_mf.get('high', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule2 * self._priority_mf('very_high'))

        # Rule 3: Critical + far = High
        rule3 = min(battery_mf.get('critical', 0), distance_mf.get('far', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule3 * self._priority_mf('high'))

        # Rule 4: Low battery + very near + high urgency = High
        rule4 = min(battery_mf.get('low', 0), 
                   distance_mf.get('very_near', 0), 
                   urgency_mf.get('high', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule4 * self._priority_mf('high'))

        # Rule 5: Low + near + medium urgency = Medium
        rule5 = min(battery_mf.get('low', 0), 
                   distance_mf.get('near', 0), 
                   urgency_mf.get('medium', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule5 * self._priority_mf('medium'))

        # Rule 6: Medium battery + low urgency = Low
        rule6 = min(battery_mf.get('medium', 0), urgency_mf.get('low', 0))
        output_fuzzy = np.fmax(output_fuzzy, rule6 * self._priority_mf('low'))

        # Rule 7: High battery = Very Low
        rule7 = battery_mf.get('high', 0)
        output_fuzzy = np.fmax(output_fuzzy, rule7 * self._priority_mf('very_low'))

        return output_fuzzy

    def _priority_mf(self, category: str) -> np.ndarray:
        """Output membership functions"""
        mf = np.zeros(len(self.priority_range))

        if category == 'very_low':
            for i, p in enumerate(self.priority_range):
                if p <= 15:
                    mf[i] = 1.0
                elif p <= 25:
                    mf[i] = (25 - p) / 10

        elif category == 'low':
            for i, p in enumerate(self.priority_range):
                if 20 <= p <= 35:
                    mf[i] = (p - 20) / 15
                elif 35 < p <= 50:
                    mf[i] = (50 - p) / 15

        elif category == 'medium':
            for i, p in enumerate(self.priority_range):
                if 45 <= p <= 60:
                    mf[i] = (p - 45) / 15
                elif 60 < p <= 75:
                    mf[i] = (75 - p) / 15

        elif category == 'high':
            for i, p in enumerate(self.priority_range):
                if 70 <= p <= 85:
                    mf[i] = (p - 70) / 15
                elif 85 < p <= 95:
                    mf[i] = (95 - p) / 10

        elif category == 'very_high':
            for i, p in enumerate(self.priority_range):
                if p <= 90:
                    mf[i] = 0.0
                elif p <= 95:
                    mf[i] = (p - 90) / 5
                else:
                    mf[i] = 1.0

        return mf

    def defuzzify_centroid(self, output_fuzzy: np.ndarray) -> float:
        """Centroid defuzzification"""
        numerator = np.sum(self.priority_range * output_fuzzy)
        denominator = np.sum(output_fuzzy)
        if denominator == 0:
            return 50.0
        return numerator / denominator

    @staticmethod
    def calculate_priority_score(vehicle_data: Dict, station_data: Dict, 
                                 cooperation_score: float = 0) -> float:
        """Main interface - compatible with existing code"""
        fuzzy_system = MamdaniFuzzyLogic()

        # Calculate distance
        distance_km = fuzzy_system._calculate_distance(
            vehicle_data['current_lat'], vehicle_data['current_lon'],
            station_data['lat'], station_data['lon']
        )

        # Get membership values
        battery_mf = fuzzy_system.battery_membership(vehicle_data['battery_level'])
        distance_mf = fuzzy_system.distance_membership(distance_km)
        urgency_mf = fuzzy_system.urgency_membership(
            vehicle_data.get('urgency_level', 5)
        )

        # Apply fuzzy rules
        output_fuzzy = fuzzy_system.fuzzy_rules(battery_mf, distance_mf, urgency_mf)

        # Defuzzify
        base_priority = fuzzy_system.defuzzify_centroid(output_fuzzy)

        # Add cooperation bonus
        final_priority = min(100, base_priority + cooperation_score * 0.1)

        return final_priority

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
class QLearningQueueOptimizer:
    """
    Reinforcement Learning for optimal queue assignment
    """

    def __init__(self, num_stations: int = 3):
        self.num_stations = num_stations
        self.alpha = 0.1  
        self.gamma = 0.95 
        self.epsilon = 0.15  
        self.q_table = {}

    def get_state(self, queue_lengths: List[int], priority: int, battery: int) -> str:
        """Discretize state"""
        battery_bin = min(battery // 20, 4)
        priority_bin = min(priority // 20, 4)
        state = tuple(queue_lengths + [priority_bin, battery_bin])
        return str(state)

    def select_station(self, queue_lengths: List[int], priority: int, 
                      battery: int, explore: bool = False) -> int:
        """Select best station using learned policy"""
        state = self.get_state(queue_lengths, priority, battery)

        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_stations)

        if state not in self.q_table:
            # Default: choose shortest queue
            return int(np.argmin(queue_lengths))

        return int(np.argmax(self.q_table[state]))

    def calculate_reward(self, queue_lengths: List[int], selected: int,
                        priority: int, battery: int) -> float:
        """Reward function"""
        queue_penalty = -queue_lengths[selected] * 2

        if battery < 20 and queue_lengths[selected] < 3:
            critical_bonus = 20
        else:
            critical_bonus = 0

        avg_queue = np.mean(queue_lengths)
        if abs(queue_lengths[selected] - avg_queue) < 1:
            fairness_bonus = 5
        else:
            fairness_bonus = 0

        return queue_penalty + critical_bonus + fairness_bonus + priority * 0.1

    def update(self, state: str, action: int, reward: float, next_state: str):
        """Q-learning update"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_stations)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_stations)

        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
# ============================================================================
# AI BATTERY PREDICTION ENGINE
# ============================================================================
class AIBatteryPredictor:
    """
    Simple AI-based battery health prediction and recommendation engine
    """

    @staticmethod
    def predict_battery_degradation(cycle_count: int, current_health: float, 
                                    usage_pattern: str = "normal") -> Dict:
        """
        Predict future battery health
        """
        degradation_rate = 0.05

        if usage_pattern == "aggressive":
            degradation_rate *= 1.5
        elif usage_pattern == "gentle":
            degradation_rate *= 0.7

        predictions = {}
        for future_cycles in [100, 500, 1000]:
            total_cycles = cycle_count + future_cycles
            predicted_health = max(50, 100 - (total_cycles * degradation_rate))
            predictions[f"after_{future_cycles}_cycles"] = round(predicted_health, 2)

        cycles_until_70_percent = int((current_health - 70) / degradation_rate)

        return {
            "predictions": predictions,
            "current_health": current_health,
            "cycles_until_service": max(0, cycles_until_70_percent),
            "degradation_rate_per_cycle": degradation_rate,
            "usage_pattern": usage_pattern
        }

    @staticmethod
    def recommend_charging_strategy(battery_level: int, battery_health: float, 
                                   destination_distance_km: float) -> Dict:
        """
        AI-powered charging recommendations
        """
        max_range = 400 * (battery_health / 100)
        current_range = (battery_level / 100) * max_range

        safety_margin_km = 50
        needed_range = destination_distance_km + safety_margin_km

        if current_range >= needed_range:
            charge_needed = False
            target_soc = battery_level
            reason = "Sufficient range for destination"
        else:
            charge_needed = True
            range_deficit = needed_range - current_range
            target_soc = min(100, battery_level + int((range_deficit / max_range) * 100) + 10)
            reason = f"Need {range_deficit:.1f} km additional range"

        if charge_needed:
            soc_increase = target_soc - battery_level
            estimated_time_min = int(soc_increase / 2)
        else:
            estimated_time_min = 0

        if battery_health < 80:
            recommendation = "Charge to 80% maximum to preserve battery health"
            target_soc = min(target_soc, 80)
        elif battery_level < 20:
            recommendation = "Fast charge to 50%, then slow charge to preserve health"
        else:
            recommendation = "Standard charging profile recommended"

        return {
            "charge_needed": charge_needed,
            "current_range_km": round(current_range, 1),
            "needed_range_km": needed_range,
            "target_soc": target_soc,
            "estimated_charge_time_min": estimated_time_min,
            "recommendation": recommendation,
            "reason": reason
        }

# ============================================================================
# QUEUE SWAP LOGIC (Standalone function for clean separation)
# ============================================================================
def process_queue_swap_request(self, db_manager: DatabaseManager, station_id: str, car1_id: str, car2_id: str, offer_points: int) -> Tuple[bool, str, List[Dict]]:
    """
    Complete queue swap implementation with real database operations
    """
    
    try:
        logger.info(f"Processing swap: {car1_id} <-> {car2_id} at {station_id} for {offer_points} RP")
        
        conn = sqlite3.connect(db_manager.db_path) # Use DB Manager path
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. GET CURRENT QUEUE
        cursor.execute("SELECT queue_json FROM station_queues WHERE station_id = ?", (station_id,))
        queue_row = cursor.fetchone()
        
        if not queue_row or not queue_row['queue_json']:
            conn.close()
            return False, f"No active queue found for station {station_id}", []
        
        queue = json.loads(queue_row['queue_json'])
        
        # 2. FIND CAR POSITIONS
        car1_idx = next((i for i, entry in enumerate(queue) if entry.get('carID') == car1_id), None)
        car2_idx = next((i for i, entry in enumerate(queue) if entry.get('carID') == car2_id), None)
        
        if car1_idx is None or car2_idx is None:
            conn.close()
            return False, f"One or both vehicles not found in queue", queue
        
        if car1_idx < car2_idx:
            conn.close()
            return False, f"Car {car1_id} is already ahead of {car2_id}. Swap unnecessary.", queue

        # 3. GET VEHICLE DATA AND URGENCY
        # Fetching cooperation_score (used as points/reward)
        car1_data = db_manager.execute_query("SELECT battery_level, cooperation_score FROM vehicles WHERE vehicle_id = ?", (car1_id,), fetch=True)
        car2_data = db_manager.execute_query("SELECT battery_level, cooperation_score FROM vehicles WHERE vehicle_id = ?", (car2_id,), fetch=True)
        
        if not car1_data or not car2_data:
            conn.close()
            return False, "Vehicle data retrieval failed.", queue

        car1_score = car1_data[0]['cooperation_score']
        
        car1_urgency = queue[car1_idx].get('urgency', 5)
        car2_urgency = queue[car2_idx].get('urgency', 5)
        
        # 4. CALCULATE BENEFIT SCORE (Car 2's perspective for accepting the deal)
        BENEFIT_POINTS_WEIGHT = 0.6
        BENEFIT_URGENCY_WEIGHT = 0.4
        
        # Normalize offered points (0-1 scale, based on typical max offer of 100 RP equivalent)
        # 4. CALCULATE BENEFIT SCORE (Car 2's perspective for accepting the deal)
        points_normalized = min(offer_points / 100.0, 1.0)
        urgency_delta = car1_urgency - car2_urgency

        # Get battery levels for enhanced urgency calculation
        car1_battery = car1_data[0]['battery_level']
        car2_battery = car2_data[0]['battery_level']
        battery_delta = car1_battery - car2_battery  # Negative if car1 has lower battery

        # Add battery-based bonus to benefit
        battery_factor = 0.0
        if battery_delta < -20:  # Car1 has 20%+ less battery
            battery_factor = 0.3
        elif battery_delta < -10:  # Car1 has 10%+ less battery
            battery_factor = 0.15

        urgency_score = max(0.0, min(1.0, 0.5 + urgency_delta / 20.0 + battery_factor))
        benefit = points_normalized * BENEFIT_POINTS_WEIGHT + urgency_score * BENEFIT_URGENCY_WEIGHT

        logger.info(f"[SWAP ANALYSIS] Battery(C1-C2):{battery_delta}%, Urgency:{urgency_delta}, Factor:{battery_factor:.2f}, Benefit:{benefit:.3f}")

        
        # 5. MAKE DECISION
        ACCEPT_THRESHOLD = 0.500
        
        if benefit >= ACCEPT_THRESHOLD:
            # ✅ ACCEPT SWAP
            
            # Use the cooperation score from the vehicle DB for points balance check
            if car1_score < offer_points:
                conn.close()
                logger.warning(f"❌ {car1_id} has insufficient points: {car1_score} < {offer_points}")
                return False, f"{car1_id} has insufficient cooperation score (score: {car1_score} < cost: {offer_points})", queue

            
            # Perform the swap in queue
            queue[car1_idx], queue[car2_idx] = queue[car2_idx], queue[car1_idx]
            
            # Update queue in database
            cursor.execute(
                "UPDATE station_queues SET queue_json = ?, queue_length = ?, last_updated = datetime('now') WHERE station_id = ?",
                (json.dumps(queue), len(queue), station_id)
            )
            
            # Transfer reward points (cooperation_score)
            cursor.execute(
                "UPDATE vehicles SET cooperation_score = cooperation_score - ? WHERE vehicle_id = ?",
                (offer_points, car1_id)
            )
            cursor.execute(
                "UPDATE vehicles SET cooperation_score = cooperation_score + ? WHERE vehicle_id = ?",
                (offer_points, car2_id)
            )
            
            # Log cooperation event
            cursor.execute(
                """INSERT INTO cooperation_history 
                   (vehicle_id, event_type, urgency_score, reward_points, timestamp, yielded_to)
                   VALUES (?, 'swap_offer_pay', ?, ?, datetime('now'), ?)""",
                (car1_id, car1_urgency, -offer_points, car2_id)
            )
            cursor.execute(
                """INSERT INTO cooperation_history 
                   (vehicle_id, event_type, urgency_score, reward_points, timestamp, yielded_to)
                   VALUES (?, 'swap_accept_receive', ?, ?, datetime('now'), ?)""",
                (car2_id, car2_urgency, offer_points, car1_id)
            )
            
            conn.commit()
            conn.close()
            
            message = f"✅ SWAP ACCEPTED! {car1_id} (pos {car1_idx}) <-> {car2_id} (pos {car2_idx})"
            logger.info(message)
            
            return True, message, queue
        
        else:
            # ❌ REJECT SWAP
            conn.close()
            message = f"❌ SWAP REJECTED! Benefit {benefit:.3f} < {ACCEPT_THRESHOLD}. Insufficient offer."
            logger.info(message)
            
            return False, message, queue
    
    except Exception as e:
        logger.error(f"Error processing swap: {e}")
        import traceback
        traceback.print_exc()
        try:
            conn.close()
        except:
            pass
        return False, f"Internal error during swap: {e}", []


# ============================================================================
# QR CODE MANAGER
# ============================================================================
class QRCodeManager:
    """Handles QR code generation, validation, and timeout management"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.active_qr_codes = {}
        self.lock = threading.Lock()

    def generate_qr(self, vehicle_id: str, station_id: str, 
                   charge_minutes: int) -> Dict:
        """Generate QR code for charging authorization"""
        timestamp = int(time.time())
        qr_data = f"{vehicle_id}|{station_id}|{timestamp}|{charge_minutes}"
        qr_id = hashlib.md5(qr_data.encode()).hexdigest()[:12]

        expires_at = timestamp + config.QR_TIMEOUT_SECONDS

        with self.lock:
            self.active_qr_codes[qr_id] = {
                'vehicle_id': vehicle_id,
                'station_id': station_id,
                'qr_data': qr_data,
                'generated_at': timestamp,
                'expires_at': expires_at,
                'charge_minutes': charge_minutes,
                'status': 'active'
            }

        # Store in database
        self.db.execute_query(
            """INSERT INTO qr_codes (qr_id, vehicle_id, station_id, qr_data, 
               generated_at, expires_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (qr_id, vehicle_id, station_id, qr_data, 
             datetime.fromtimestamp(timestamp), 
             datetime.fromtimestamp(expires_at), 'active')
        )

        logger.info(f"🔲 QR generated: {qr_id} for {vehicle_id} at {station_id} ({charge_minutes} min)")

        return {
            'qr_id': qr_id,
            'qr_data': qr_data,
            'expires_in_seconds': config.QR_TIMEOUT_SECONDS,
            'charge_minutes': charge_minutes
        }

    def validate_qr(self, qr_data: str, scanned_by_station: str) -> Dict:
        """Validate scanned QR code"""
        parts = qr_data.split('|')
        if len(parts) != 4:
            return {'valid': False, 'reason': 'Invalid QR format'}

        vehicle_id, station_id, timestamp_str, charge_minutes_str = parts

        # Check if station matches
        if station_id != scanned_by_station:
            return {'valid': False, 'reason': 'Station mismatch'}

        # Check timeout
        timestamp = int(timestamp_str)
        current_time = int(time.time())

        if current_time > timestamp + config.QR_TIMEOUT_SECONDS:
            return {'valid': False, 'reason': 'QR code expired'}

        # Check if already scanned
        qr_id = hashlib.md5(qr_data.encode()).hexdigest()[:12]

        with self.lock:
            # Check active in-memory list first
            if qr_id in self.active_qr_codes:
                qr_info = self.active_qr_codes[qr_id]
                if qr_info['status'] == 'scanned':
                    return {'valid': False, 'reason': 'QR already used'}

                # Mark as scanned
                qr_info['status'] = 'scanned'
                self.active_qr_codes[qr_id] = qr_info

        # Update database
        self.db.execute_query(
            "UPDATE qr_codes SET scanned = 1, scanned_at = ?, status = 'scanned' WHERE qr_id = ?",
            (datetime.now(), qr_id)
        )

        logger.info(f"✅ QR validated: {qr_id} for {vehicle_id}")

        return {
            'valid': True,
            'vehicle_id': vehicle_id,
            'station_id': station_id,
            'charge_minutes': int(charge_minutes_str),
            'qr_id': qr_id
        }

    def cleanup_expired_qr(self):
        """Remove expired QR codes"""
        current_time = int(time.time())
        expired_ids = []

        with self.lock:
            for qr_id, qr_info in list(self.active_qr_codes.items()):
                if current_time > qr_info['expires_at'] and qr_info['status'] == 'active':
                    qr_info['status'] = 'expired'
                    expired_ids.append(qr_id)
                # Remove stale entries regardless of status if they are very old (e.g., 2 hours past expiry)
                elif current_time > qr_info['expires_at'] + 7200: 
                    del self.active_qr_codes[qr_id]

        if expired_ids:
            # Need to convert list of IDs to a safe tuple of parameters for SQL
            placeholders = ','.join(['?'] * len(expired_ids))
            self.db.execute_query(
                f"UPDATE qr_codes SET status = 'expired' WHERE qr_id IN ({placeholders})",
                tuple(expired_ids)
            )
            logger.info(f"🗑️  Cleaned up {len(expired_ids)} expired QR codes in DB")

# ============================================================================
# BILLING MANAGER
# ============================================================================
class BillingManager:
    """Handles charging session billing and overtime calculations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def start_charging_session(self, vehicle_id: str, station_id: str, 
                              qr_id: str, allocated_minutes: int, 
                              start_battery: int) -> int:
        """Start a new charging session"""
        session_id = self.db.execute_query(
            """INSERT INTO charging_sessions (vehicle_id, station_id, qr_id, 
               start_time, allocated_minutes, start_battery, status)
               VALUES (?, ?, ?, ?, ?, ?, 'active')""",
            (vehicle_id, station_id, qr_id, datetime.now(), 
             allocated_minutes, start_battery)
        )

        logger.info(f"💰 Charging session started: #{session_id} for {vehicle_id} ({allocated_minutes} min)")
        return session_id

    def end_charging_session(self, session_id: int, end_battery: int, 
                            actual_minutes: int) -> Dict:
        """End charging session and calculate billing"""
        # Fetch session details
        session = self.db.execute_query(
            "SELECT * FROM charging_sessions WHERE session_id = ?",
            (session_id,), fetch=True
        )

        if not session:
            return {'error': 'Session not found'}

        session = session[0]
        allocated_minutes = session['allocated_minutes']
        start_battery = session['start_battery']
        station_id = session['station_id']

        # Calculate energy delivered (simplified: 1 kWh per 2% charge for 60 kWh battery)
        battery_increase = end_battery - start_battery
        energy_kwh = (battery_increase / 100.0) * 60  # Assuming 60 kWh battery standard

        # Calculate costs
        electricity_cost = energy_kwh * config.ELECTRICITY_RATE_PER_KWH
        parking_cost = (allocated_minutes / 60.0) * config.PARKING_RATE_PER_HOUR

        # Overtime calculation
        overtime_minutes = max(0, actual_minutes - allocated_minutes)
        overtime_cost = 0.0

        if overtime_minutes > 0:
            overtime_hours = overtime_minutes / 60.0
            overtime_cost = (
                overtime_hours * config.PARKING_RATE_PER_HOUR * config.OVERTIME_MULTIPLIER 
            )

        total_cost = electricity_cost + parking_cost + overtime_cost

        # Apply cooperation bonus placeholder (will be updated by a separate call if applicable)
        coop_bonus = session.get('cooperation_bonus', 0.0)
        final_cost = total_cost - coop_bonus

        # Update session
        self.db.execute_query(
            """UPDATE charging_sessions 
               SET end_time = ?, end_battery = ?, actual_minutes = ?,
                   overtime_minutes = ?, energy_delivered_kwh = ?,
                   electricity_cost = ?, parking_cost = ?, overtime_cost = ?,
                   total_cost = ?, status = 'completed'
               WHERE session_id = ?""",
            (datetime.now(), end_battery, actual_minutes, overtime_minutes,
             energy_kwh, electricity_cost, parking_cost, overtime_cost,
             final_cost, session_id)
        )

        billing_details = {
            'session_id': session_id,
            'vehicle_id': session['vehicle_id'],
            'station_id': station_id,
            'energy_delivered_kwh': round(energy_kwh, 2),
            'electricity_cost': round(electricity_cost, 2),
            'parking_cost': round(parking_cost, 2),
            'overtime_minutes': overtime_minutes,
            'overtime_cost': round(overtime_cost, 2),
            'total_cost': round(final_cost, 2),
            'allocated_minutes': allocated_minutes,
            'actual_minutes': actual_minutes
        }

        logger.info(f"💰 Session #{session_id} completed: ₹{final_cost:.2f}")

        return billing_details

    def apply_cooperation_discount(self, session_id: int, cooperation_score: float):
        """Apply discount based on cooperation score"""
        # Max discount percentage is 20%
        # cooperation_score is 0-10, so / 5 scales to 0-20%
        discount_percentage = min(20, cooperation_score / 5) 

        session_data = self.db.execute_query(
            "SELECT total_cost, electricity_cost FROM charging_sessions WHERE session_id = ?",
            (session_id,), fetch=True
        )

        if session_data:
            original_cost = session_data[0]['total_cost']
            # Discount applied only to electricity component
            electricity_cost = session_data[0]['electricity_cost']
            
            discount_amount = electricity_cost * (discount_percentage / 100)
            new_total = original_cost - discount_amount

            self.db.execute_query(
                "UPDATE charging_sessions SET cooperation_bonus = cooperation_bonus + ?, total_cost = ? WHERE session_id = ?",
                (discount_amount, new_total, session_id)
            )

            logger.info(f"💎 Cooperation discount applied to session #{session_id}: "
                       f"{discount_percentage:.1f}% (₹{discount_amount:.2f})")

# ============================================================================
# SMART ROUTING ENGINE
# ============================================================================
class SmartRoutingEngine:
    """Handles intelligent routing with destination preferences and rerouting"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def calculate_route_with_charging(self, vehicle_id: str, 
                                     destination_lat: float, destination_lon: float) -> Dict:
        """Calculate optimal route with charging stops"""
        
        vehicle_data = self.db.execute_query(
            "SELECT * FROM vehicles WHERE vehicle_id = ?",
            (vehicle_id,), fetch=True
        )
        if not vehicle_data:
            return {'error': 'Vehicle not found'}
        vehicle = vehicle_data[0]

        battery_level = vehicle['battery_level']
        battery_health = vehicle['battery_health']
        current_lat = vehicle['current_lat']
        current_lon = vehicle['current_lon']
        
        # Handle initial zero coordinates safely
        if current_lat is None or current_lon is None or destination_lat is None or destination_lon is None:
            return {'error': 'Invalid coordinates provided for routing'}
        if current_lat == 0 and current_lon == 0:
            return {'error': 'Vehicle location unknown (0, 0)'}


        total_distance = MamdaniFuzzyLogic._haversine_distance(
            current_lat, current_lon, destination_lat, destination_lon
        )

        max_range_km = 400 * (battery_health / 100)
        current_range = (battery_level / 100) * max_range_km
        safety_margin = 30  # km

        if current_range - safety_margin >= total_distance:
            return {
                'charging_needed': False,
                'direct_route': True,
                'total_distance_km': round(total_distance, 1),
                'estimated_battery_on_arrival': int(battery_level - (total_distance / max_range_km * 100))
            }

        # Find charging stations along route
        stations = self.db.execute_query(
            "SELECT * FROM charging_stations WHERE operational = 1",
            fetch=True
        )

        suitable_stations = []
        for station in stations:
            # Skip stations with invalid coordinates
            if station['lat'] == 0.0 and station['lon'] == 0.0:
                 continue
                 
            dist_to_station = MamdaniFuzzyLogic._haversine_distance(
                current_lat, current_lon, station['lat'], station['lon']
            )

            dist_station_to_dest = MamdaniFuzzyLogic._haversine_distance(
                station['lat'], station['lon'], destination_lat, destination_lon
            )

            detour = (dist_to_station + dist_station_to_dest) - total_distance

            if detour <= config.MAX_DETOUR_KM and dist_to_station < current_range - safety_margin:
                suitable_stations.append({
                    'station': station,
                    'distance_from_current': round(dist_to_station, 1),
                    'distance_to_destination': round(dist_station_to_dest, 1),
                    'detour_km': round(detour, 1)
                })

        suitable_stations.sort(key=lambda x: (x['detour_km'], -x['station']['available_slots']))

        if not suitable_stations:
            return {
                'charging_needed': True,
                'stations_found': False,
                'recommendation': 'No stations nearby that meet detour criteria.'
            }

        recommended_station = suitable_stations[0]

        distance_to_station = recommended_station['distance_from_current']
        remaining_distance = recommended_station['distance_to_destination']

        battery_on_arrival_station = battery_level - (distance_to_station / max_range_km * 100)
        needed_battery_for_dest = (remaining_distance / max_range_km * 100) + 20  # +20% buffer

        charge_needed = max(0, needed_battery_for_dest - battery_on_arrival_station)
        estimated_charge_time = int(charge_needed / 2) 

        return {
            'charging_needed': True,
            'stations_found': True,
            'recommended_station': recommended_station['station']['station_id'],
            'station_name': recommended_station['station']['name'],
            'distance_to_station_km': distance_to_station,
            'estimated_charge_time_min': estimated_charge_time,
            'total_journey_time_min': int((total_distance / 60) * 60 + estimated_charge_time) 
        }

    def handle_rerouting_request(self, vehicle_id: str, urgency: int = 9) -> Dict:
        """Handle emergency rerouting for low battery"""
        vehicle_data = self.db.execute_query(
            "SELECT * FROM vehicles WHERE vehicle_id = ?",
            (vehicle_id,), fetch=True
        )
        if not vehicle_data:
            return {'success': False, 'message': 'Vehicle not found'}
            
        vehicle = vehicle_data[0]
        current_lat = vehicle.get('current_lat', 0)
        current_lon = vehicle.get('current_lon', 0)
        current_range = vehicle.get('range_km', 0)
        
        if current_lat == 0 and current_lon == 0:
            return {'success': False, 'message': 'Vehicle location unknown'}


        stations = self.db.execute_query(
            "SELECT * FROM charging_stations WHERE operational = 1 AND available_slots > 0",
            fetch=True
        )

        nearby_stations = []
        for station in stations:
            # Skip stations with invalid coordinates
            if station['lat'] == 0.0 and station['lon'] == 0.0:
                 continue
                 
            distance = MamdaniFuzzyLogic._haversine_distance(
                current_lat, current_lon, station['lat'], station['lon']
            )

            if distance < current_range * 0.8: # 80% of range for safety
                nearby_stations.append({
                    'station': station,
                    'distance_km': round(distance, 1),
                    'eta_minutes': int((distance / 60) * 60)
                })

        nearby_stations.sort(key=lambda x: x['distance_km'])

        if not nearby_stations:
            logger.error(f"❌ NO REACHABLE STATIONS for {vehicle_id}!")
            return {
                'success': False,
                'message': 'No charging stations within safe range',
                'recommendation': 'Request roadside assistance'
            }

        return {
            'success': True,
            'nearest_stations': nearby_stations[:3],
            'recommended_station_id': nearby_stations[0]['station']['station_id'],
            'urgency_level': urgency
        }

# ============================================================================
# HOTEL & DESTINATION MANAGER
# ============================================================================
class HotelDestinationManager:
    """Manages user destinations (home, friends) and hotel charging reservations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._initialize_sample_hotels()

    def _initialize_sample_hotels(self):
        """Add sample hotel partnerships"""
        sample_hotels = [
            ("Grand Kerala Resort", 9.9674, 76.2807, 4, 20, 35.0, "Pool,Restaurant,Spa", 4.5, "+91-9876543210"),
            ("Hilltop Charging Inn", 9.9676, 76.3216, 6, 30, 40.0, "Restaurant,WiFi,Parking", 4.2, "+91-9876543211"),
            ("Coastal EV Hotel", 10.0014, 76.3159, 8, 40, 45.0, "Beach,Restaurant,Charging Lounge", 4.7, "+91-9876543212"),
        ]

        for hotel in sample_hotels:
            self.db.execute_query(
                """INSERT OR IGNORE INTO hotels (name, lat, lon, charging_stations, 
                   parking_slots, combined_rate_hour, amenities, rating, phone)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                hotel
            )

    def add_user_destination(self, vehicle_id: str, dest_type: str, 
                           dest_name: str, lat: float, lon: float,
                           has_charging: bool = False, notes: str = "") -> int:
        """Add personal destination (home, friend's house, work, etc.)"""
        dest_id = self.db.execute_query(
            """INSERT INTO user_destinations (vehicle_id, dest_type, dest_name, 
               lat, lon, has_charging, notes) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (vehicle_id, dest_type, dest_name, lat, lon, has_charging, notes)
        )

        logger.info(f"📍 Added {dest_type} destination '{dest_name}' for {vehicle_id}")
        return dest_id

    def create_hotel_reservation(self, vehicle_id: str, hotel_id: int,
                                checkin_time: datetime, hours: int) -> Dict:
        """Create combined hotel + charging reservation"""
        hotel = self.db.execute_query(
            "SELECT * FROM hotels WHERE hotel_id = ?",
            (hotel_id,), fetch=True
        )
        
        if not hotel:
            return {'error': 'Hotel not found'}

        hotel = hotel[0]
        checkout_time = checkin_time + timedelta(hours=hours)

        reservation_id = self.db.execute_query(
            """INSERT INTO reservations (vehicle_id, station_id, reservation_type,
               start_time, end_time, destination_type, destination_name,
               destination_lat, destination_lon, status)
               VALUES (?, ?, 'hotel', ?, ?, 'hotel', ?, ?, ?, 'confirmed')""",
            (vehicle_id, f"HOTEL_{hotel_id}", checkin_time, checkout_time,
             hotel['name'], hotel['lat'], hotel['lon'])
        )

        total_cost = hotel['combined_rate_hour'] * hours

        logger.info(f"🏨 Hotel reservation created: {hotel['name']} for {vehicle_id} ")

        return {
            'reservation_id': reservation_id,
            'hotel_name': hotel['name'],
            'checkin': checkin_time.isoformat(),
            'checkout': checkout_time.isoformat(),
            'hours': hours,
            'estimated_cost': total_cost,
            'charging_available': True,
        }

# ============================================================================
# MAIN CHARGEUP CONTROLLER
# ============================================================================
class ChargeUpController:
    """Main system controller orchestrating all components"""

    def __init__(self):
        logger.info("=" * 80)
        logger.info("ChargeUp EV Management System v2.0 - Initializing")
        logger.info("=" * 80)

        self.config = config
        self.db = DatabaseManager()
        self.qr_manager = QRCodeManager(self.db)
        self.billing_manager = BillingManager(self.db)
        self.routing_engine = SmartRoutingEngine(self.db)
        self.hotel_manager = HotelDestinationManager(self.db)
        self.fuzzy_logic = MamdaniFuzzyLogic()
        self.ai_predictor = AIBatteryPredictor()
        
        self.mqtt_client = None
        self._subscribed = False

        self.active_sessions = {}
        self.station_queues = {} 

        if config.SIMULATION_MODE:
            self._initialize_simulation_data()

        self.running = True
        self.qr_cleanup_thread = threading.Thread(target=self._qr_cleanup_worker, daemon=True)
        self.qr_cleanup_thread.start()

        logger.info("✅ ChargeUp Controller initialized successfully")

    # ---------------------------------------------------------------------
    # START / STOP + MQTT HANDLERS
    # ---------------------------------------------------------------------

    def start(self):
        """Start the main controller and MQTT client (safe, single start)."""
        try:
            pid = os.getpid()
            hostname = socket.gethostname().split('.')[0]
            client_id = f"ChargeUpController_v2_{hostname}_{pid}"

            self.mqtt_client = mqtt.Client(client_id=client_id)
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

            self.mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
            self.mqtt_client.loop_start()

            logger.info("MQTT client started (loop thread running). Press Ctrl+C to stop.")

            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in start(): {e}", exc_info=True)
            self.stop()

    def stop(self):
        """Stop the controller cleanly."""
        self.running = False
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop(force=False)
                self.mqtt_client.disconnect()
        except Exception as e:
            logger.error(f"Error while stopping MQTT client: {e}")

        logger.info("ChargeUp Controller stopped")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("Connected to MQTT broker (rc=0)")
            if not self._subscribed:
                topics = [
                    "chargeup/telemetry/#",
                    "chargeup/commands/#",
                    "chargeup/qr/+/scan",
                    "chargeup/rerouting/request",
                    "chargeup/billing/overtime",
                    "chargeup/health/#",
                    "chargeup/reservation/request",
                    "chargeup/destination/add",
                    "chargeup/queue_manager/swap_request/#",
                    "chargeup/queue_manager/remove_request", 
                    "chargeup/station/+/queue_command",
                    "chargeup/station/+/status"
                ]

                for topic in topics:
                    client.subscribe(topic, qos=1)
                    logger.debug(f"Subscribed: {topic}")
                self._subscribed = True
        else:
            logger.error(f"MQTT connect returned rc={rc}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback."""
        if rc != 0:
            logger.warning(f"MQTT unexpectedly disconnected (rc={rc}).")
            self._subscribed = False

    def on_mqtt_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            # Decode payload, assuming JSON for most commands
            payload_str = msg.payload.decode()
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                payload = payload_str # Keep as string if not JSON
        except Exception:
            logger.warning(f"Failed to decode MQTT payload for topic {topic}")
            return
        
        logger.debug(f"MQTT message received on: {topic}")
        
        # Priority handler: SWAP REQUEST
        if '/queue_manager/swap_request' in topic:
            self._handle_swap_request(payload)
            return
        
        # REMOVE REQUEST HANDLER
        if topic == 'chargeup/queue_manager/remove_request':
            try:
                station_id = payload.get('stationID')
                car_id = payload.get('carID')
                
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT queue_json FROM station_queues WHERE station_id = ?", (station_id,))
                result = cursor.fetchone()
                
                if not result or not result[0]:
                    logger.error(f"[REMOVE] No queue found for {station_id}")
                    conn.close()
                    return
                
                queue = json.loads(result[0])
                queue = [v for v in queue if v['carID'] != car_id]
                
                cursor.execute("""
                    UPDATE station_queues 
                    SET queue_json = ?, queue_length = ?, last_updated = datetime('now')
                    WHERE station_id = ?
                """, (json.dumps(queue), len(queue), station_id))
                
                conn.commit()
                conn.close()
                logger.info(f"[REMOVE] SUCCESS! Removed {car_id} from {station_id}")
                
            except Exception as e:
                logger.error(f"[REMOVE] Error: {e}")
                
            return
        
        # Other handlers
        try:
            if "/telemetry/" in topic:
                self._handle_telemetry(payload)
            elif "/qr/" in topic and "/scan" in topic:
                self._handle_qr_scan(payload)
            elif "/rerouting/request" in topic:
                self._handle_rerouting_request(payload)
            elif "/billing/overtime" in topic:
                # Placeholder for overtime logic if the front end triggers it explicitly
                pass
            elif "/health/" in topic:
                self._handle_health_alert(payload)
            elif "/reservation/request" in topic:
                self._handle_reservation_request(payload)
            elif "/destination/add" in topic:
                self._handle_destination_add(payload)
            elif "queue_command" in topic:
                station_id = topic.split('/')[2]
                self._handle_queue_command(station_id, payload)
            elif "status" in topic and "station" in topic:
                station_id = topic.split('/')[2]
                self._handle_station_status(station_id, payload)
            else:
                logger.debug(f"Unhandled topic: {topic}")
        except Exception as e:
            logger.error(f"Error processing MQTT topic {topic}: {e}\n{traceback.format_exc()}")


    # ---------------------------------------------------------------------
    # INTERNAL HANDLERS
    # ---------------------------------------------------------------------
    
    def _handle_swap_request(self, payload):
        """Handle queue swap request"""
        try:
            station_id = payload.get('stationID')
            car1_id = payload.get('carID')
            car2_id = payload.get('carID2')
            offer_points = payload.get('offerPoints', 0)
            
            # Use the dedicated function for robust swap logic
            success, message, new_queue = process_queue_swap_request(
                self, self.db, station_id, car1_id, car2_id, offer_points
            )
            
            result_payload = {
                "success": success,
                "message": message,
                "station": station_id,
                "newQueue": [v['carID'] for v in new_queue] # Simplify queue list for transmission
            }
            
            # Publish result
            result_topic = f"chargeup/queue_manager/swap_result/{station_id}"
            self.mqtt_client.publish(result_topic, json.dumps(result_payload))
            logger.info(f"Swap result published: {success}")
            
        except Exception as e:
            logger.error(f"Swap error: {e}")
            import traceback
            traceback.print_exc()

    def _handle_telemetry(self, data: Dict):
        """Process vehicle telemetry"""
        vehicle_id = data.get('device')

        # Update database (Ensure cooperation_score is present for insertion)
        self.db.execute_query(
            """INSERT OR REPLACE INTO vehicles 
               (vehicle_id, battery_level, battery_health, current_lat, current_lon, 
                range_km, charging, status, last_updated, cooperation_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT cooperation_score FROM vehicles WHERE vehicle_id = ?), ?))""",
            (vehicle_id, data.get('battery_level', 0), data.get('battery_health', 100.0), 
             data.get('lat', 0.0), data.get('lon', 0.0), data.get('range_km', 0.0), 
             data.get('charging', 0), data.get('status', 'IDLE'), datetime.now(),
             vehicle_id, 0.0) # COALESCE handles insertion if not exists, keeps old score if it does
        )

        if data.get('battery_level', 100) <= config.CRITICAL_BATTERY_THRESHOLD and not data.get('charging', 0):
            self._trigger_auto_rerouting(vehicle_id)

        if not data.get('charging', 0) and vehicle_id in self.active_sessions:
            self._handle_charging_stop(vehicle_id, data.get('battery_level', 0))


    def _handle_qr_scan(self, data: Dict):
        """Handle QR code scan from station"""
        station_id = data.get('station_id')
        qr_data = data.get('qr_data')

        validation = self.qr_manager.validate_qr(qr_data, station_id)

        if validation['valid']:
            vehicle_id = validation['vehicle_id']
            charge_minutes = validation['charge_minutes']

            vehicle_data = self.db.execute_query("SELECT battery_level, cooperation_score FROM vehicles WHERE vehicle_id = ?", (vehicle_id,), fetch=True)
            if not vehicle_data:
                logger.error(f"Vehicle data not found for QR validation: {vehicle_id}")
                return

            session_id = self.billing_manager.start_charging_session(
                vehicle_id, station_id, validation['qr_id'], charge_minutes, vehicle_data[0]['battery_level']
            )

            self.active_sessions[vehicle_id] = {
                'session_id': session_id,
                'station_id': station_id,
                'start_time': time.time(),
                'allocated_minutes': charge_minutes
            }

            self.mqtt_client.publish(
                f"chargeup/charging/{vehicle_id}/start",
                json.dumps({'session_id': session_id, 'allocated_minutes': charge_minutes})
            )

            self.mqtt_client.publish(
                f"chargeup/station/{station_id}/command",
                json.dumps({'command': 'START_CHARGE', 'vehicle_id': vehicle_id, 'session_id': session_id})
            )
            
        else:
            self.mqtt_client.publish(
                f"chargeup/station/{station_id}/qr_result",
                json.dumps({'valid': False, 'reason': validation['reason']})
            )

    def _handle_charging_stop(self, vehicle_id: str, end_battery: int):
        """Handles session completion based on vehicle telemetry or station command."""
        session_info = self.active_sessions.pop(vehicle_id, None)

        if session_info:
            session_id = session_info['session_id']
            actual_minutes = int((time.time() - session_info['start_time']) / 60)
            
            billing_details = self.billing_manager.end_charging_session(
                session_id, end_battery, actual_minutes
            )
            
            vehicle_data = self.db.execute_query("SELECT cooperation_score FROM vehicles WHERE vehicle_id = ?", (vehicle_id,), fetch=True)
            if vehicle_data:
                self.billing_manager.apply_cooperation_discount(session_id, vehicle_data[0]['cooperation_score'])

            self._update_station_slots(session_info['station_id'], 1)

            self.mqtt_client.publish(
                f"chargeup/billing/{vehicle_id}/completed",
                json.dumps(billing_details)
            )
            logger.info(f"Session {session_id} finished for {vehicle_id}.")

    def _handle_rerouting_request(self, data: Dict):
        """Handle emergency rerouting"""
        vehicle_id = data.get('car_id')
        urgency = data.get('urgency', 9)

        rerouting_result = self.routing_engine.handle_rerouting_request(vehicle_id, urgency)

        self.mqtt_client.publish(
            f"chargeup/rerouting/{vehicle_id}/response",
            json.dumps(rerouting_result)
        )

    def _trigger_auto_rerouting(self, vehicle_id: str):
        """Automatically trigger rerouting for critical battery"""
        logger.warning(f"🚨 AUTO-REROUTING triggered for {vehicle_id}")
        rerouting_result = self.routing_engine.handle_rerouting_request(vehicle_id, urgency=10)
        self.mqtt_client.publish(
            f"chargeup/rerouting/{vehicle_id}/auto",
            json.dumps(rerouting_result)
        )
        
    def _handle_reservation_request(self, data: Dict):
        """Handle charging/hotel reservation"""
        vehicle_id = data['vehicle_id']
        station_id = data.get('station_id')
        
        # Simple charging reservation (assuming duration is passed)
        if station_id:
            start_time = datetime.fromisoformat(data['start_time'])
            duration_minutes = data['duration_minutes']
            end_time = start_time + timedelta(minutes=duration_minutes)

            res_id = self.db.execute_query(
                """INSERT INTO reservations (vehicle_id, station_id, reservation_type,
                   start_time, end_time, estimated_charge_time, status)
                   VALUES (?, ?, 'charging', ?, ?, ?, 'confirmed')""",
                (vehicle_id, station_id, start_time, end_time, duration_minutes)
            )
            logger.info(f"Reservation confirmed for {vehicle_id} at {station_id}, ID: {res_id}")
            self.mqtt_client.publish(
                f"chargeup/reservations/status/{vehicle_id}",
                json.dumps({'status': 'CONFIRMED', 'reservation_id': res_id, 'station_id': station_id, 'start_time': start_time.isoformat()})
            )
        
        # NOTE: Hotel reservation logic is incomplete in payload handling, keeping simple charge res.

    def _handle_destination_add(self, data: Dict):
        """Handle adding a new user destination"""
        try:
            vehicle_id = data['vehicle_id']
            dest_type = data['dest_type']
            dest_name = data['dest_name']
            lat = data['lat']
            lon = data['lon']
            
            self.hotel_manager.add_user_destination(
                vehicle_id, dest_type, dest_name, lat, lon
            )
            self.mqtt_client.publish(
                f"chargeup/destination/status/{vehicle_id}",
                json.dumps({'status': 'SUCCESS', 'message': f"Added destination: {dest_name}"})
            )
        except Exception as e:
            logger.error(f"Failed to add destination: {e}")
            self.mqtt_client.publish(
                f"chargeup/destination/status/{data.get('vehicle_id', 'UNKNOWN')}",
                json.dumps({'status': 'FAILED', 'message': str(e)})
            )


    def _handle_station_status(self, station_id: str, data: Dict):
        """Update station status and capacity."""
        available = data.get('available_slots')
        operational = data.get('operational')

        update_fields = []
        params = []
        if available is not None:
            update_fields.append("available_slots = ?")
            params.append(available)
        if operational is not None:
            update_fields.append("operational = ?")
            params.append(operational)

        if update_fields:
            params.append(station_id)
            query = f"UPDATE charging_stations SET {', '.join(update_fields)}, last_updated = datetime('now') WHERE station_id = ?"
            self.db.execute_query(query, tuple(params))
            logger.info(f"Station {station_id} status updated. Available: {available}")

    def _update_station_slots(self, station_id: str, change: int):
        """Utility to safely increment/decrement available slots."""
        self.db.execute_query(
            "UPDATE charging_stations SET available_slots = available_slots + ? WHERE station_id = ?",
            (change, station_id)
        )


    def _handle_queue_command(self, station_id: str, data: Dict):
        """Handle specific queue commands like adding or removing a vehicle."""
        command = data.get('command')
        car_id = data.get('car_id')
        
        if command == "JOIN":
            vehicle_data = self.db.execute_query(
                "SELECT battery_level, current_lat, current_lon, cooperation_score FROM vehicles WHERE vehicle_id = ?", 
                (car_id,), 
                fetch=True
            )
            
            if vehicle_data:
                battery = vehicle_data[0]['battery_level']
                vlat = vehicle_data[0]['current_lat']
                vlon = vehicle_data[0]['current_lon']
                coop_score = vehicle_data[0]['cooperation_score']
                
                # Get station data
                station_data = self.db.execute_query(
                    "SELECT lat, lon FROM charging_stations WHERE station_id = ?",
                    (station_id,),
                    fetch=True
                )
                
                # Calculate urgency (0-10 scale based on battery)
                if battery <= 10:
                    urgency = 10
                elif battery <= 20:
                    urgency = 9
                elif battery <= 30:
                    urgency = 7
                elif battery <= 50:
                    urgency = 5
                else:
                    urgency = max(1, int((100 - battery) / 10))
                
                # Calculate priority if station data available
                if station_data:
                    slat = station_data[0]['lat']
                    slon = station_data[0]['lon']
                    
                    vehicle_context = {
                        'battery_level': battery,
                        'current_lat': vlat,
                        'current_lon': vlon,
                        'urgency_level': urgency,
                        'battery_health': 100.0
                    }
                    station_context = {'lat': slat, 'lon': slon}
                    
                    priority = MamdaniFuzzyLogic.calculate_priority_score(
                        vehicle_context, 
                        station_context, 
                        coop_score
                    )
                else:
                    priority = 50.0
            else:
                battery = 50
                urgency = 5
                priority = 50.0
                coop_score = 5.0
            
            queue_data = self.db.execute_query(
                "SELECT queue_json FROM station_queues WHERE station_id = ?", 
                (station_id,), 
                fetch=True
            )
            queue = json.loads(queue_data[0]['queue_json']) if (queue_data and queue_data[0]['queue_json']) else []
            
            new_entry = {
                "carID": car_id, 
                "battery": battery, 
                "urgency": urgency,
                "priority": round(priority, 2),
                "cooperationScore": round(coop_score, 2),
                "timestamp": datetime.now().isoformat()
            }
            queue.append(new_entry)
            
            logger.info(f"✅ {car_id} joined queue: Priority={priority:.1f}, Urgency={urgency}, Battery={battery}%")

            
            # Save back to DB
            self.db.execute_query(
                "UPDATE station_queues SET queue_json = ?, queue_length = ? WHERE station_id = ?",
                (json.dumps(queue), len(queue), station_id)
            )
            logger.info(f"Car {car_id} joined queue at {station_id}. New length: {len(queue)}")
            self.mqtt_client.publish(f"chargeup/queue_manager/update/{station_id}", json.dumps(queue))


    def _qr_cleanup_worker(self):
        """Background thread to cleanup expired QR codes"""
        while self.running:
            time.sleep(config.QR_TIMEOUT_SECONDS / 2)
            self.qr_manager.cleanup_expired_qr()

    def _initialize_simulation_data(self):
        """Initialize simulation data with VALID Kerala coordinates"""
        logger.info("🎮 Initializing simulation data...")
        
        # Add sample charging stations (VALID Kochi area coordinates)
        stations = [
            ("STN01", "TechPark Charging Hub", 9.9674, 76.2807, 4, 4, 7.5, 20.0),
            ("STN02", "City Center ChargePoint", 9.9600, 76.2800, 6, 5, 7.5, 20.0),
            ("STN03", "Highway Express Charge", 10.0014, 76.3159, 8, 6, 7.5, 20.0),
        ]
        
        for station in stations:
            self.db.execute_query(
                """INSERT OR REPLACE INTO charging_stations
                (station_id, name, lat, lon, total_slots, available_slots,
                charging_rate_kwh, parking_rate_hour)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                station
            )
            
            # Initialize empty queue structure
            self.db.execute_query(
                "INSERT OR IGNORE INTO station_queues (station_id, queue_json, queue_length) VALUES (?, ?, 0)",
                (station[0], "[]")
            )
        
        # **CRITICAL FIX: Add vehicles with VALID Kerala road coordinates**
        # These are actual road locations in Kochi, not water!
        kerala_road_locations = [
            (9.9674, 76.2807),  # InfoPark, Kochi
            (9.9816, 76.2999),  # Kakkanad
            (10.0014, 76.3159), # Edappally
            (9.9312, 76.2673),  # Ernakulam City
        ]
        
        for i in range(1, config.NUM_SIMULATED_CARS + 1):
            car_id = f"CAR{i:01d}"
            battery = random.randint(25, 90)
            
            # Use pre-defined road locations and add small random offset
            base_location = kerala_road_locations[(i-1) % len(kerala_road_locations)]
            lat = base_location[0] + random.uniform(-0.005, 0.005)  # ~500m radius
            lon = base_location[1] + random.uniform(-0.005, 0.005)
            
            health = random.uniform(85.0, 100.0)
            range_km = battery * 4
            
            self.db.execute_query(
                """INSERT OR REPLACE INTO vehicles
                (vehicle_id, battery_level, battery_health, current_lat, current_lon,
                range_km, status, charging, cooperation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (car_id, battery, health, lat, lon, range_km, 'IDLE', 0,
                random.uniform(5.0, 100.0))
            )
        
        logger.info(f"✅ Added {len(stations)} charging stations and {config.NUM_SIMULATED_CARS} vehicles at VALID locations.")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Ensure the database file is always created/initialized before starting the controller
    # The DatabaseManager __init__ handles this automatically.
    
    controller = ChargeUpController()
    controller.start()