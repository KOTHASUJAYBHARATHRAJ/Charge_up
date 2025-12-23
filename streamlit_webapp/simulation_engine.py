"""
ChargeUp EV System - Enterprise Simulation Engine
Complete EV charging workflow simulation with real-time visualization.
"""

import random
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import threading

# Local imports
try:
    from database import db, DatabaseManager
    from fuzzy_logic import fuzzy_engine, quick_priority_score
    from qlearning import q_optimizer, QLearningOptimizer
except ImportError:
    # Fallback for standalone testing
    db = None
    fuzzy_engine = None
    q_optimizer = None


@dataclass
class SimulatedUser:
    """Represents a simulated EV user"""
    user_id: str
    username: str
    vehicle_id: str
    vehicle_model: str
    battery_level: float
    current_lat: float
    current_lon: float
    urgency: int
    cooperation_score: float
    points: int
    behavior_type: str  # 'normal', 'aggressive', 'patient'


@dataclass
class SimulatedBooking:
    """Represents a simulated booking"""
    id: str
    user: SimulatedUser
    station_id: str
    slot: int
    start_time: datetime
    end_time: datetime
    status: str
    priority_score: float
    qr_code: str
    fuzzy_result: Optional[Dict] = None
    ql_result: Optional[Dict] = None


@dataclass
class SimulationProgress:
    """Tracks simulation progress"""
    phase: str
    current_step: int
    total_steps: int
    message: str
    data: Dict = field(default_factory=dict)


class SimulationEngine:
    """
    Enterprise-grade simulation engine for EV charging system.
    
    Features:
    - 30 synthetic users with behavior models
    - 5-day simulation window
    - Real-time animation callbacks
    - Q-Learning optimization
    - Fuzzy priority scoring
    - Barcode workflow simulation
    - Excel export ready data
    """
    
    # Kerala stations data
    STATIONS = {
        'STN01': {'name': 'Kochi Central Hub', 'lat': 9.9312, 'lon': 76.2673, 'power_kw': 50, 'slots': 4},
        'STN02': {'name': 'Trivandrum Tech Park', 'lat': 8.5241, 'lon': 76.9366, 'power_kw': 60, 'slots': 4},
        'STN03': {'name': 'Calicut Highway', 'lat': 11.2588, 'lon': 75.7804, 'power_kw': 25, 'slots': 3},
        'STN04': {'name': 'Thrissur Mall', 'lat': 10.5276, 'lon': 76.2144, 'power_kw': 30, 'slots': 4},
        'STN05': {'name': 'Kottayam Junction', 'lat': 9.5916, 'lon': 76.5222, 'power_kw': 22, 'slots': 2},
    }
    
    # Vehicle models
    VEHICLE_MODELS = [
        {'model': 'Tata Nexon EV Max', 'capacity': 40.5, 'range': 437, 'port': 'CCS2'},
        {'model': 'MG ZS EV', 'capacity': 50.3, 'range': 461, 'port': 'CCS2'},
        {'model': 'Hyundai Kona', 'capacity': 39.2, 'range': 452, 'port': 'CCS2'},
        {'model': 'Tata Tiago EV', 'capacity': 24, 'range': 315, 'port': 'CCS2'},
        {'model': 'Mahindra XUV400', 'capacity': 39.4, 'range': 456, 'port': 'CCS2'},
    ]
    
    def __init__(self, num_users: int = 30, num_days: int = 5):
        self.num_users = num_users
        self.num_days = num_days
        
        # Simulation data
        self.users: List[SimulatedUser] = []
        self.bookings: List[SimulatedBooking] = []
        self.swap_requests: List[Dict] = []
        self.feedback_records: List[Dict] = []
        
        # Queue state per station
        self.station_queues: Dict[str, List] = {s: [] for s in self.STATIONS}
        
        # Metrics
        self.metrics = {
            'total_bookings': 0,
            'total_swaps': 0,
            'swap_success': 0,
            'swap_rejected': 0,
            'total_fuzzy_calcs': 0,
            'total_ql_iterations': 0,
            'avg_wait_time': 0,
            'peak_queue_length': 0,
        }
        
        # Animation callbacks
        self.on_progress: Optional[Callable[[SimulationProgress], None]] = None
        self.on_car_move: Optional[Callable[[Dict], None]] = None
        self.on_booking: Optional[Callable[[Dict], None]] = None
        self.on_barcode_scan: Optional[Callable[[Dict], None]] = None
        
        # Database run ID
        self.db_run_id: Optional[int] = None
        
        # Animation speed (seconds per step)
        self.animation_speed = 0.1
    
    def _report_progress(self, phase: str, step: int, total: int, message: str, data: Dict = None):
        """Report progress to callback"""
        if self.on_progress:
            self.on_progress(SimulationProgress(
                phase=phase,
                current_step=step,
                total_steps=total,
                message=message,
                data=data or {}
            ))
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _generate_qr_code(self, booking_id: str, station_id: str) -> str:
        """Generate QR code data for booking"""
        return json.dumps({
            'booking_id': booking_id,
            'station_id': station_id,
            'timestamp': datetime.now().isoformat(),
            'signature': f"SIG_{booking_id[-4:]}"
        })
    
    # ============ USER GENERATION ============
    
    def generate_users(self) -> List[SimulatedUser]:
        """Generate synthetic users with varied behaviors"""
        self._report_progress("SETUP", 0, self.num_users, "Generating users...")
        
        users = []
        behavior_types = ['normal'] * 20 + ['aggressive'] * 5 + ['patient'] * 5
        
        for i in range(self.num_users):
            # Random position in Kerala region
            lat = random.uniform(8.2, 12.0)
            lon = random.uniform(74.8, 77.5)
            
            vehicle = random.choice(self.VEHICLE_MODELS)
            behavior = behavior_types[i % len(behavior_types)]
            
            user = SimulatedUser(
                user_id=f"SIM_USER_{i+1:03d}",
                username=f"user_{i+1}",
                vehicle_id=f"KL-{random.randint(1,15):02d}-{chr(65+i%26)}{chr(65+(i+1)%26)}-{random.randint(1000,9999)}",
                vehicle_model=vehicle['model'],
                battery_level=random.randint(15, 95),
                current_lat=lat,
                current_lon=lon,
                urgency=random.randint(1, 10),
                cooperation_score=random.uniform(30, 90),
                points=random.randint(50, 500),
                behavior_type=behavior
            )
            users.append(user)
            
            self._report_progress("SETUP", i+1, self.num_users, f"Created user {user.username}")
        
        self.users = users
        return users
    
    # ============ STATION SELECTION (Q-LEARNING) ============
    
    def select_station_ql(self, user: SimulatedUser) -> tuple:
        """Use Q-Learning to select optimal station"""
        station_ids = list(self.STATIONS.keys())
        
        # Get queue lengths
        queue_lengths = [len(self.station_queues[s]) for s in station_ids]
        
        # Calculate distances
        distances = [
            self._calculate_distance(
                user.current_lat, user.current_lon,
                self.STATIONS[s]['lat'], self.STATIONS[s]['lon']
            )
            for s in station_ids
        ]
        
        # Get charging powers
        powers = [self.STATIONS[s]['power_kw'] for s in station_ids]
        
        # Q-Learning optimization
        if q_optimizer:
            result = q_optimizer.optimize_station(
                queue_lengths=queue_lengths,
                battery=user.battery_level,
                urgency=user.urgency,
                distances=distances,
                charging_powers=powers,
                explore=True
            )
            
            selected_idx = result.selected_station
            self.metrics['total_ql_iterations'] += 1
            
            return station_ids[selected_idx], {
                'state': result.state_repr,
                'q_values': result.q_values,
                'reward': result.reward,
                'was_exploration': result.was_exploration
            }
        else:
            # Fallback: select station with shortest queue
            selected_idx = queue_lengths.index(min(queue_lengths))
            return station_ids[selected_idx], None
    
    # ============ PRIORITY SCORING (FUZZY) ============
    
    def calculate_priority_fuzzy(self, user: SimulatedUser, station_id: str, 
                                  wait_mins: int = 0) -> tuple:
        """Calculate priority using Fuzzy Logic"""
        station = self.STATIONS[station_id]
        distance = self._calculate_distance(
            user.current_lat, user.current_lon,
            station['lat'], station['lon']
        )
        
        if fuzzy_engine:
            result = fuzzy_engine.calculate_priority(
                battery=user.battery_level,
                distance_km=distance,
                urgency=user.urgency,
                wait_mins=wait_mins,
                cooperation_bonus=user.cooperation_score
            )
            
            self.metrics['total_fuzzy_calcs'] += 1
            
            return result.defuzzified_value, {
                'battery_mf': result.battery,
                'distance_mf': result.distance,
                'urgency_mf': result.urgency,
                'wait_mf': result.wait_time,
                'priority': result.defuzzified_value
            }
        else:
            # Fallback: simple weighted calculation
            score = (100 - user.battery_level) * 0.4 + user.urgency * 5 + (user.cooperation_score * 0.1)
            return min(100, max(0, score)), None
    
    # ============ CAR ANIMATION ============
    
    def animate_car_to_station(self, user: SimulatedUser, station_id: str, 
                                steps: int = 10) -> None:
        """Animate car movement from user position to station"""
        station = self.STATIONS[station_id]
        
        start_lat, start_lon = user.current_lat, user.current_lon
        end_lat, end_lon = station['lat'], station['lon']
        
        for step in range(steps + 1):
            progress = step / steps
            
            # Linear interpolation
            current_lat = start_lat + (end_lat - start_lat) * progress
            current_lon = start_lon + (end_lon - start_lon) * progress
            
            # Update user position
            user.current_lat = current_lat
            user.current_lon = current_lon
            
            # Battery drain during travel
            if step > 0:
                user.battery_level = max(5, user.battery_level - 0.5)
            
            # Callback for animation
            if self.on_car_move:
                self.on_car_move({
                    'user_id': user.user_id,
                    'vehicle_id': user.vehicle_id,
                    'lat': current_lat,
                    'lon': current_lon,
                    'battery': user.battery_level,
                    'progress': progress * 100,
                    'station_id': station_id,
                    'step': step,
                    'total_steps': steps
                })
            
            time.sleep(self.animation_speed)
    
    # ============ BARCODE WORKFLOW ============
    
    def simulate_barcode_scan(self, booking: SimulatedBooking) -> Dict:
        """Simulate barcode/QR scan at station"""
        scan_data = {
            'booking_id': booking.id,
            'station_id': booking.station_id,
            'slot': booking.slot,
            'qr_code': booking.qr_code,
            'scan_time': datetime.now().isoformat(),
            'verified': True,
            'message': f"Vehicle {booking.user.vehicle_id} verified at Slot {booking.slot}"
        }
        
        if self.on_barcode_scan:
            self.on_barcode_scan(scan_data)
        
        return scan_data
    
    # ============ BOOKING SIMULATION ============
    
    def create_booking(self, user: SimulatedUser, day_offset: int = 0) -> SimulatedBooking:
        """Create a single booking with full workflow"""
        
        # 1. Q-Learning station selection
        station_id, ql_result = self.select_station_ql(user)
        
        # 2. Fuzzy priority calculation
        wait_time = len(self.station_queues[station_id]) * 15  # 15 min per car
        priority_score, fuzzy_result = self.calculate_priority_fuzzy(user, station_id, wait_time)
        
        # 3. Generate booking
        base_time = datetime.now() - timedelta(days=self.num_days - day_offset - 1)
        
        # Peak hour bias
        if random.random() < 0.6:
            hour = random.choice([8, 9, 10, 17, 18, 19])  # Peak hours
        else:
            hour = random.randint(6, 22)
        
        start_time = base_time.replace(hour=hour, minute=random.choice([0, 15, 30, 45]))
        duration = random.choice([30, 45, 60])
        
        # Find available slot
        station_slots = self.STATIONS[station_id]['slots']
        used_slots = [b.slot for b in self.bookings if b.station_id == station_id and b.status == 'confirmed']
        available_slots = [i for i in range(1, station_slots + 1) if i not in used_slots]
        slot = random.choice(available_slots) if available_slots else random.randint(1, station_slots)
        
        booking_id = f"SIM_BK_{len(self.bookings):04d}"
        
        booking = SimulatedBooking(
            id=booking_id,
            user=user,
            station_id=station_id,
            slot=slot,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration),
            status='confirmed',
            priority_score=priority_score,
            qr_code=self._generate_qr_code(booking_id, station_id),
            fuzzy_result=fuzzy_result,
            ql_result=ql_result
        )
        
        self.bookings.append(booking)
        self.station_queues[station_id].append(booking)
        self.metrics['total_bookings'] += 1
        
        # Update peak queue metric
        queue_len = len(self.station_queues[station_id])
        if queue_len > self.metrics['peak_queue_length']:
            self.metrics['peak_queue_length'] = queue_len
        
        # Callback
        if self.on_booking:
            self.on_booking({
                'booking_id': booking_id,
                'user': user.username,
                'station': self.STATIONS[station_id]['name'],
                'slot': slot,
                'priority': priority_score,
                'time': start_time.strftime('%H:%M')
            })
        
        # Log to database
        if db and self.db_run_id and fuzzy_result:
            db.log_fuzzy_calculation(self.db_run_id, {
                'vehicle_id': user.vehicle_id,
                'battery_level': int(user.battery_level),
                'urgency': user.urgency,
                'distance_km': self._calculate_distance(
                    user.current_lat, user.current_lon,
                    self.STATIONS[station_id]['lat'], self.STATIONS[station_id]['lon']
                ),
                'wait_time_mins': wait_time,
                'priority_score': priority_score,
                'membership_values': fuzzy_result
            })
        
        if db and self.db_run_id and ql_result:
            db.log_qlearning_state(self.db_run_id, {
                'state': ql_result['state'],
                'action': list(self.STATIONS.keys()).index(station_id),
                'reward': ql_result['reward'],
                'q_value': max(ql_result['q_values']),
                'iteration': self.metrics['total_ql_iterations']
            })
        
        return booking
    
    # ============ SWAP SIMULATION ============
    
    def simulate_swap_request(self, booking1: SimulatedBooking, 
                               booking2: SimulatedBooking) -> Dict:
        """Simulate a swap request between two bookings"""
        
        # Determine if swap should be accepted (based on priority difference)
        priority_diff = booking1.priority_score - booking2.priority_score
        
        # Higher priority user requests from lower priority
        if priority_diff > 10:
            # Calculate acceptance probability
            accept_prob = min(0.9, 0.5 + (priority_diff / 100))
            accepted = random.random() < accept_prob
        else:
            accepted = random.random() < 0.3  # Low acceptance for small differences
        
        swap = {
            'id': f"SWAP_{len(self.swap_requests):04d}",
            'from_user': booking1.user.user_id,
            'from_vehicle': booking1.user.vehicle_id,
            'to_user': booking2.user.user_id,
            'to_vehicle': booking2.user.vehicle_id,
            'from_priority': booking1.priority_score,
            'to_priority': booking2.priority_score,
            'points_offered': random.randint(10, 50),
            'status': 'accepted' if accepted else 'rejected',
            'timestamp': datetime.now()
        }
        
        self.swap_requests.append(swap)
        self.metrics['total_swaps'] += 1
        
        if accepted:
            self.metrics['swap_success'] += 1
            # Transfer points
            booking1.user.points -= swap['points_offered']
            booking2.user.points += swap['points_offered']
        else:
            self.metrics['swap_rejected'] += 1
        
        return swap
    
    # ============ FEEDBACK SIMULATION ============
    
    def generate_feedback(self, booking: SimulatedBooking) -> Dict:
        """Generate feedback for completed booking"""
        
        # Rating weighted towards positive
        rating = random.choices([5, 4, 3, 2, 1], weights=[50, 30, 12, 5, 3])[0]
        
        comments = {
            5: ["Excellent service!", "Very smooth experience!", "Highly recommended!"],
            4: ["Good charging speed.", "Nice station.", "Will come again."],
            3: ["Average experience.", "Queue was a bit long.", "Okay."],
            2: ["Slow charging.", "Not well maintained.", "Could be better."],
            1: ["Very disappointed.", "Long wait time.", "Poor service."]
        }
        
        feedback = {
            'booking_id': booking.id,
            'station_id': booking.station_id,
            'user_id': booking.user.user_id,
            'rating': rating,
            'comment': random.choice(comments[rating]),
            'timestamp': booking.end_time
        }
        
        self.feedback_records.append(feedback)
        return feedback
    
    # ============ MAIN SIMULATION ============
    
    def run_full_simulation(self, animate: bool = False) -> Dict:
        """
        Run complete simulation workflow.
        
        Args:
            animate: Whether to include car movement animation
        
        Returns:
            Complete simulation results
        """
        start_time = time.time()
        
        # Create database run
        if db:
            self.db_run_id = db.create_simulation_run(
                f"Enterprise_Sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                self.num_users
            )
        
        # Phase 1: Generate users
        self._report_progress("PHASE_1", 0, 1, "Generating synthetic users...")
        self.generate_users()
        
        # Phase 2: Simulate each day
        total_bookings_target = self.num_users * self.num_days * 2  # ~2 bookings per user per day
        booking_count = 0
        
        for day in range(self.num_days):
            self._report_progress("PHASE_2", day, self.num_days, f"Simulating Day {day+1}...")
            
            # Daily bookings (30-50 per day)
            daily_bookings = random.randint(30, 50)
            
            for i in range(daily_bookings):
                user = random.choice(self.users)
                
                # Animate car movement for first few bookings
                if animate and booking_count < 5:
                    station_id, _ = self.select_station_ql(user)
                    self.animate_car_to_station(user, station_id, steps=8)
                
                booking = self.create_booking(user, day)
                booking_count += 1
                
                # Simulate barcode scan for animated bookings
                if animate and booking_count <= 5:
                    self.simulate_barcode_scan(booking)
                
                self._report_progress("PHASE_2", 
                                     day * daily_bookings + i, 
                                     self.num_days * daily_bookings,
                                     f"Created booking {booking.id}")
        
        # Phase 3: Simulate swap requests
        self._report_progress("PHASE_3", 0, 25, "Simulating swap requests...")
        
        confirmed_bookings = [b for b in self.bookings if b.status == 'confirmed']
        for i in range(min(25, len(confirmed_bookings) // 2)):
            b1, b2 = random.sample(confirmed_bookings, 2)
            self.simulate_swap_request(b1, b2)
            self._report_progress("PHASE_3", i+1, 25, f"Processed swap request {i+1}")
        
        # Phase 4: Generate feedback
        self._report_progress("PHASE_4", 0, len(self.bookings), "Generating feedback...")
        
        for i, booking in enumerate(self.bookings):
            if random.random() < 0.7:  # 70% give feedback
                self.generate_feedback(booking)
            self._report_progress("PHASE_4", i+1, len(self.bookings), f"Feedback for booking {booking.id}")
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        
        self.metrics['avg_wait_time'] = sum(
            len(self.station_queues[s]) * 15 for s in self.STATIONS
        ) / len(self.STATIONS)
        
        swap_success_rate = (self.metrics['swap_success'] / self.metrics['total_swaps'] * 100) if self.metrics['total_swaps'] > 0 else 0
        
        # Update database
        if db and self.db_run_id:
            db.update_simulation_run(self.db_run_id, {
                'total_bookings': self.metrics['total_bookings'],
                'total_swaps': self.metrics['total_swaps'],
                'swap_success_rate': swap_success_rate,
                'avg_wait_time': self.metrics['avg_wait_time'],
                'qlearning_iterations': self.metrics['total_ql_iterations'],
                'fuzzy_calculations': self.metrics['total_fuzzy_calcs']
            })
        
        self._report_progress("COMPLETE", 1, 1, "Simulation complete!")
        
        return {
            'run_id': self.db_run_id,
            'users': len(self.users),
            'bookings': len(self.bookings),
            'swaps': len(self.swap_requests),
            'feedback': len(self.feedback_records),
            'metrics': self.metrics,
            'swap_success_rate': swap_success_rate,
            'elapsed_seconds': round(elapsed_time, 2),
            'qlearning_summary': q_optimizer.get_q_table_summary() if q_optimizer else {},
            'fuzzy_calculations': self.metrics['total_fuzzy_calcs']
        }
    
    # ============ EXPORT METHODS ============
    
    def get_bookings_data(self) -> List[Dict]:
        """Get bookings data for export"""
        return [{
            'id': b.id,
            'user_id': b.user.user_id,
            'vehicle_id': b.user.vehicle_id,
            'vehicle_model': b.user.vehicle_model,
            'station_id': b.station_id,
            'station_name': self.STATIONS[b.station_id]['name'],
            'slot': b.slot,
            'start_time': b.start_time.isoformat(),
            'end_time': b.end_time.isoformat(),
            'status': b.status,
            'priority_score': round(b.priority_score, 2),
            'battery_at_booking': b.user.battery_level,
            'urgency': b.user.urgency
        } for b in self.bookings]
    
    def get_swaps_data(self) -> List[Dict]:
        """Get swap requests data for export"""
        return [{
            'id': s['id'],
            'from_user': s['from_user'],
            'to_user': s['to_user'],
            'from_priority': round(s['from_priority'], 2),
            'to_priority': round(s['to_priority'], 2),
            'points_offered': s['points_offered'],
            'status': s['status'],
            'timestamp': s['timestamp'].isoformat()
        } for s in self.swap_requests]
    
    def get_fuzzy_data(self) -> List[Dict]:
        """Get fuzzy calculation data for export"""
        return [{
            'booking_id': b.id,
            'battery': b.user.battery_level,
            'urgency': b.user.urgency,
            'priority_score': round(b.priority_score, 2),
            'fuzzy_battery_critical': round(b.fuzzy_result.get('battery_mf', {}).get('critical', 0), 3) if b.fuzzy_result else 0,
            'fuzzy_battery_low': round(b.fuzzy_result.get('battery_mf', {}).get('low', 0), 3) if b.fuzzy_result else 0,
            'fuzzy_urgency_high': round(b.fuzzy_result.get('urgency_mf', {}).get('high', 0), 3) if b.fuzzy_result else 0,
        } for b in self.bookings if b.fuzzy_result]
    
    def get_qlearning_data(self) -> Dict:
        """Get Q-Learning data for export"""
        if q_optimizer:
            return {
                'q_table_summary': q_optimizer.get_q_table_summary(),
                'convergence_data': q_optimizer.get_convergence_data(),
                'heatmap_data': q_optimizer.get_q_table_heatmap_data()
            }
        return {}


# Factory function
def create_simulation(num_users: int = 30, num_days: int = 5) -> SimulationEngine:
    """Create a new simulation engine instance"""
    return SimulationEngine(num_users=num_users, num_days=num_days)
