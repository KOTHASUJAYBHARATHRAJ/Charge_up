
"""
ChargeUp EV System - Live Fleet Simulator
Real-time simulation engine for visual demonstration of fleet dynamics.
"""

import random
import math
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass, field

# Import logic engines
try:
    from simulation_engine import SimulationEngine
    from qlearning import q_optimizer
    from fuzzy_logic import fuzzy_engine
except ImportError:
    SimulationEngine = None
    q_optimizer = None
    fuzzy_engine = None

# Kerala Stations Data (Duplicate here or import)
KERALA_STATIONS = {
    'STN01': {'name': 'Kochi Central Hub', 'lat': 9.9312, 'lon': 76.2673, 'power_kw': 50},
    'STN02': {'name': 'Trivandrum Tech Park', 'lat': 8.5241, 'lon': 76.9366, 'power_kw': 60},
    'STN03': {'name': 'Calicut Highway', 'lat': 11.2588, 'lon': 75.7804, 'power_kw': 25},
    'STN04': {'name': 'Thrissur Mall', 'lat': 10.5276, 'lon': 76.2144, 'power_kw': 30},
    'STN05': {'name': 'Kottayam Junction', 'lat': 9.5916, 'lon': 76.5222, 'power_kw': 22},
    'STN06': {'name': 'Alappuzha Beach Road', 'lat': 9.4981, 'lon': 76.3388, 'power_kw': 35},
    'STN07': {'name': 'Kannur Smart City', 'lat': 11.8745, 'lon': 75.3704, 'power_kw': 50},
}

@dataclass
class LiveAgent:
    """Agent for real-time map simulation"""
    id: str
    lat: float
    lon: float
    battery: float
    status: str  # IDLE, MOVING, CHARGING, WAITING, SWAPPING
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    target_station: Optional[str] = None
    speed_kmh: float = 60.0 # Fast simulation speed
    color: List[int] = field(default_factory=lambda: [0, 255, 0, 200])  # RGBA
    booking_id: Optional[str] = None

class LiveFleetSimulator:
    """
    Real-time fleet simulator for visual demonstrations.
    Handles frame-by-frame updates of agent positions and states.
    """
    
    def __init__(self, num_agents: int = 20):
        self.agents: List[LiveAgent] = []
        self.simulation_time = datetime.now()
        self._initialize_agents(num_agents)
        self.active_bookings = {}
        
    def _initialize_agents(self, count: int):
        """Spawn agents around Kerala stations"""
        for i in range(count):
            # Pick a random station as "home" region
            center = random.choice(list(KERALA_STATIONS.values()))
            
            # Scatter around 50km radius
            lat = center['lat'] + random.uniform(-0.3, 0.3)
            lon = center['lon'] + random.uniform(-0.3, 0.3)
            
            self.agents.append(LiveAgent(
                id=f"AGENT_{i+1:02d}",
                lat=lat,
                lon=lon,
                battery=random.uniform(30, 90),
                status="IDLE"
            ))
            
    def update(self, dt_seconds: float = 1.0):
        """Advance simulation by dt seconds (accelerated time)"""
        # Accelerate time: 1 sec real = 5 mins sim
        self.simulation_time += timedelta(minutes=5) 
        
        for agent in self.agents:
            self._update_agent(agent, dt_seconds)
            
    def _update_agent(self, agent: LiveAgent, dt: float):
        """Update single agent logic"""
        
        # 1. IDLE -> Decide to move or charge
        if agent.status == "IDLE":
            agent.battery -= 0.05 * dt # Idle drain
            
            if agent.battery < 25:
                # Critical: Find station
                self._assign_station(agent)
            elif random.random() < 0.1: # 10% chance to move per frame
                # Random movement to another city or waypoint
                dest = random.choice(list(KERALA_STATIONS.values()))
                agent.status = "MOVING"
                agent.target_lat = dest['lat'] + random.uniform(-0.1, 0.1)
                agent.target_lon = dest['lon'] + random.uniform(-0.1, 0.1)
                agent.color = [0, 255, 0, 200] # Green
                agent.speed_kmh = 100.0 # Highway speed
                
        # 2. MOVING -> Interpolate position
        elif agent.status == "MOVING":
            if agent.target_lat is None: 
                agent.status = "IDLE"
                return
                
            dist = self._dist(agent.lat, agent.lon, agent.target_lat, agent.target_lon)
            
            if dist < 2.0: # Arrived (<2km)
                agent.lat = agent.target_lat
                agent.lon = agent.target_lon
                
                if agent.target_station:
                    agent.status = "WAITING"
                    agent.color = [255, 165, 0, 200] # Orange
                else:
                    agent.status = "IDLE"
            else:
                # Move towards target
                # Distance step = speed * time
                step_km = (agent.speed_kmh / 3600) * (dt * 300) # Accelerate movement
                
                if step_km > dist: step_km = dist
                
                ratio = step_km / dist
                agent.lat += (agent.target_lat - agent.lat) * ratio
                agent.lon += (agent.target_lon - agent.lon) * ratio
                
                # Drain battery (1% per 4km approx)
                agent.battery -= (step_km / 4.0)
                
                if agent.battery < 0: agent.battery = 0
        
        # 3. WAITING -> Enter station logic
        elif agent.status == "WAITING":
             # Simulate queue - random clearance
             if random.random() < 0.2: 
                 agent.status = "CHARGING" if random.random() > 0.3 else "SWAPPING"
                 agent.color = [0, 100, 255, 200] # Blue
                 
        # 4. CHARGING/SWAPPING -> Refill
        elif agent.status in ["CHARGING", "SWAPPING"]:
            if agent.status == "CHARGING":
                # Dynamic rate based on station power (approx 1% per 10kW per tick)
                if agent.target_station and agent.target_station in KERALA_STATIONS:
                     pwr = KERALA_STATIONS[agent.target_station].get('power_kw', 22)
                     rate = pwr / 10.0 
                else: 
                     rate = 2.0
            else:
                 rate = 15.0 # Swap is instant-ish
                 
            agent.battery += rate
            
            if agent.battery >= 95:
                agent.status = "IDLE"
                agent.target_station = None
                agent.color = [0, 255, 0, 200]

    def _assign_station(self, agent: LiveAgent):
        """Use Q-Learning to find station"""
        
        stat_keys = list(KERALA_STATIONS.keys())
        dists = [self._dist(agent.lat, agent.lon, 
                           KERALA_STATIONS[k]['lat'], 
                           KERALA_STATIONS[k]['lon']) 
                 for k in stat_keys]
        
        # Select best station (lowest distance roughly + random noise for realism)
        # Using simplified selection if optimizer unavailable
        if q_optimizer:
             res = q_optimizer.optimize_station(
                [random.randint(0,3) for _ in stat_keys], 
                agent.battery,
                8, 
                dists
            )
             target_id = stat_keys[res.selected_station]
        else:
            # Simple nearest
            target_id = stat_keys[dists.index(min(dists))]
        
        station = KERALA_STATIONS[target_id]
        
        agent.target_station = target_id
        agent.target_lat = station['lat']
        agent.target_lon = station['lon']
        agent.status = "MOVING"
        agent.color = [255, 0, 0, 255] # Red (Low bat moving to station)
        agent.speed_kmh = 80.0 # Urgent

    def _dist(self, lat1, lon1, lat2, lon2):
        """Approx Euclidean for speed (local area)"""
        # Degree to km approx at Kerala latitude
        return math.sqrt(((lat1-lat2)*111)**2 + ((lon1-lon2)*111)**2)

    def get_pydeck_data(self):
        """Return list of dicts for PyDeck"""
        return [{
            "id": a.id,
            "position": [a.lon, a.lat],
            "color": a.color,
            "battery": int(a.battery),
            "status": a.status,
            "radius": 5000 if a.status != "MOVING" else 3000
        } for a in self.agents]
