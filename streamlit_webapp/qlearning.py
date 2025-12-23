"""
ChargeUp EV System - Q-Learning Station Optimizer
Reinforcement learning for optimal charging station assignment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import random


@dataclass
class QLearningState:
    """Represents a discretized state for Q-Learning"""
    queue_lengths: Tuple[int, ...]  # Queue length per station (discretized)
    battery_bin: int                 # Battery level bin (0-4)
    urgency_bin: int                 # Urgency bin (0-2)
    distance_bins: Tuple[int, ...]   # Distance to each station (discretized)
    
    def to_string(self) -> str:
        """Convert state to hashable string"""
        return f"Q{self.queue_lengths}_B{self.battery_bin}_U{self.urgency_bin}_D{self.distance_bins}"


@dataclass
class QLearningResult:
    """Result of Q-Learning optimization"""
    selected_station: int
    state_repr: str
    q_values: List[float]
    reward: float
    was_exploration: bool


class QLearningOptimizer:
    """
    Q-Learning based station selection optimizer.
    
    Features:
    - Discrete state space with queue lengths, battery, urgency
    - Epsilon-greedy exploration
    - Experience replay (simplified)
    - Convergence tracking
    """
    
    def __init__(self, num_stations: int = 5, 
                 alpha: float = 0.1,      # Learning rate
                 gamma: float = 0.95,     # Discount factor
                 epsilon: float = 0.15,   # Exploration rate
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01):
        
        self.num_stations = num_stations
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state -> action values
        self.q_table: Dict[str, np.ndarray] = {}
        
        # Training statistics
        self.total_iterations = 0
        self.total_reward = 0.0
        self.reward_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.convergence_history: List[float] = []
        
        # Experience buffer for visualization
        self.experience_buffer: List[Dict] = []
    
    def _discretize_queue(self, length: int) -> int:
        """Discretize queue length into bins"""
        if length == 0:
            return 0
        elif length <= 2:
            return 1
        elif length <= 5:
            return 2
        elif length <= 10:
            return 3
        else:
            return 4
    
    def _discretize_battery(self, battery: float) -> int:
        """Discretize battery level"""
        if battery <= 15:
            return 0  # Critical
        elif battery <= 30:
            return 1  # Low
        elif battery <= 50:
            return 2  # Medium
        elif battery <= 75:
            return 3  # Good
        else:
            return 4  # High
    
    def _discretize_urgency(self, urgency: float) -> int:
        """Discretize urgency level"""
        if urgency <= 3:
            return 0  # Low
        elif urgency <= 6:
            return 1  # Medium
        else:
            return 2  # High
    
    def _discretize_distance(self, distance_km: float) -> int:
        """Discretize distance to station"""
        if distance_km <= 5:
            return 0  # Very near
        elif distance_km <= 15:
            return 1  # Near
        elif distance_km <= 30:
            return 2  # Moderate
        elif distance_km <= 50:
            return 3  # Far
        else:
            return 4  # Very far
    
    def get_state(self, queue_lengths: List[int], battery: float, 
                  urgency: float, distances: List[float]) -> QLearningState:
        """Create a discretized state from raw inputs"""
        return QLearningState(
            queue_lengths=tuple(self._discretize_queue(q) for q in queue_lengths),
            battery_bin=self._discretize_battery(battery),
            urgency_bin=self._discretize_urgency(urgency),
            distance_bins=tuple(self._discretize_distance(d) for d in distances)
        )
    
    def _get_q_values(self, state: str) -> np.ndarray:
        """Get Q-values for a state, initializing if needed"""
        if state not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[state] = np.random.uniform(-0.1, 0.1, self.num_stations)
        return self.q_table[state]
    
    def select_action(self, state: QLearningState, 
                      explore: bool = True) -> Tuple[int, bool]:
        """
        Select action using epsilon-greedy policy.
        
        Returns:
            (action, was_exploration)
        """
        state_str = state.to_string()
        q_values = self._get_q_values(state_str)
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            action = random.randint(0, self.num_stations - 1)
            return action, True
        
        # Greedy selection
        action = int(np.argmax(q_values))
        return action, False
    
    def calculate_reward(self, 
                         queue_length: int,
                         distance_km: float,
                         battery: float,
                         wait_time_mins: float = 0,
                         charging_power_kw: float = 50) -> float:
        """
        Calculate reward for selecting a station.
        
        Reward components:
        - Queue penalty: Longer queues = negative reward
        - Distance penalty: Farther stations = negative (battery cost)
        - Battery bonus: Critical battery + short queue = positive
        - Power bonus: Higher power = faster charging
        - Wait time penalty: Long waits = negative
        """
        reward = 0.0
        
        # 1. Queue penalty (-5 per car in queue)
        queue_penalty = -queue_length * 5
        reward += queue_penalty
        
        # 2. Distance penalty (-0.5 per km for low battery, -0.2 otherwise)
        if battery < 30:
            distance_penalty = -distance_km * 0.5
        else:
            distance_penalty = -distance_km * 0.2
        reward += distance_penalty
        
        # 3. Critical battery bonus (encourage nearby short-queue stations)
        if battery < 20:
            if queue_length <= 2 and distance_km <= 15:
                reward += 30  # Big bonus for quick access
            elif queue_length <= 5:
                reward += 15
        
        # 4. Power bonus (higher charging power = faster)
        power_bonus = (charging_power_kw - 30) * 0.3  # Baseline 30kW
        reward += power_bonus
        
        # 5. Wait time penalty
        wait_penalty = -wait_time_mins * 0.5
        reward += wait_penalty
        
        # 6. Fairness bonus (balance load across stations)
        # This would require global queue info, simplified here
        
        return reward
    
    def update(self, state: QLearningState, action: int, 
               reward: float, next_state: QLearningState):
        """
        Update Q-value using Q-Learning update rule:
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        """
        state_str = state.to_string()
        next_state_str = next_state.to_string()
        
        current_q = self._get_q_values(state_str)[action]
        next_max_q = np.max(self._get_q_values(next_state_str))
        
        # Q-Learning update
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_str][action] = new_q
        
        # Update statistics
        self.total_iterations += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        self.epsilon_history.append(self.epsilon)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Track convergence (moving average of reward)
        if len(self.reward_history) >= 10:
            avg_reward = np.mean(self.reward_history[-10:])
            self.convergence_history.append(avg_reward)
        
        # Store experience
        self.experience_buffer.append({
            'state': state_str,
            'action': action,
            'reward': reward,
            'q_value': new_q,
            'iteration': self.total_iterations
        })
        
        # Limit buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]
    
    def optimize_station(self, 
                         queue_lengths: List[int],
                         battery: float,
                         urgency: float,
                         distances: List[float],
                         charging_powers: List[float] = None,
                         explore: bool = True) -> QLearningResult:
        """
        Main interface: Select optimal station and return result.
        
        Args:
            queue_lengths: Current queue at each station
            battery: Vehicle battery level (0-100)
            urgency: User urgency (0-10)
            distances: Distance to each station in km
            charging_powers: Power of each station in kW
            explore: Whether to use epsilon-greedy exploration
        
        Returns:
            QLearningResult with selection and details
        """
        if charging_powers is None:
            charging_powers = [50.0] * self.num_stations
        
        # Create state
        state = self.get_state(queue_lengths, battery, urgency, distances)
        state_str = state.to_string()
        
        # Select action
        action, was_exploration = self.select_action(state, explore)
        
        # Calculate reward for selected action
        reward = self.calculate_reward(
            queue_length=queue_lengths[action],
            distance_km=distances[action],
            battery=battery,
            charging_power_kw=charging_powers[action]
        )
        
        # Get current Q-values for visualization
        q_values = self._get_q_values(state_str).tolist()
        
        return QLearningResult(
            selected_station=action,
            state_repr=state_str,
            q_values=q_values,
            reward=reward,
            was_exploration=was_exploration
        )
    
    def get_q_table_summary(self) -> Dict:
        """Get summary of Q-table for visualization"""
        if not self.q_table:
            return {'states': 0, 'avg_q': 0, 'max_q': 0, 'min_q': 0}
        
        all_q = np.array([v for v in self.q_table.values()])
        return {
            'states': len(self.q_table),
            'avg_q': float(np.mean(all_q)),
            'max_q': float(np.max(all_q)),
            'min_q': float(np.min(all_q)),
            'std_q': float(np.std(all_q))
        }
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting"""
        return {
            'reward_history': self.reward_history[-100:],  # Last 100
            'convergence': self.convergence_history[-50:],
            'epsilon_history': self.epsilon_history[-100:],
            'total_iterations': self.total_iterations,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0
        }
    
    def get_q_table_heatmap_data(self) -> Dict:
        """Get Q-table data formatted for heatmap visualization"""
        if not self.q_table:
            return {'states': [], 'values': []}
        
        # Get top states by max Q-value
        sorted_states = sorted(
            self.q_table.items(),
            key=lambda x: np.max(x[1]),
            reverse=True
        )[:20]  # Top 20 states
        
        return {
            'states': [s[0][:15] for s in sorted_states],  # Truncate state string
            'values': [s[1].tolist() for s in sorted_states]
        }
    
    def reset(self):
        """Reset the optimizer"""
        self.q_table = {}
        self.total_iterations = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.epsilon_history = []
        self.convergence_history = []
        self.experience_buffer = []
        self.epsilon = 0.15


# Singleton instance
q_optimizer = QLearningOptimizer(num_stations=7)


def quick_station_select(queue_lengths: List[int], battery: float, 
                         distances: List[float]) -> int:
    """Quick helper for station selection"""
    result = q_optimizer.optimize_station(queue_lengths, battery, 5, distances)
    return result.selected_station
