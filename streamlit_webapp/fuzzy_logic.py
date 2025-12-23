"""
ChargeUp EV System - Mamdani Fuzzy Logic Engine
Enterprise-grade fuzzy inference system for priority scoring.
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json


@dataclass
class FuzzyMembershipResult:
    """Stores membership function evaluation results"""
    battery: Dict[str, float]
    distance: Dict[str, float]
    urgency: Dict[str, float]
    wait_time: Dict[str, float]
    priority_output: np.ndarray
    defuzzified_value: float


class MamdaniFuzzyEngine:
    """
    Enterprise Mamdani-type Fuzzy Inference System for EV Priority Scoring.
    
    Features:
    - 4 Input Variables: Battery, Distance, Urgency, Wait Time
    - Triangular & Trapezoidal Membership Functions
    - 15+ Fuzzy Rules
    - Centroid Defuzzification
    - Full Membership Visualization Support
    """
    
    def __init__(self):
        # Define universes of discourse
        self.battery_range = np.arange(0, 101, 1)      # 0-100%
        self.distance_range = np.arange(0, 101, 1)     # 0-100 km
        self.urgency_range = np.arange(0, 11, 1)       # 0-10 scale
        self.wait_range = np.arange(0, 61, 1)          # 0-60 mins
        self.priority_range = np.arange(0, 101, 1)     # 0-100 priority
    
    # ============ MEMBERSHIP FUNCTIONS ============
    
    def battery_membership(self, level: float) -> Dict[str, float]:
        """
        Battery level membership functions.
        Critical: 0-20%, Low: 10-40%, Medium: 30-70%, High: 60-100%
        """
        mf = {}
        
        # Critical: Trapezoidal [0, 0, 10, 25]
        if level <= 10:
            mf['critical'] = 1.0
        elif level <= 25:
            mf['critical'] = (25 - level) / 15
        else:
            mf['critical'] = 0.0
        
        # Low: Triangular [10, 25, 45]
        if level <= 10:
            mf['low'] = 0.0
        elif level <= 25:
            mf['low'] = (level - 10) / 15
        elif level <= 45:
            mf['low'] = (45 - level) / 20
        else:
            mf['low'] = 0.0
        
        # Medium: Triangular [35, 55, 75]
        if level <= 35:
            mf['medium'] = 0.0
        elif level <= 55:
            mf['medium'] = (level - 35) / 20
        elif level <= 75:
            mf['medium'] = (75 - level) / 20
        else:
            mf['medium'] = 0.0
        
        # High: Trapezoidal [65, 85, 100, 100]
        if level <= 65:
            mf['high'] = 0.0
        elif level <= 85:
            mf['high'] = (level - 65) / 20
        else:
            mf['high'] = 1.0
        
        return mf
    
    def distance_membership(self, dist_km: float) -> Dict[str, float]:
        """
        Distance to station membership functions.
        Very Near: 0-5km, Near: 2-15km, Moderate: 10-35km, Far: 25-60km, Very Far: 50+km
        """
        mf = {}
        
        # Very Near: Trapezoidal [0, 0, 3, 8]
        if dist_km <= 3:
            mf['very_near'] = 1.0
        elif dist_km <= 8:
            mf['very_near'] = (8 - dist_km) / 5
        else:
            mf['very_near'] = 0.0
        
        # Near: Triangular [5, 12, 22]
        if dist_km <= 5:
            mf['near'] = 0.0
        elif dist_km <= 12:
            mf['near'] = (dist_km - 5) / 7
        elif dist_km <= 22:
            mf['near'] = (22 - dist_km) / 10
        else:
            mf['near'] = 0.0
        
        # Moderate: Triangular [18, 30, 45]
        if dist_km <= 18:
            mf['moderate'] = 0.0
        elif dist_km <= 30:
            mf['moderate'] = (dist_km - 18) / 12
        elif dist_km <= 45:
            mf['moderate'] = (45 - dist_km) / 15
        else:
            mf['moderate'] = 0.0
        
        # Far: Triangular [35, 55, 80]
        if dist_km <= 35:
            mf['far'] = 0.0
        elif dist_km <= 55:
            mf['far'] = (dist_km - 35) / 20
        elif dist_km <= 80:
            mf['far'] = (80 - dist_km) / 25
        else:
            mf['far'] = 0.0
        
        # Very Far: Trapezoidal [60, 90, 100, 100]
        if dist_km <= 60:
            mf['very_far'] = 0.0
        elif dist_km <= 90:
            mf['very_far'] = (dist_km - 60) / 30
        else:
            mf['very_far'] = 1.0
        
        return mf
    
    def urgency_membership(self, urgency: float) -> Dict[str, float]:
        """
        User urgency level membership functions.
        Low: 0-4, Medium: 3-7, High: 6-10
        """
        mf = {}
        
        # Low: Trapezoidal [0, 0, 2, 5]
        if urgency <= 2:
            mf['low'] = 1.0
        elif urgency <= 5:
            mf['low'] = (5 - urgency) / 3
        else:
            mf['low'] = 0.0
        
        # Medium: Triangular [3, 5, 8]
        if urgency <= 3:
            mf['medium'] = 0.0
        elif urgency <= 5:
            mf['medium'] = (urgency - 3) / 2
        elif urgency <= 8:
            mf['medium'] = (8 - urgency) / 3
        else:
            mf['medium'] = 0.0
        
        # High: Trapezoidal [6, 8, 10, 10]
        if urgency <= 6:
            mf['high'] = 0.0
        elif urgency <= 8:
            mf['high'] = (urgency - 6) / 2
        else:
            mf['high'] = 1.0
        
        return mf
    
    def wait_time_membership(self, wait_mins: float) -> Dict[str, float]:
        """
        Queue wait time membership functions.
        Short: 0-10min, Medium: 5-25min, Long: 20-45min, Very Long: 40+min
        """
        mf = {}
        
        # Short: Trapezoidal [0, 0, 5, 15]
        if wait_mins <= 5:
            mf['short'] = 1.0
        elif wait_mins <= 15:
            mf['short'] = (15 - wait_mins) / 10
        else:
            mf['short'] = 0.0
        
        # Medium: Triangular [10, 20, 35]
        if wait_mins <= 10:
            mf['medium'] = 0.0
        elif wait_mins <= 20:
            mf['medium'] = (wait_mins - 10) / 10
        elif wait_mins <= 35:
            mf['medium'] = (35 - wait_mins) / 15
        else:
            mf['medium'] = 0.0
        
        # Long: Triangular [25, 40, 55]
        if wait_mins <= 25:
            mf['long'] = 0.0
        elif wait_mins <= 40:
            mf['long'] = (wait_mins - 25) / 15
        elif wait_mins <= 55:
            mf['long'] = (55 - wait_mins) / 15
        else:
            mf['long'] = 0.0
        
        # Very Long: Trapezoidal [45, 55, 60, 60]
        if wait_mins <= 45:
            mf['very_long'] = 0.0
        elif wait_mins <= 55:
            mf['very_long'] = (wait_mins - 45) / 10
        else:
            mf['very_long'] = 1.0
        
        return mf
    
    def _priority_output_mf(self, category: str) -> np.ndarray:
        """Output membership functions for priority score"""
        mf = np.zeros(len(self.priority_range))
        
        if category == 'very_low':
            for i, p in enumerate(self.priority_range):
                if p <= 15:
                    mf[i] = 1.0
                elif p <= 30:
                    mf[i] = (30 - p) / 15
        
        elif category == 'low':
            for i, p in enumerate(self.priority_range):
                if 20 <= p <= 35:
                    mf[i] = (p - 20) / 15
                elif 35 < p <= 50:
                    mf[i] = (50 - p) / 15
        
        elif category == 'medium':
            for i, p in enumerate(self.priority_range):
                if 40 <= p <= 55:
                    mf[i] = (p - 40) / 15
                elif 55 < p <= 70:
                    mf[i] = (70 - p) / 15
        
        elif category == 'high':
            for i, p in enumerate(self.priority_range):
                if 60 <= p <= 75:
                    mf[i] = (p - 60) / 15
                elif 75 < p <= 90:
                    mf[i] = (90 - p) / 15
        
        elif category == 'very_high':
            for i, p in enumerate(self.priority_range):
                if p <= 80:
                    mf[i] = 0.0
                elif p <= 90:
                    mf[i] = (p - 80) / 10
                else:
                    mf[i] = 1.0
        
        return mf
    
    # ============ FUZZY RULES ============
    
    def apply_rules(self, battery_mf: Dict, distance_mf: Dict, 
                    urgency_mf: Dict, wait_mf: Dict) -> np.ndarray:
        """
        Apply fuzzy rules and aggregate outputs.
        Returns aggregated output fuzzy set.
        """
        output = np.zeros(len(self.priority_range))
        
        # Rule 1: Critical battery + High urgency = Very High Priority
        r1 = min(battery_mf.get('critical', 0), urgency_mf.get('high', 0))
        output = np.fmax(output, r1 * self._priority_output_mf('very_high'))
        
        # Rule 2: Critical battery + Medium urgency = High Priority
        r2 = min(battery_mf.get('critical', 0), urgency_mf.get('medium', 0))
        output = np.fmax(output, r2 * self._priority_output_mf('high'))
        
        # Rule 3: Critical battery + Very near = Very High Priority
        r3 = min(battery_mf.get('critical', 0), distance_mf.get('very_near', 0))
        output = np.fmax(output, r3 * self._priority_output_mf('very_high'))
        
        # Rule 4: Low battery + High urgency = High Priority
        r4 = min(battery_mf.get('low', 0), urgency_mf.get('high', 0))
        output = np.fmax(output, r4 * self._priority_output_mf('high'))
        
        # Rule 5: Low battery + Near + Very long wait = Very High Priority
        r5 = min(battery_mf.get('low', 0), distance_mf.get('near', 0), wait_mf.get('very_long', 0))
        output = np.fmax(output, r5 * self._priority_output_mf('very_high'))
        
        # Rule 6: Low battery + Medium urgency = Medium Priority
        r6 = min(battery_mf.get('low', 0), urgency_mf.get('medium', 0))
        output = np.fmax(output, r6 * self._priority_output_mf('medium'))
        
        # Rule 7: Medium battery + High urgency = Medium Priority
        r7 = min(battery_mf.get('medium', 0), urgency_mf.get('high', 0))
        output = np.fmax(output, r7 * self._priority_output_mf('medium'))
        
        # Rule 8: Medium battery + Low urgency = Low Priority
        r8 = min(battery_mf.get('medium', 0), urgency_mf.get('low', 0))
        output = np.fmax(output, r8 * self._priority_output_mf('low'))
        
        # Rule 9: High battery = Very Low Priority
        r9 = battery_mf.get('high', 0)
        output = np.fmax(output, r9 * self._priority_output_mf('very_low'))
        
        # Rule 10: Very long wait + Near = High Priority (fairness)
        r10 = min(wait_mf.get('very_long', 0), distance_mf.get('near', 0))
        output = np.fmax(output, r10 * self._priority_output_mf('high'))
        
        # Rule 11: Long wait + Low battery = High Priority
        r11 = min(wait_mf.get('long', 0), battery_mf.get('low', 0))
        output = np.fmax(output, r11 * self._priority_output_mf('high'))
        
        # Rule 12: Very far + Critical = Medium (can't reach quickly anyway)
        r12 = min(distance_mf.get('very_far', 0), battery_mf.get('critical', 0))
        output = np.fmax(output, r12 * self._priority_output_mf('medium'))
        
        # Rule 13: Near + Short wait + High battery = Very Low
        r13 = min(distance_mf.get('near', 0), wait_mf.get('short', 0), battery_mf.get('high', 0))
        output = np.fmax(output, r13 * self._priority_output_mf('very_low'))
        
        # Rule 14: Medium everything = Medium Priority
        r14 = min(battery_mf.get('medium', 0), urgency_mf.get('medium', 0), 
                  distance_mf.get('moderate', 0))
        output = np.fmax(output, r14 * self._priority_output_mf('medium'))
        
        # Rule 15: Cooperation bonus (represented as urgency modifier)
        # High urgency always gets attention
        r15 = urgency_mf.get('high', 0) * 0.5
        output = np.fmax(output, r15 * self._priority_output_mf('high'))
        
        return output
    
    def defuzzify_centroid(self, output: np.ndarray) -> float:
        """Centroid defuzzification"""
        numerator = np.sum(self.priority_range * output)
        denominator = np.sum(output)
        if denominator == 0:
            return 50.0  # Default middle priority
        return float(numerator / denominator)
    
    # ============ MAIN INTERFACE ============
    
    def calculate_priority(self, battery: float, distance_km: float, 
                          urgency: int = 5, wait_mins: int = 0,
                          cooperation_bonus: float = 0.0) -> FuzzyMembershipResult:
        """
        Calculate priority score with full membership details.
        
        Args:
            battery: Battery percentage (0-100)
            distance_km: Distance to station in km
            urgency: Urgency level (0-10)
            wait_mins: Current wait time in minutes
            cooperation_bonus: Bonus points from cooperation score
        
        Returns:
            FuzzyMembershipResult with all membership values and final score
        """
        # Get all membership values
        battery_mf = self.battery_membership(battery)
        distance_mf = self.distance_membership(distance_km)
        urgency_mf = self.urgency_membership(urgency)
        wait_mf = self.wait_time_membership(wait_mins)
        
        # Apply rules
        output_fuzzy = self.apply_rules(battery_mf, distance_mf, urgency_mf, wait_mf)
        
        # Defuzzify
        base_priority = self.defuzzify_centroid(output_fuzzy)
        
        # Add cooperation bonus (max 10 points)
        final_priority = min(100, base_priority + (cooperation_bonus * 0.1))
        
        return FuzzyMembershipResult(
            battery=battery_mf,
            distance=distance_mf,
            urgency=urgency_mf,
            wait_time=wait_mf,
            priority_output=output_fuzzy,
            defuzzified_value=final_priority
        )
    
    def get_membership_plot_data(self) -> Dict:
        """Get data for plotting membership functions"""
        return {
            'battery': {
                'range': self.battery_range.tolist(),
                'critical': [self.battery_membership(x)['critical'] for x in self.battery_range],
                'low': [self.battery_membership(x)['low'] for x in self.battery_range],
                'medium': [self.battery_membership(x)['medium'] for x in self.battery_range],
                'high': [self.battery_membership(x)['high'] for x in self.battery_range],
            },
            'priority': {
                'range': self.priority_range.tolist(),
                'very_low': self._priority_output_mf('very_low').tolist(),
                'low': self._priority_output_mf('low').tolist(),
                'medium': self._priority_output_mf('medium').tolist(),
                'high': self._priority_output_mf('high').tolist(),
                'very_high': self._priority_output_mf('very_high').tolist(),
            }
        }


# Singleton instance
fuzzy_engine = MamdaniFuzzyEngine()


def quick_priority_score(battery: float, distance_km: float, 
                         urgency: int = 5, wait_mins: int = 0) -> float:
    """Quick helper function for priority calculation"""
    result = fuzzy_engine.calculate_priority(battery, distance_km, urgency, wait_mins)
    return result.defuzzified_value
