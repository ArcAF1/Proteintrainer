"""
Lightweight Simulation Tools for Biomedical Research
Provides simple models for testing hypotheses without real experiments
"""
from __future__ import annotations

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Lightweight simulation engine for biomedical experiments.
    Provides simple models for muscle training, supplements, and performance.
    """
    
    def __init__(self):
        self.available = True
        self.simulations = {
            'training_response': self._simulate_training_response,
            'supplement_effect': self._simulate_supplement_effect,
            'muscle_growth': self._simulate_muscle_growth,
            'fatigue_recovery': self._simulate_fatigue_recovery,
            'dose_response': self._simulate_dose_response,
            'metabolic_adaptation': self._simulate_metabolic_adaptation
        }
        
    async def run_simulation(self, simulation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific simulation with given parameters.
        
        Args:
            simulation_type: Type of simulation to run
            parameters: Simulation parameters
            
        Returns:
            Simulation results
        """
        if simulation_type not in self.simulations:
            return {
                'error': f'Unknown simulation type: {simulation_type}',
                'available_types': list(self.simulations.keys())
            }
            
        try:
            simulator = self.simulations[simulation_type]
            results = simulator(parameters)
            
            # Add metadata
            results['simulation_type'] = simulation_type
            results['timestamp'] = datetime.now().isoformat()
            results['parameters_used'] = parameters
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation error: {str(e)}")
            return {
                'error': str(e),
                'simulation_type': simulation_type,
                'parameters': parameters
            }
            
    def _simulate_training_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate training adaptation over time."""
        # Parameters
        duration_weeks = params.get('duration_weeks', 8)
        training_intensity = params.get('intensity', 0.7)  # 0-1 scale
        baseline_performance = params.get('baseline', {}).get('performance', 100)
        training_frequency = params.get('frequency_per_week', 3)
        
        # Simple training adaptation model
        weeks = np.arange(0, duration_weeks + 1)
        
        # Performance improvement follows logarithmic curve with diminishing returns
        # Factors: intensity, frequency, recovery
        adaptation_rate = 0.05 * training_intensity * (training_frequency / 3)
        performance = baseline_performance * (1 + adaptation_rate * np.log1p(weeks))
        
        # Add some noise
        noise = np.random.normal(0, 2, len(weeks))
        performance += noise
        
        # Fatigue accumulation
        fatigue = 10 * training_intensity * np.sqrt(weeks)
        effective_performance = performance - fatigue * 0.3
        
        return {
            'weeks': weeks.tolist(),
            'performance': performance.tolist(),
            'effective_performance': effective_performance.tolist(),
            'fatigue': fatigue.tolist(),
            'improvement_percent': ((performance[-1] - baseline_performance) / baseline_performance * 100),
            'peak_week': int(np.argmax(effective_performance)),
            'recommendation': self._get_training_recommendation(adaptation_rate, fatigue[-1])
        }
        
    def _simulate_supplement_effect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the effect of a supplement on performance."""
        # Parameters
        supplement_type = params.get('supplement', 'creatine')
        dose_mg = params.get('dose_mg', 5000)
        duration_weeks = params.get('duration_weeks', 8)
        baseline = params.get('baseline', {})
        
        # Simple supplement effect models
        supplement_effects = {
            'creatine': {
                'strength_multiplier': 1.05 + (dose_mg / 5000) * 0.03,
                'endurance_multiplier': 1.02,
                'recovery_multiplier': 1.1,
                'onset_weeks': 2
            },
            'beta_alanine': {
                'strength_multiplier': 1.02,
                'endurance_multiplier': 1.08 + (dose_mg / 3000) * 0.02,
                'recovery_multiplier': 1.05,
                'onset_weeks': 4
            },
            'caffeine': {
                'strength_multiplier': 1.03,
                'endurance_multiplier': 1.06,
                'recovery_multiplier': 0.95,  # Can impair recovery
                'onset_weeks': 0.1  # Immediate effect
            }
        }
        
        # Get supplement profile
        if supplement_type.lower() in supplement_effects:
            effects = supplement_effects[supplement_type.lower()]
        else:
            # Generic supplement
            effects = {
                'strength_multiplier': 1.02,
                'endurance_multiplier': 1.02,
                'recovery_multiplier': 1.02,
                'onset_weeks': 2
            }
            
        # Simulate time course
        weeks = np.arange(0, duration_weeks + 1)
        
        # Sigmoid onset curve
        onset_curve = 1 / (1 + np.exp(-2 * (weeks - effects['onset_weeks'])))
        
        # Apply effects
        strength = baseline.get('strength', 100) * (1 + (effects['strength_multiplier'] - 1) * onset_curve)
        endurance = baseline.get('endurance', 100) * (1 + (effects['endurance_multiplier'] - 1) * onset_curve)
        recovery = baseline.get('recovery', 100) * (1 + (effects['recovery_multiplier'] - 1) * onset_curve)
        
        return {
            'weeks': weeks.tolist(),
            'strength': strength.tolist(),
            'endurance': endurance.tolist(),
            'recovery': recovery.tolist(),
            'onset_curve': onset_curve.tolist(),
            'effects_summary': {
                'strength_gain': f"{(effects['strength_multiplier'] - 1) * 100:.1f}%",
                'endurance_gain': f"{(effects['endurance_multiplier'] - 1) * 100:.1f}%",
                'recovery_change': f"{(effects['recovery_multiplier'] - 1) * 100:.1f}%",
                'time_to_effect': f"{effects['onset_weeks']} weeks"
            }
        }
        
    def _simulate_muscle_growth(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate muscle hypertrophy response."""
        # Parameters
        training_volume = params.get('volume_sets_per_week', 12)
        protein_intake_g_per_kg = params.get('protein_g_per_kg', 1.6)
        calories_surplus = params.get('calorie_surplus', 300)
        duration_weeks = params.get('duration_weeks', 12)
        baseline_muscle_mass = params.get('baseline_mass_kg', 30)
        
        # Muscle growth model
        weeks = np.arange(0, duration_weeks + 1)
        
        # Growth rate factors
        volume_factor = np.clip(training_volume / 15, 0.5, 1.5)
        protein_factor = np.clip(protein_intake_g_per_kg / 1.6, 0.7, 1.2)
        calorie_factor = np.clip(calories_surplus / 300, 0.5, 1.3)
        
        # Combined growth rate (% per week)
        growth_rate = 0.25 * volume_factor * protein_factor * calorie_factor
        
        # Diminishing returns over time
        growth_multiplier = 1 - np.exp(-0.1 * weeks)
        
        # Calculate muscle mass
        muscle_mass = baseline_muscle_mass * (1 + growth_rate * 0.01 * weeks * growth_multiplier)
        
        # Add variability
        noise = np.random.normal(0, 0.1, len(weeks))
        muscle_mass += noise
        
        # Calculate cross-sectional area (proportional to mass^(2/3))
        csa = 100 * (muscle_mass / baseline_muscle_mass) ** (2/3)
        
        return {
            'weeks': weeks.tolist(),
            'muscle_mass_kg': muscle_mass.tolist(),
            'cross_sectional_area': csa.tolist(),
            'total_gain_kg': muscle_mass[-1] - baseline_muscle_mass,
            'gain_percent': ((muscle_mass[-1] - baseline_muscle_mass) / baseline_muscle_mass * 100),
            'growth_factors': {
                'volume_factor': volume_factor,
                'protein_factor': protein_factor,
                'calorie_factor': calorie_factor,
                'weekly_growth_rate': f"{growth_rate:.2f}%"
            }
        }
        
    def _simulate_fatigue_recovery(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate fatigue and recovery dynamics."""
        # Parameters
        initial_fatigue = params.get('initial_fatigue', 80)  # 0-100 scale
        recovery_rate = params.get('recovery_rate', 0.3)  # Per day
        sleep_quality = params.get('sleep_quality', 0.8)  # 0-1 scale
        nutrition_quality = params.get('nutrition_quality', 0.8)  # 0-1 scale
        days = params.get('days', 7)
        
        # Recovery model
        time_points = np.linspace(0, days, days * 24)  # Hourly resolution
        
        # Adjust recovery rate based on factors
        effective_recovery_rate = recovery_rate * sleep_quality * nutrition_quality
        
        # Exponential recovery
        fatigue = initial_fatigue * np.exp(-effective_recovery_rate * time_points / 24)
        
        # Performance is inverse of fatigue
        performance = 100 - fatigue
        
        # Time to specific recovery levels
        time_to_50 = -np.log(0.5) / effective_recovery_rate * 24
        time_to_90 = -np.log(0.1) / effective_recovery_rate * 24
        
        return {
            'hours': time_points.tolist(),
            'fatigue': fatigue.tolist(),
            'performance': performance.tolist(),
            'recovery_metrics': {
                'time_to_50_percent_recovery': f"{time_to_50:.1f} hours",
                'time_to_90_percent_recovery': f"{time_to_90:.1f} hours",
                'effective_recovery_rate': effective_recovery_rate,
                'recovery_quality_score': sleep_quality * nutrition_quality
            }
        }
        
    def _simulate_dose_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate dose-response relationship for an intervention."""
        # Parameters
        intervention = params.get('intervention', 'generic')
        min_dose = params.get('min_dose', 0)
        max_dose = params.get('max_dose', 100)
        optimal_dose = params.get('optimal_dose', None)
        
        # Generate dose range
        doses = np.linspace(min_dose, max_dose, 50)
        
        if optimal_dose is None:
            optimal_dose = (max_dose - min_dose) * 0.7 + min_dose
            
        # Different dose-response curves
        if params.get('response_type', 'sigmoid') == 'sigmoid':
            # Sigmoid response
            response = 100 / (1 + np.exp(-0.1 * (doses - optimal_dose * 0.5)))
        elif params.get('response_type') == 'inverted_u':
            # Inverted U-shape (too much is harmful)
            response = 100 * np.exp(-0.5 * ((doses - optimal_dose) / (optimal_dose * 0.5)) ** 2)
        else:
            # Linear with plateau
            response = np.minimum(100, 100 * doses / optimal_dose)
            
        # Add side effects (increase with dose)
        side_effects = 10 * (doses / max_dose) ** 2
        
        # Net benefit
        net_benefit = response - side_effects
        
        # Find optimal dose
        optimal_idx = np.argmax(net_benefit)
        
        return {
            'doses': doses.tolist(),
            'response': response.tolist(),
            'side_effects': side_effects.tolist(),
            'net_benefit': net_benefit.tolist(),
            'optimal_dose_calculated': doses[optimal_idx],
            'max_response': response[optimal_idx],
            'recommendation': self._get_dose_recommendation(doses[optimal_idx], optimal_dose)
        }
        
    def _simulate_metabolic_adaptation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate metabolic adaptations to training."""
        # Parameters
        training_type = params.get('training_type', 'mixed')  # endurance, strength, mixed
        duration_weeks = params.get('duration_weeks', 12)
        baseline_vo2max = params.get('baseline_vo2max', 40)  # ml/kg/min
        baseline_rmr = params.get('baseline_rmr', 1800)  # kcal/day
        
        # Training-specific adaptations
        adaptations = {
            'endurance': {
                'vo2max_gain': 0.02,  # 2% per week
                'rmr_change': -0.002,  # Slight decrease
                'mitochondrial_gain': 0.03,
                'glycolytic_change': -0.01
            },
            'strength': {
                'vo2max_gain': 0.005,
                'rmr_change': 0.003,  # Increase due to muscle
                'mitochondrial_gain': 0.01,
                'glycolytic_change': 0.02
            },
            'mixed': {
                'vo2max_gain': 0.012,
                'rmr_change': 0.001,
                'mitochondrial_gain': 0.02,
                'glycolytic_change': 0.01
            }
        }
        
        adapt = adaptations.get(training_type, adaptations['mixed'])
        
        # Simulate time course
        weeks = np.arange(0, duration_weeks + 1)
        
        # Adaptations with diminishing returns
        time_factor = 1 - np.exp(-0.2 * weeks)
        
        vo2max = baseline_vo2max * (1 + adapt['vo2max_gain'] * weeks * time_factor)
        rmr = baseline_rmr * (1 + adapt['rmr_change'] * weeks)
        mitochondrial_density = 100 * (1 + adapt['mitochondrial_gain'] * weeks * time_factor)
        glycolytic_capacity = 100 * (1 + adapt['glycolytic_change'] * weeks * time_factor)
        
        return {
            'weeks': weeks.tolist(),
            'vo2max': vo2max.tolist(),
            'rmr': rmr.tolist(),
            'mitochondrial_density': mitochondrial_density.tolist(),
            'glycolytic_capacity': glycolytic_capacity.tolist(),
            'summary': {
                'vo2max_improvement': f"{(vo2max[-1] - baseline_vo2max) / baseline_vo2max * 100:.1f}%",
                'rmr_change': f"{(rmr[-1] - baseline_rmr):.0f} kcal/day",
                'primary_adaptation': 'aerobic' if training_type == 'endurance' else 'anaerobic'
            }
        }
        
    def _get_training_recommendation(self, adaptation_rate: float, final_fatigue: float) -> str:
        """Generate training recommendation based on simulation."""
        if final_fatigue > 50:
            return "Consider reducing training intensity or adding more recovery days"
        elif adaptation_rate < 0.03:
            return "Training stimulus may be too low - consider increasing intensity or volume"
        else:
            return "Training parameters appear well-balanced for continued adaptation"
            
    def _get_dose_recommendation(self, calculated_optimal: float, specified_optimal: float) -> str:
        """Generate dosing recommendation."""
        if abs(calculated_optimal - specified_optimal) / specified_optimal > 0.2:
            return f"Calculated optimal dose ({calculated_optimal:.1f}) differs from expected - consider individual variation"
        else:
            return f"Optimal dose confirmed around {calculated_optimal:.1f} units"
            
    async def analyze_experiment_results(self, experiment_design: Dict, simulation_results: Dict) -> Dict[str, Any]:
        """Analyze simulation results in context of experiment design."""
        hypothesis = experiment_design.get('hypothesis', 'Unknown hypothesis')
        
        # Determine if hypothesis is supported based on results
        hypothesis_supported = False
        confidence = 0.5
        
        # Simple analysis based on simulation type
        if simulation_results.get('simulation_type') == 'training_response':
            improvement = simulation_results.get('improvement_percent', 0)
            hypothesis_supported = improvement > 5  # Meaningful improvement threshold
            confidence = min(0.9, 0.5 + improvement / 20)
            
        elif simulation_results.get('simulation_type') == 'supplement_effect':
            effects = simulation_results.get('effects_summary', {})
            # Check if any effect is meaningful
            for effect, value in effects.items():
                if 'gain' in effect and float(value.rstrip('%')) > 3:
                    hypothesis_supported = True
                    confidence = 0.7
                    break
                    
        analysis = {
            'hypothesis_supported': hypothesis_supported,
            'confidence': confidence,
            'key_findings': self._extract_key_findings(simulation_results),
            'outcome': 'positive' if hypothesis_supported else 'negative',
            'recommendation': 'Proceed to trial design' if hypothesis_supported else 'Revise hypothesis'
        }
        
        return analysis
        
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from simulation results."""
        findings = []
        
        if 'improvement_percent' in results:
            findings.append(f"Performance improved by {results['improvement_percent']:.1f}%")
            
        if 'effects_summary' in results:
            for effect, value in results['effects_summary'].items():
                findings.append(f"{effect.replace('_', ' ').title()}: {value}")
                
        if 'total_gain_kg' in results:
            findings.append(f"Muscle mass gain: {results['total_gain_kg']:.2f} kg")
            
        if 'optimal_dose_calculated' in results:
            findings.append(f"Optimal dose: {results['optimal_dose_calculated']:.1f} units")
            
        return findings[:3]  # Top 3 findings 