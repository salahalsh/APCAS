"""
Polymer Construction Monitoring Module
-----------------------------------
Provides real-time monitoring and analysis during polymer construction process.
"""

from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np

class PolymerMonitor:
    """
    Real-time monitoring system for polymer construction process.
    Tracks structural properties, validates construction steps,
    and generates comprehensive monitoring reports.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize the monitoring system with output directory setup."""
        self.output_dir = output_dir
        self.monitor_dir = output_dir / 'monitoring'
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.alerts = []
    
    def monitor_step(self, polymer: Chem.Mol, step: int, 
                    current_monomer: str) -> Dict:
        """
        Monitor a single construction step and calculate relevant metrics.
        
        Args:
            polymer: Current polymer structure
            step: Construction step number
            current_monomer: Name of current monomer
            
        Returns:
            Dictionary containing monitoring metrics and any alerts
        """
        try:
            metrics = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'current_monomer': current_monomer,
                'metrics': self._calculate_step_metrics(polymer)
            }
            
            # Check for potential issues
            alerts = self._check_alerts(metrics['metrics'])
            if alerts:
                metrics['alerts'] = alerts
                self.alerts.extend(alerts)
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Save monitoring data periodically
            if step % 10 == 0:
                self._save_monitoring_data()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring step {step}: {e}")
            return {}
    
    def _calculate_step_metrics(self, mol: Chem.Mol) -> Dict:
        """
        Calculate comprehensive metrics for the current structure.
        
        Calculates structural, energetic, and geometric properties
        to assess the quality of the growing polymer.
        """
        metrics = {
            # Structure metrics
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'molecular_weight': Descriptors.ExactMolWt(mol),
            
            # Physical properties
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'ring_count': rdMolDescriptors.CalcNumRings(mol),
            'hbond_donors': rdMolDescriptors.CalcNumHBD(mol),
            'hbond_acceptors': rdMolDescriptors.CalcNumHBA(mol),
            'tpsa': Descriptors.TPSA(mol),
            
            # Energy and strain
            'strain_energy': self._estimate_strain_energy(mol),
            'total_energy': self._calculate_total_energy(mol)
        }
        
        # Add 3D metrics if conformer exists
        if mol.GetNumConformers() > 0:
            metrics.update({
                'end_to_end_distance': self._calculate_end_to_end_distance(mol),
                'radius_of_gyration': self._calculate_radius_of_gyration(mol),
                'structure_density': self._calculate_structure_density(mol)
            })
        
        return metrics
    
    def _calculate_end_to_end_distance(self, mol: Chem.Mol) -> float:
        """Calculate end-to-end distance of polymer chain."""
        try:
            conf = mol.GetConformer()
            pos_0 = conf.GetAtomPosition(0)
            pos_n = conf.GetAtomPosition(mol.GetNumAtoms() - 1)
            
            distance = np.sqrt(
                (pos_0.x - pos_n.x)**2 + 
                (pos_0.y - pos_n.y)**2 +
                (pos_0.z - pos_n.z)**2
            )
            
            return float(distance)
            
        except Exception as e:
            self.logger.error(f"Error calculating end-to-end distance: {e}")
            return 0.0
    
    def _calculate_radius_of_gyration(self, mol: Chem.Mol) -> float:
        """Calculate radius of gyration for current structure."""
        try:
            conf = mol.GetConformer()
            positions = []
            
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            
            positions = np.array(positions)
            center = np.mean(positions, axis=0)
            rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
            
            return float(rg)
            
        except Exception as e:
            self.logger.error(f"Error calculating radius of gyration: {e}")
            return 0.0
    
    def _calculate_structure_density(self, mol: Chem.Mol) -> float:
        """Estimate structure density based on molecular volume."""
        try:
            # Calculate molecular volume
            volume = AllChem.ComputeMolVolume(mol)
            
            # Calculate mass
            mass = Descriptors.ExactMolWt(mol)
            
            if volume > 0:
                return mass / volume
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating structure density: {e}")
            return 0.0
    
    def _estimate_strain_energy(self, mol: Chem.Mol) -> float:
        """Estimate molecular strain energy using MMFF94."""
        try:
            # Calculate MMFF94 energy as proxy for strain
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return 0.0
            
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return 0.0
            
            energy = ff.CalcEnergy()
            return float(energy)
            
        except Exception as e:
            self.logger.error(f"Error estimating strain energy: {e}")
            return 0.0
    
    def _calculate_total_energy(self, mol: Chem.Mol) -> float:
        """Calculate total energy of current structure."""
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return 0.0
            
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return 0.0
            
            return float(ff.CalcEnergy())
            
        except Exception as e:
            self.logger.error(f"Error calculating total energy: {e}")
            return 0.0
    
    def _check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for potential issues in current metrics."""
        alerts = []
        
        # Check strain energy
        strain_per_atom = metrics['strain_energy'] / max(1, metrics['num_atoms'])
        if strain_per_atom > 10.0:
            alerts.append({
                'type': 'high_strain',
                'message': 'High strain energy detected',
                'value': strain_per_atom,
                'threshold': 10.0
            })
        
        # Check structure density if available
        if 'structure_density' in metrics:
            if metrics['structure_density'] < 0.8:
                alerts.append({
                    'type': 'low_density',
                    'message': 'Structure density below threshold',
                    'value': metrics['structure_density'],
                    'threshold': 0.8
                })
        
        # Check end-to-end distance if available
        if 'end_to_end_distance' in metrics:
            expected_distance = metrics['num_bonds'] * 1.5  # Approximate
            if metrics['end_to_end_distance'] < expected_distance * 0.5:
                alerts.append({
                    'type': 'chain_folding',
                    'message': 'Possible excessive chain folding',
                    'value': metrics['end_to_end_distance'],
                    'threshold': expected_distance * 0.5
                })
        
        return alerts
    
    def _save_monitoring_data(self):
        """Save current monitoring data to file."""
        try:
            data = {
                'metrics_history': self.metrics_history,
                'alerts': self.alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.monitor_dir / 'monitoring_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report with analysis."""
        try:
            sections = []
            
            # Header
            sections.append("Polymer Construction Monitoring Report")
            sections.append("=====================================")
            sections.append(f"Generated: {datetime.now().isoformat()}\n")
            
            # Summary statistics
            sections.append("Construction Summary")
            sections.append("--------------------")
            sections.append(f"Total Steps: {len(self.metrics_history)}")
            sections.append(f"Total Alerts: {len(self.alerts)}\n")
            
            # Alert summary
            if self.alerts:
                sections.append("Construction Alerts")
                sections.append("------------------")
                for alert in self.alerts:
                    sections.append(
                        f"- {alert['message']} "
                        f"(Value: {alert['value']:.2f}, "
                        f"Threshold: {alert['threshold']:.2f})"
                    )
                sections.append("")
            
            # Metric trends
            sections.append("Metric Trends")
            sections.append("-------------")
            trends = self._analyze_trends()
            for metric, trend in trends.items():
                sections.append(f"{metric}:")
                sections.append(f"  Final Value: {trend['final_value']:.2f}")
                sections.append(f"  Trend: {trend['trend']}")
                if trend.get('warning'):
                    sections.append(f"  Warning: {trend['warning']}")
                sections.append("")
            
            report = '\n'.join(sections)
            
            # Save report
            with open(self.monitor_dir / 'monitoring_report.txt', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {e}")
            return ""
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends in monitored metrics."""
        trends = {}
        
        try:
            if not self.metrics_history:
                return trends
            
            # Get metrics keys from first entry
            metric_keys = self.metrics_history[0]['metrics'].keys()
            
            for key in metric_keys:
                values = [step['metrics'].get(key, 0) 
                         for step in self.metrics_history]
                
                if not values:
                    continue
                
                trend_info = {
                    'final_value': values[-1],
                    'trend': self._determine_trend(values)
                }
                
                # Add warnings for concerning trends
                warning = self._check_trend_warning(key, values)
                if warning:
                    trend_info['warning'] = warning
                
                trends[key] = trend_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
        
        return trends
    
    def _determine_trend(self, values: List[float]) -> str:
        """Determine trend direction in values."""
        try:
            if len(values) < 2:
                return "insufficient data"
            
            # Calculate moving average
            window = min(5, len(values))
            avg_values = np.convolve(values, 
                                   np.ones(window)/window, 
                                   mode='valid')
            
            # Calculate trend
            slope = (avg_values[-1] - avg_values[0]) / len(avg_values)
            
            if abs(slope) < 0.01:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
            
        except Exception:
            return "unknown"
    
    def _check_trend_warning(self, metric: str, values: List[float]) -> Optional[str]:
        """Check if trend warrants a warning message."""
        try:
            if len(values) < 3:
                return None
            
            # Calculate rate of change
            changes = np.diff(values)
            avg_change = np.mean(changes)
            
            # Metric-specific thresholds
            thresholds = {
                'strain_energy': (5.0, "Rapidly increasing strain energy"),
                'structure_density': (-0.05, "Decreasing structure density"),
                'end_to_end_distance': (-2.0, "Decreasing end-to-end distance")
            }
            
            if metric in thresholds:
                threshold, message = thresholds[metric]
                if metric in ['strain_energy']:
                    if avg_change > threshold:
                        return message
                else:
                    if avg_change < threshold:
                        return message
            
            return None
            
        except Exception:
            return None