"""
wartegg_clinical_integration_optimized.py

Versione ottimizzata per perfetta integrazione con Wartegg_optimized.claude.py
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np
from datetime import datetime

# ----------------------------
# STRUTTURE DATI ESTESE
# ----------------------------

@dataclass
class PatternDescriptor:
    id: str
    name: str
    description: str
    severity: float  # 0..1
    evidence_level: Optional[str] = None
    supporting_features: Optional[Dict[str, float]] = None

@dataclass
class InterventionDescriptor:
    id: str
    title: str
    short_description: str
    rationale: str
    primary_evidence: List[str]
    recommended_level: str
    efficacy_score: Optional[float] = None  # 0-1 scale

@dataclass 
class ClinicalIndicatorReport:
    pattern_detected: List[PatternDescriptor]
    suggested_interventions: List[InterventionDescriptor]
    caveats: List[str]
    feature_summary: Optional[Dict[str, Any]] = None

# ----------------------------
# BIBLIOGRAFIA ESTESA
# ----------------------------

BIBLIOGRAPHY = {
    # ... (mantenere bibliografia esistente)
    "Cuijpers2014": "Cuijpers et al., JAMA Psychiatry (2014)",
    "Kuyken2016": "Kuyken et al., The Lancet (2016)"
}

# ----------------------------
# INTERVENTI CLINICI AGGIORNATI
# ----------------------------

INTERVENTION_LIBRARY = {
    "CBT_ANXIETY": InterventionDescriptor(
        id="CBT_ANXIETY",
        title="Cognitive Behavioral Therapy - Anxiety Protocol",
        short_description="CBT with exposure and cognitive restructuring",
        rationale="Effective for high pressure, tremor, and fragmented strokes",
        primary_evidence=[BIBLIOGRAPHY["Beck2011"], BIBLIOGRAPHY["Hofmann2012"]],
        recommended_level="first-line",
        efficacy_score=0.85
    ),
    
    "DBT_DYSREGULATION": InterventionDescriptor(
        id="DBT_DYSREGULATION",
        title="DBT Emotion Regulation Skills",
        short_description="Dialectical Behavior Therapy modules",
        rationale="For variable pressure, spatial disorganization",
        primary_evidence=[BIBLIOGRAPHY["Linehan2015"]],
        recommended_level="first-line",
        efficacy_score=0.78
    ),
    
    # ... (altri interventi)
}

# ----------------------------
# MAPPER OTTIMIZZATO
# ----------------------------

class WarteggClinicalMapper:
    def __init__(self, config: Optional[Dict] = None):
        # Configurazione allineata con Claude
        self.config = {
            'pressure': {
                'high': 0.75,
                'low': 0.25,
                'variance_threshold': 0.15
            },
            'symmetry': {
                'high': 0.8,
                'low': 0.2,
                'variance_threshold': 0.1
            },
            'space': {
                'low_occupation': 0.05,
                'high_occupation': 0.4
            },
            'contours': {
                'low': 1,
                'high': 6
            }
        }
        if config:
            self._deep_update(self.config, config)
    
    def process_claude_output(self, claude_data: Union[Dict, List[Dict]]) -> ClinicalIndicatorReport:
        """Processa output diretto da Claude, singolo quadro o lista"""
        if isinstance(claude_data, list):
            return self.generate_report(claude_data)
        
        # Converti singolo quadro in lista
        return self.generate_report([claude_data])
    
    def generate_report(self, frame_analyses: List[Dict]) -> ClinicalIndicatorReport:
        """Genera report clinico completo con feature avanzate"""
        patterns = []
        interventions = []
        feature_stats = self._compute_advanced_stats(frame_analyses)
        
        # 1. Alta pressione e ansia
        if feature_stats['pressure']['mean'] >= self.config['pressure']['high']:
            severity = self._calculate_severity(
                feature_stats['pressure']['mean'],
                self.config['pressure']['high'],
                1.0
            )
            patterns.append(
                self._create_pattern(
                    "HIGH_PRESSURE", 
                    "High drawing pressure",
                    "Indicative of tension/hyperactivation",
                    severity,
                    {
                        'mean_pressure': feature_stats['pressure']['mean'],
                        'pressure_variance': feature_stats['pressure']['variance'],
                        'tremor_index': feature_stats['stroke']['mean_tremor']
                    }
                )
            )
            interventions.append(INTERVENTION_LIBRARY["CBT_ANXIETY"])
        
        # 2. Disregolazione emotiva
        if (feature_stats['pressure']['variance'] > self.config['pressure']['variance_threshold'] or
            feature_stats['symmetry']['variance'] > self.config['symmetry']['variance_threshold']):
            
            severity = max(
                self._normalize(feature_stats['pressure']['variance'], 0, 0.5),
                self._normalize(feature_stats['symmetry']['variance'], 0, 0.5)
            )
            
            patterns.append(
                self._create_pattern(
                    "EMOTIONAL_DYSREGULATION",
                    "Emotional lability",
                    "High variability across frames suggests emotional dysregulation",
                    severity,
                    {
                        'pressure_variance': feature_stats['pressure']['variance'],
                        'symmetry_variance': feature_stats['symmetry']['variance'],
                        'stroke_quality_var': feature_stats['stroke']['variance_quality']
                    }
                )
            )
            interventions.append(INTERVENTION_LIBRARY["DBT_DYSREGULATION"])
        
        # ... (altri pattern)
        
        # Deduplica e ordina interventi per efficacia
        unique_interventions = self._deduplicate_and_sort_interventions(interventions)
        
        return ClinicalIndicatorReport(
            pattern_detected=patterns,
            suggested_interventions=unique_interventions,
            caveats=[
                "Clinical interpretation should be done by qualified professionals",
                "Patterns are probabilistic indicators based on drawing features",
                f"Analysis generated on {datetime.now().isoformat()}"
            ],
            feature_summary=feature_stats
        )
    
    def _extract_features(self, frame: Dict) -> Dict:
        """Estrae e allinea le feature dal formato Claude"""
        return {
            'pressure_mean_intensity': frame.get('pressure_score', 0.5),
            'pressure_variance': frame.get('pressure_variance', 0),
            'vertical_symmetry_diff': frame.get('symmetry_score', 0.5),
            'n_contours': frame.get('detail_score', 3),
            'stroke_area_ratio': frame.get('space_usage', 0.2),
            'tremor_index': frame.get('tremor_index', 0),
            'stroke_quality': frame.get('stroke_quality', 0.7),
            'spatial_entropy': frame.get('spatial_entropy', 0.5)
        }
    
    def _compute_advanced_stats(self, frames: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche avanzate allineate con feature di Claude"""
        features = [self._extract_features(f) for f in frames]
        
        stats = {
            'pressure': {
                'mean': np.mean([f['pressure_mean_intensity'] for f in features]),
                'variance': np.var([f['pressure_mean_intensity'] for f in features]),
                'min': np.min([f['pressure_mean_intensity'] for f in features]),
                'max': np.max([f['pressure_mean_intensity'] for f in features])
            },
            'symmetry': {
                'mean': np.mean([f['vertical_symmetry_diff'] for f in features]),
                'variance': np.var([f['vertical_symmetry_diff'] for f in features])
            },
            'stroke': {
                'mean_tremor': np.mean([f['tremor_index'] for f in features]),
                'mean_quality': np.mean([f['stroke_quality'] for f in features]),
                'variance_quality': np.var([f['stroke_quality'] for f in features])
            },
            'space': {
                'mean_occupation': np.mean([f['stroke_area_ratio'] for f in features]),
                'mean_entropy': np.mean([f['spatial_entropy'] for f in features])
            }
        }
        
        return stats
    
    def _deduplicate_and_sort_interventions(self, interventions: List[InterventionDescriptor]):
        """Deduplica e ordina interventi per efficacy_score"""
        unique = {i.id: i for i in interventions}
        return sorted(unique.values(), key=lambda x: x.efficacy_score, reverse=True)
    
    @staticmethod
    def _calculate_severity(value: float, threshold: float, max_value: float) -> float:
        """Calcola severità normalizzata 0-1"""
        return min(1.0, max(0.0, (value - threshold) / (max_value - threshold)))
    
    @staticmethod 
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalizza valore tra 0 e 1"""
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def _deep_update(original: Dict, update: Dict):
        """Aggiornamento ricorsivo dei dizionari"""
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                WarteggClinicalMapper._deep_update(original[key], value)
            else:
                original[key] = value

# ----------------------------
# ESEMPIO DI UTILIZZO AGGIORNATO
# ----------------------------

if __name__ == "__main__":
    # Esempio con output avanzato da Claude
    claude_output = {
        'pressure_score': 0.82,
        'pressure_variance': 0.18,
        'symmetry_score': 0.25,
        'detail_score': 5,
        'space_usage': 0.12,
        'tremor_index': 0.45,
        'stroke_quality': 0.65,
        'spatial_entropy': 0.78
    }
    
    # Configurazione personalizzata
    config = {
        'pressure': {
            'high': 0.8,  # Soglia più alta per contesto clinico
            'variance_threshold': 0.12
        }
    }
    
    mapper = WarteggClinicalMapper(config)
    report = mapper.process_claude_output(claude_output)
    
    # Salva report
    with open("wartegg_clinical_report.json", "w") as f:
        json.dump(asdict(report), f, indent=2)
    
    print("Report clinico generato:")
    print(json.dumps(asdict(report), indent=2))