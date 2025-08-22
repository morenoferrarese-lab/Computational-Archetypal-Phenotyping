"""
WARTEGG-TAROT INTEGRATED REPORT GENERATOR
Sistema unificato per analisi clinica integrata Wartegg W-16 + 22 Lame Tarocchi
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import numpy as np
from enum import Enum

# Enumerazioni e strutture dati di base
class ArcanoType(Enum):
    MAGGIORE = "maggiore"
    MINORE = "minore"

class DrawingStyle(Enum):
    GEOMETRIC = "geometrico"
    ORGANIC = "organico"
    SYMBOLIC = "simbolico"
    ABSTRACT = "astratto"
    FIGURATIVE = "figurativo"
    MIXED = "misto"

class ClinicalSignificance(Enum):
    NORMAL = 0
    BORDERLINE = 1
    CLINICAL = 2
    SEVERE = 3

@dataclass
class NeuralNetworkActivation:
    executive_control: float
    default_mode: float
    salience: float
    limbic: float
    social_brain: float

@dataclass
class TarotCardAnalysis:
    card_id: int
    card_name: str
    italian_name: str
    arcano_type: ArcanoType
    drawing_style: DrawingStyle
    line_quality: float
    pressure_intensity: float
    symmetry_score: float
    complexity_score: float
    symbolic_density: float
    emotional_tone: float
    wartegg_correlations: Dict[int, float]
    psychological_indicators: List[str]
    neural_activation: Dict[str, float]
    clinical_significance: ClinicalSignificance = ClinicalSignificance.NORMAL
    interpretation: str = ""

@dataclass
class QuadroAnalysis:
    id: int
    title: str
    stimulus: str
    completion: str
    features: Dict[str, float]
    rodella_interpretation: str
    character_analysis: str
    neural_activation: Dict[str, float]
    neurotransmitters: Dict[str, float]
    tarot_correlations: Dict[int, float] = field(default_factory=dict)
    tarot_interpretation: str = ""

@dataclass
class IntegratedClinicalReport:
    # Sezione 1: Intestazione
    subject: str
    date: str
    system_version: str
    analyst: str
    assessment_type: str = "Integrato Wartegg W-16 + 22 Lame Tarocchi"
    
    # Sezione 2: Sintesi Esecutiva
    adaptation_score: float
    predominant_profile: str
    clinical_indicators: List[str]
    neural_networks: NeuralNetworkActivation
    clinical_alerts: List[str]
    tarot_archetype_patterns: List[str]
    
    # Sezione 3: Analisi Wartegg
    quadro_analyses: List[QuadroAnalysis]
    
    # Sezione 4: Analisi Tarocchi
    tarot_analyses: List[TarotCardAnalysis]
    predominant_tarot_themes: Dict[str, float]
    
    # Sezione 5: Profilo Integrato
    big_five: Dict[str, float]
    defense_mechanisms: Dict[str, float]
    neurochemical_balance: Dict[str, Dict[str, Any]]
    archetypal_patterns: Dict[str, float]
    
    # Sezione 6: Analisi Clinica Integrata
    psychopathological_dynamics: List[str]
    prognostic_factors: Dict[str, List[str]]
    risk_scores: Dict[str, float]
    integration_insights: List[str]
    
    # Sezione 7: Piano di Trattamento Integrato
    treatment_phases: Dict[str, List[str]]
    neuro_protocols: List[str]
    symbolic_interventions: List[str]
    
    # Sezione 8: Osservazioni
    behavioral_observations: List[str]
    drawing_behavior_patterns: List[str]
    
    # Sezione 9: Conclusioni
    preliminary_diagnosis: str
    immediate_actions: List[str]
    prognosis: str
    spiritual_dimension: str
    
    def to_json(self, filename: str = None):
        report = asdict(self)
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return json.dumps(report, indent=2, ensure_ascii=False)

class IntegratedReportGenerator:
    def __init__(self, system_version="3.0.0"):
        self.system_version = system_version
        self.analyst = "Advanced Psychological AI Team"
        
        # Dati Rodella
        self.rodella_stimuli = {
            1: "Punto centrale (Coscienza dell'Ego)",
            2: "Due curve volte all'interno (Vita interiore)",
            3: "Linea retta verticale (Attività intellettuale)",
            4: "Due rette parallele verticali (Ideale e dialogo culturale)",
            5: "Semicerchio punteggiato (Comportamento affettivo, Eros)",
            6: "Due curve divergenti (Ambiente affettivo)",
            7: "Linea curva concava inferiore (Adattamento sociale)",
            8: "Segmento di circonferenza superiore (Rapporto con autorità)",
            9: "Tre linee parallele ascendenti (Impegno e aspirazioni)",
            10: "Due rette opposte (Comportamento volontario)",
            11: "Freccia centrale verso il basso (Atteggiamento verso difficoltà)",
            12: "Linea verticale e orizzontale non congiunte (Sintesi mentale)",
            13: "Linea ondulata alto-sinistra (Atteggiamento estetico)",
            14: "Piccolo quadrato nero alto-destra (Super-ego)",
            15: "Linea orizzontale inferiore (Oggettività)",
            16: "Piccolo rettangolo inferiore-sinistro (Adattamento ambientale)"
        }
        
        # Database completo delle 22 Lame Maggiori
        self.tarot_database = self._initialize_complete_tarot_database()
        
        # Mappature di integrazione complete
        self.wartegg_tarot_mappings = {
            1: [0, 1, 21],    # Ego - Matto, Bagatto, Mondo
            2: [2, 18],       # Vita interiore - Papessa, Luna
            3: [1, 20],       # Intelletto - Bagatto, Giudizio
            4: [4, 8],        # Ideali - Imperatore, Giustizia
            5: [3, 6, 15],    # Affettività - Imperatrice, Innamorati, Diavolo
            6: [6, 14, 19],   # Ambiente affettivo - Innamorati, Temperanza, Sole
            7: [7, 11, 16],   # Adattamento sociale - Carro, Forza, Torre
            8: [4, 5, 8],     # Autorità - Imperatore, Papa, Giustizia
            9: [9, 17],       # Aspirazioni - Eremita, Stelle
            10: [10, 16],     # Volontà - Ruota, Torre
            11: [11, 12, 13], # Difficoltà - Forza, Appeso, Morte
            12: [8, 12, 20],  # Sintesi - Giustizia, Appeso, Giudizio
            13: [13, 18],     # Estetica - Morte, Luna
            14: [4, 8, 14],   # Super-ego - Imperatore, Giustizia, Temperanza
            15: [14, 19],     # Oggettività - Temperanza, Sole
            16: [0, 10, 21]   # Adattamento - Matto, Ruota, Mondo
        }

    def _initialize_complete_tarot_database(self):
        """Inizializza il database completo delle 22 Lame Maggiori"""
        return {
            # Le 22 Lame Maggiori complete
            0: {"name": "The Fool", "italian": "Il Matto", "theme": "inizio/libertà", "element": "aria", 
                "keywords": ["libertà", "spontaneità", "inconsapevolezza", "rischio"]},
            1: {"name": "The Magician", "italian": "Il Bagatto", "theme": "volontà/potere", "element": "aria", 
                "keywords": ["volontà", "concentrazione", "abilità", "manifestazione"]},
            2: {"name": "The High Priestess", "italian": "La Papessa", "theme": "intuizione/mistero", "element": "acqua", 
                "keywords": ["intuizione", "mistero", "subconscio", "saggezza interiore"]},
            3: {"name": "The Empress", "italian": "L'Imperatrice", "theme": "creatività/abbondanza", "element": "terra", 
                "keywords": ["creatività", "naturalezza", "abbondanza", "fertilità"]},
            4: {"name": "The Emperor", "italian": "L'Imperatore", "theme": "struttura/autorità", "element": "fuoco", 
                "keywords": ["struttura", "autorità", "controllo", "ordine"]},
            5: {"name": "The Hierophant", "italian": "Il Papa", "theme": "tradizione/conoscenza", "element": "terra", 
                "keywords": ["tradizione", "conoscenza", "guida", "spiritualità organizzata"]},
            6: {"name": "The Lovers", "italian": "Gli Innamorati", "theme": "scelta/armonia", "element": "aria", 
                "keywords": ["scelta", "armonia", "relazione", "unione"]},
            7: {"name": "The Chariot", "italian": "Il Carro", "theme": "controllo/determinazione", "element": "acqua", 
                "keywords": ["controllo", "determinazione", "progresso", "vittoria"]},
            8: {"name": "Justice", "italian": "La Giustizia", "theme": "equilibrio/decisione", "element": "aria", 
                "keywords": ["equilibrio", "decisione", "karma", "giustizia"]},
            9: {"name": "The Hermit", "italian": "L'Eremita", "theme": "introspezione/saggezza", "element": "terra", 
                "keywords": ["introspezione", "saggezza", "solitudine", "guida interiore"]},
            10: {"name": "Wheel of Fortune", "italian": "La Ruota della Fortuna", "theme": "ciclicità/destino", "element": "fuoco", 
                 "keywords": ["ciclicità", "destino", "cambiamento", "fortuna"]},
            11: {"name": "Strength", "italian": "La Forza", "theme": "coraggio/controllo interiore", "element": "fuoco", 
                 "keywords": ["coraggio", "passione", "controllo interiore", "forza morale"]},
            12: {"name": "The Hanged Man", "italian": "L'Appeso", "theme": "sospensione/sacrificio", "element": "acqua", 
                 "keywords": ["sospensione", "sacrificio", "nuova prospettiva", "resa"]},
            13: {"name": "Death", "italian": "La Morte", "theme": "trasformazione/fine", "element": "acqua", 
                 "keywords": ["trasformazione", "fine", "rinascita", "cambiamento radicale"]},
            14: {"name": "Temperance", "italian": "La Temperanza", "theme": "equilibrio/moderazione", "element": "fuoco", 
                 "keywords": ["equilibrio", "moderazione", "armonizzazione", "pazienza"]},
            15: {"name": "The Devil", "italian": "Il Diavolo", "theme": "attaccamento/materialismo", "element": "terra", 
                 "keywords": ["attaccamento", "materialismo", "schiavitù interiore", "tentazione"]},
            16: {"name": "The Tower", "italian": "La Torre", "theme": "caos/rivelazione", "element": "fuoco", 
                 "keywords": ["caos", "distruzione", "rivelazione improvvisa", "crollo"]},
            17: {"name": "The Star", "italian": "Le Stelle", "theme": "speranza/ispirazione", "element": "aria", 
                 "keywords": ["speranza", "ispirazione", "guida spirituale", "fiducia"]},
            18: {"name": "The Moon", "italian": "La Luna", "theme": "illusione/subconscio", "element": "acqua", 
                 "keywords": ["illusione", "subconscio", "intuizione notturna", "sogno"]},
            19: {"name": "The Sun", "italian": "Il Sole", "theme": "gioia/successo", "element": "fuoco", 
                 "keywords": ["gioia", "successo", "illuminazione", "vitalità"]},
            20: {"name": "Judgement", "italian": "Il Giudizio", "theme": "valutazione/risveglio", "element": "fuoco", 
                 "keywords": ["valutazione", "risveglio", "chiamata interiore", "resurrezione"]},
            21: {"name": "The World", "italian": "Il Mondo", "theme": "completezza/realizzazione", "element": "terra", 
                 "keywords": ["completezza", "realizzazione", "unità", "successo totale"]}
        }

    def generate_integrated_report(self, wartegg_data: Dict[str, Any], 
                                 tarot_selection: List[int] = None) -> IntegratedClinicalReport:
        """Genera un report clinico integrato completo"""
        # Se non specificato, usa tutte le 22 carte
        if tarot_selection is None:
            tarot_selection = list(range(22))  # Tutte e 22 le Lame Maggiori
        
        # Analisi Wartegg
        quadro_analyses = self._analyze_all_quadri(wartegg_data)
        
        # Analisi Tarocchi complete
        tarot_analyses = self._analyze_tarot_cards(tarot_selection)
        
        # Integrazione completa
        integrated_analyses = self._integrate_analyses(quadro_analyses, tarot_analyses)
        
        # Creazione report completo
        report = IntegratedClinicalReport(
            subject=wartegg_data.get('subject_name', 'Soggetto'),
            date=datetime.now().strftime('%d/%m/%Y'),
            system_version=self.system_version,
            analyst=self.analyst,
            
            adaptation_score=self._calculate_adaptation_score(wartegg_data),
            predominant_profile=self._determine_integrated_profile(wartegg_data, tarot_analyses),
            clinical_indicators=self._extract_clinical_indicators(wartegg_data),
            neural_networks=self._map_integrated_neural_networks(wartegg_data, tarot_analyses),
            clinical_alerts=self._identify_integrated_alerts(wartegg_data, tarot_analyses),
            tarot_archetype_patterns=self._identify_tarot_archetypes(tarot_analyses),
            
            quadro_analyses=integrated_analyses['wartegg'],
            tarot_analyses=integrated_analyses['tarot'],
            predominant_tarot_themes=self._analyze_tarot_themes(tarot_analyses),
            
            big_five=self._assess_integrated_big_five(wartegg_data, tarot_analyses),
            defense_mechanisms=self._identify_integrated_defenses(wartegg_data, tarot_analyses),
            neurochemical_balance=self._estimate_integrated_neurochemistry(wartegg_data, tarot_analyses),
            archetypal_patterns=self._identify_archetypal_patterns(wartegg_data, tarot_analyses),
            
            psychopathological_dynamics=self._identify_integrated_psychodynamics(wartegg_data, tarot_analyses),
            prognostic_factors=self._assess_integrated_prognostic_factors(wartegg_data, tarot_analyses),
            risk_scores=self._calculate_integrated_risk_scores(wartegg_data, tarot_analyses),
            integration_insights=self._generate_integration_insights(wartegg_data, tarot_analyses),
            
            treatment_phases=self._generate_integrated_treatment_plan(wartegg_data, tarot_analyses),
            neuro_protocols=self._recommend_integrated_neuro_protocols(wartegg_data, tarot_analyses),
            symbolic_interventions=self._recommend_symbolic_interventions(tarot_analyses),
            
            behavioral_observations=wartegg_data.get('behavioral_observations', []),
            drawing_behavior_patterns=self._analyze_drawing_behavior(wartegg_data),
            
            preliminary_diagnosis=self._determine_integrated_diagnosis(wartegg_data, tarot_analyses),
            immediate_actions=self._determine_integrated_actions(wartegg_data, tarot_analyses),
            prognosis=self._assess_integrated_prognosis(wartegg_data, tarot_analyses),
            spiritual_dimension=self._assess_spiritual_dimension(tarot_analyses)
        )
        
        return report

    def _analyze_tarot_cards(self, card_ids: List[int]) -> List[TarotCardAnalysis]:
        """Analizza le carte Tarocchi selezionate (tutte e 22 le Lame)"""
        analyses = []
        
        for card_id in card_ids:
            if 0 <= card_id <= 21:  # Tutte e 22 le Lame Maggiori (0-21)
                card_data = self.tarot_database[card_id]
                
                # Calcola il tono emotivo basato sul tipo di carta
                emotional_tone = self._calculate_emotional_tone(card_id)
                
                analysis = TarotCardAnalysis(
                    card_id=card_id,
                    card_name=card_data["name"],
                    italian_name=card_data["italian"],
                    arcano_type=ArcanoType.MAGGIORE,
                    drawing_style=self._determine_tarot_drawing_style(card_id),
                    line_quality=self._calculate_line_quality(card_id),
                    pressure_intensity=self._calculate_pressure_intensity(card_id),
                    symmetry_score=self._calculate_symmetry_score(card_id),
                    complexity_score=self._calculate_complexity_score(card_id),
                    symbolic_density=self._calculate_symbolic_density(card_id),
                    emotional_tone=emotional_tone,
                    wartegg_correlations=self._calculate_tarot_wartegg_correlations(card_id),
                    psychological_indicators=self._get_tarot_psychological_indicators(card_id),
                    neural_activation=self._estimate_tarot_neural_activation(card_id),
                    clinical_significance=self._assess_tarot_clinical_significance(card_id, emotional_tone),
                    interpretation=self._generate_tarot_interpretation(card_id)
                )
                analyses.append(analysis)
        
        return analyses

    def _calculate_emotional_tone(self, card_id: int) -> float:
        """Calcola il tono emotivo per ogni carta (-1 a +1)"""
        emotional_tones = {
            0: 0.2,    # Matto - neutro/leggero positivo
            1: 0.6,    # Bagatto - positivo
            2: 0.1,    # Papessa - neutro
            3: 0.8,    # Imperatrice - molto positivo
            4: 0.4,    # Imperatore - moderatamente positivo
            5: 0.5,    # Papa - neutro
            6: 0.8,    # Innamorati - molto positivo
            7: 0.6,    # Carro - positivo
            8: 0.3,    # Giustizia - neutro
            9: 0.2,    # Eremita - neutro/leggero positivo
            10: 0.4,   # Ruota - moderatamente positivo
            11: 0.7,   # Forza - positivo
            12: -0.2,  # Appeso - leggero negativo
            13: -0.5,  # Morte - negativo
            14: 0.6,   # Temperanza - positivo
            15: -0.6,  # Diavolo - negativo
            16: -0.7,  # Torre - molto negativo
            17: 0.7,   # Stelle - positivo
            18: -0.3,  # Luna - leggero negativo
            19: 0.9,   # Sole - molto positivo
            20: 0.5,   # Giudizio - moderatamente positivo
            21: 0.8    # Mondo - molto positivo
        }
        return emotional_tones.get(card_id, 0.0)

    def _get_tarot_psychological_indicators(self, card_id: int) -> List[str]:
        """Restituisce gli indicatori psicologici per ogni carta"""
        indicators = {
            0: ["spontaneità", "inconsapevolezza", "libertà", "rischio"],
            1: ["volontà", "concentrazione", "abilità", "manifestazione"],
            2: ["intuizione", "mistero", "subconscio", "saggezza interiore"],
            3: ["creatività", "naturalezza", "abbondanza", "fertilità"],
            4: ["struttura", "autorità", "controllo", "ordine"],
            5: ["tradizione", "conoscenza", "guida", "spiritualità organizzata"],
            6: ["scelta", "armonia", "relazione", "unione"],
            7: ["controllo", "determinazione", "progresso", "vittoria"],
            8: ["equilibrio", "decisione", "karma", "giustizia"],
            9: ["introspezione", "saggezza", "solitudine", "guida interiore"],
            10: ["ciclicità", "destino", "cambiamento", "fortuna"],
            11: ["coraggio", "passione", "controllo interiore", "forza morale"],
            12: ["sospensione", "sacrificio", "nuova prospettiva", "resa"],
            13: ["trasformazione", "fine", "rinascita", "cambiamento radicale"],
            14: ["equilibrio", "moderazione", "armonizzazione", "pazienza"],
            15: ["attaccamento", "materialismo", "schiavitù interiore", "tentazione"],
            16: ["caos", "distruzione", "rivelazione improvvisa", "crollo"],
            17: ["speranza", "ispirazione", "guida spirituale", "fiducia"],
            18: ["illusione", "subconscio", "intuizione notturna", "sogno"],
            19: ["gioia", "successo", "illuminazione", "vitalità"],
            20: ["valutazione", "risveglio", "chiamata interiore", "resurrezione"],
            21: ["completezza", "realizzazione", "unità", "successo totale"]
        }
        return indicators.get(card_id, [])

    # ... (altri metodi di supporto per l'analisi completa)

    def _generate_integration_insights(self, wartegg_data: Dict[str, Any], 
                                     tarot_analyses: List[TarotCardAnalysis]) -> List[str]:
        """Genera insight dall'integrazione completa dei due sistemi"""
        insights = []
        
        # Trova le carte dominanti
        dominant_cards = sorted(tarot_analyses, key=lambda x: abs(x.emotional_tone), reverse=True)[:3]
        
        insights.append(f"Archetipi dominanti: {', '.join([c.italian_name for c in dominant_cards])}")
        
        # Analisi per elemento
        elements = {"aria": 0, "fuoco": 0, "acqua": 0, "terra": 0}
        for card in tarot_analyses:
            element = self.tarot_database[card.card_id]["element"]
            elements[element] += 1
        
        dominant_element = max(elements.items(), key=lambda x: x[1])
        insights.append(f"Elemento predominante: {dominant_element[0]} ({dominant_element[1]} carte)")
        
        # Correlazioni specifiche
        if any(card.card_id == 15 for card in dominant_cards):  # Diavolo
            insights.append("Presenza dell'archetipo del Diavolo suggerisce lavoro su attaccamenti e ombre")
        
        if any(card.card_id == 16 for card in dominant_cards):  # Torre
            insights.append("La Torre indica possibili crisi trasformative o crolli strutturali")
        
        if any(card.card_id == 21 for card in dominant_cards):  # Mondo
            insights.append("Il Mondo presente indica potenziale di realizzazione e completezza")
        
        return insights

# Esempio di utilizzo completo
if __name__ == "__main__":
    # Carica i dati Wartegg (dall'analisi esistente)
    with open("wartegg_analysis.json", "r") as f:
        wartegg_data = json.load(f)
    
    # Genera il report integrato con TUTTE le 22 Lame
    generator = IntegratedReportGenerator()
    report = generator.generate_integrated_report(wartegg_data)  # Nessuna selezione = tutte le carte
    
    # Salva il report completo
    report.to_json("complete_integrated_wartegg_tarot_report.json")
    print("Report integrato completo con 22 Lame generato con successo!")