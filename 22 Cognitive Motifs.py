import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum

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

@dataclass
class TarotCardAnalysis:
    card_id: int
    card_name: str
    italian_name: str
    arcano_type: ArcanoType
    drawing_style: DrawingStyle
    line_quality: float  # 0-1
    pressure_intensity: float  # 0-1
    symmetry_score: float  # 0-1
    complexity_score: float  # 0-1
    symbolic_density: float  # 0-1
    emotional_tone: float  # -1 to 1 (negative to positive)
    wartegg_correlations: Dict[int, float]  # Correlation with Wartegg frames
    psychological_indicators: List[str]
    neural_activation: Dict[str, float]
    
    def to_dict(self):
        return {
            "card_id": self.card_id,
            "card_name": self.card_name,
            "italian_name": self.italian_name,
            "arcano_type": self.arcano_type.value,
            "drawing_style": self.drawing_style.value,
            "line_quality": self.line_quality,
            "pressure_intensity": self.pressure_intensity,
            "symmetry_score": self.symmetry_score,
            "complexity_score": self.complexity_score,
            "symbolic_density": self.symbolic_density,
            "emotional_tone": self.emotional_tone,
            "wartegg_correlations": self.wartegg_correlations,
            "psychological_indicators": self.psychological_indicators,
            "neural_activation": self.neural_activation
        }

# Analisi completa delle 22 Lame dei Tarocchi
TAROT_CARD_ANALYSES = {
    0: TarotCardAnalysis(
        card_id=0,
        card_name="The Fool",
        italian_name="Il Matto",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.ORGANIC,
        line_quality=0.6,
        pressure_intensity=0.4,
        symmetry_score=0.3,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.2,
        wartegg_correlations={1: 0.7, 9: 0.6, 16: 0.5},
        psychological_indicators=["spontaneità", "inconsapevolezza", "libertà"],
        neural_activation={"insula": 0.7, "vmPFC": 0.6, "ACC": 0.5}
    ),
    
    1: TarotCardAnalysis(
        card_id=1,
        card_name="The Magician",
        italian_name="Il Bagatto",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.8,
        pressure_intensity=0.7,
        symmetry_score=0.6,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=0.6,
        wartegg_correlations={3: 0.8, 4: 0.7, 11: 0.6},
        psychological_indicators=["volontà", "concentrazione", "abilità"],
        neural_activation={"dlPFC": 0.8, "parietal": 0.7, "motor_cortex": 0.6}
    ),
    
    2: TarotCardAnalysis(
        card_id=2,
        card_name="The High Priestess",
        italian_name="La Papessa",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.5,
        symmetry_score=0.8,
        complexity_score=0.6,
        symbolic_density=0.9,
        emotional_tone=0.1,
        wartegg_correlations={2: 0.9, 5: 0.7, 13: 0.6},
        psychological_indicators=["intuizione", "mistero", "subconscio"],
        neural_activation={"vmPFC": 0.7, "temporale": 0.6, "insula": 0.5}
    ),
    
    3: TarotCardAnalysis(
        card_id=3,
        card_name="The Empress",
        italian_name="L'Imperatrice",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.ORGANIC,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.7,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.8,
        wartegg_correlations={5: 0.8, 6: 0.7, 7: 0.6},
        psychological_indicators=["creatività", "naturalezza", "abbondanza"],
        neural_activation={"vmPFC": 0.8, "insula": 0.7, "ACC": 0.6}
    ),
    
    4: TarotCardAnalysis(
        card_id=4,
        card_name="The Emperor",
        italian_name="L'Imperatore",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.GEOMETRIC,
        line_quality=0.8,
        pressure_intensity=0.8,
        symmetry_score=0.9,
        complexity_score=0.6,
        symbolic_density=0.7,
        emotional_tone=0.4,
        wartegg_correlations={4: 0.9, 8: 0.7, 14: 0.6},
        psychological_indicators=["struttura", "autorità", "controllo"],
        neural_activation={"dlPFC": 0.8, "parietal": 0.7, "motor_cortex": 0.6}
    ),
    
    5: TarotCardAnalysis(
        card_id=5,
        card_name="The Pope",
        italian_name="Il Papa",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.8,
        complexity_score=0.7,
        symbolic_density=0.9,
        emotional_tone=0.5,
        wartegg_correlations={8: 0.8, 12: 0.7, 16: 0.6},
        psychological_indicators=["tradizione", "conoscenza", "guida"],
        neural_activation={"TPJ": 0.7, "pSTS": 0.6, "precuneus": 0.5}
    ),
    
    6: TarotCardAnalysis(
        card_id=6,
        card_name="The Lovers",
        italian_name="Gli Innamorati",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.FIGURATIVE,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.7,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.8,
        wartegg_correlations={6: 0.9, 10: 0.7, 15: 0.6},
        psychological_indicators=["scelta", "armonia", "relazione"],
        neural_activation={"vmPFC": 0.8, "insula": 0.7, "ACC": 0.6}
    ),
    
    7: TarotCardAnalysis(
        card_id=7,
        card_name="The Chariot",
        italian_name="Il Carro",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.GEOMETRIC,
        line_quality=0.8,
        pressure_intensity=0.8,
        symmetry_score=0.8,
        complexity_score=0.7,
        symbolic_density=0.7,
        emotional_tone=0.6,
        wartegg_correlations={7: 0.8, 11: 0.7, 16: 0.6},
        psychological_indicators=["controllo", "determinazione", "progresso"],
        neural_activation={"dlPFC": 0.8, "motor_cortex": 0.7, "cerebellum": 0.6}
    ),
    
    8: TarotCardAnalysis(
        card_id=8,
        card_name="Justice",
        italian_name="La Giustizia",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.GEOMETRIC,
        line_quality=0.8,
        pressure_intensity=0.7,
        symmetry_score=0.9,
        complexity_score=0.6,
        symbolic_density=0.7,
        emotional_tone=0.3,
        wartegg_correlations={8: 0.9, 12: 0.7, 14: 0.6},
        psychological_indicators=["equilibrio", "decisione", "karma"],
        neural_activation={"dlPFC": 0.8, "parietal": 0.7, "TPJ": 0.6}
    ),
    
    9: TarotCardAnalysis(
        card_id=9,
        card_name="The Hermit",
        italian_name="L'Eremita",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.5,
        symmetry_score=0.6,
        complexity_score=0.6,
        symbolic_density=0.8,
        emotional_tone=0.2,
        wartegg_correlations={9: 0.9, 13: 0.7, 16: 0.6},
        psychological_indicators=["introspezione", "saggezza", "solitudine"],
        neural_activation={"dmPFC": 0.7, "precuneus": 0.6, "insula": 0.5}
    ),
    
    10: TarotCardAnalysis(
        card_id=10,
        card_name="Wheel of Fortune",
        italian_name="La Ruota della Fortuna",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.8,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=0.4,
        wartegg_correlations={10: 0.9, 15: 0.7, 16: 0.6},
        psychological_indicators=["ciclicità", "destino", "cambiamento"],
        neural_activation={"parietal": 0.7, "TPJ": 0.6, "insula": 0.5}
    ),
    
    11: TarotCardAnalysis(
        card_id=11,
        card_name="Strength",
        italian_name="La Forza",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.FIGURATIVE,
        line_quality=0.7,
        pressure_intensity=0.7,
        symmetry_score=0.6,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.7,
        wartegg_correlations={11: 0.9, 7: 0.7, 4: 0.6},
        psychological_indicators=["coraggio", "passione", "controllo interiore"],
        neural_activation={"ACC": 0.8, "insula": 0.7, "amygdala": 0.6}
    ),
    
    12: TarotCardAnalysis(
        card_id=12,
        card_name="The Hanged Man",
        italian_name="L'Impiccato",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.6,
        pressure_intensity=0.4,
        symmetry_score=0.4,
        complexity_score=0.6,
        symbolic_density=0.8,
        emotional_tone=-0.2,
        wartegg_correlations={12: 0.9, 9: 0.7, 16: 0.6},
        psychological_indicators=["sospensione", "sacrificio", "nuova prospettiva"],
        neural_activation={"insula": 0.7, "vmPFC": 0.6, "ACC": 0.5}
    ),
    
    13: TarotCardAnalysis(
        card_id=13,
        card_name="Death",
        italian_name="La Morte",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.5,
        complexity_score=0.7,
        symbolic_density=0.9,
        emotional_tone=-0.5,
        wartegg_correlations={13: 0.9, 8: 0.7, 16: 0.6},
        psychological_indicators=["trasformazione", "fine", "rinascita"],
        neural_activation={"amygdala": 0.8, "insula": 0.7, "ACC": 0.6}
    ),
    
    14: TarotCardAnalysis(
        card_id=14,
        card_name="Temperance",
        italian_name="La Temperanza",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.FIGURATIVE,
        line_quality=0.7,
        pressure_intensity=0.6,
        symmetry_score=0.8,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.6,
        wartegg_correlations={14: 0.9, 6: 0.7, 10: 0.6},
        psychological_indicators=["equilibrio", "moderazione", "armonizzazione"],
        neural_activation={"vmPFC": 0.7, "insula": 0.6, "ACC": 0.5}
    ),
    
    15: TarotCardAnalysis(
        card_id=15,
        card_name="The Devil",
        italian_name="Il Diavolo",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.7,
        symmetry_score=0.6,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=-0.6,
        wartegg_correlations={15: 0.9, 5: 0.7, 11: 0.6},
        psychological_indicators=["attaccamento", "materialismo", "schiavitù interiore"],
        neural_activation={"amygdala": 0.8, "striatum": 0.7, "vmPFC": 0.6}
    ),
    
    16: TarotCardAnalysis(
        card_id=16,
        card_name="The Tower",
        italian_name="La Torre",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.GEOMETRIC,
        line_quality=0.6,
        pressure_intensity=0.8,
        symmetry_score=0.3,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=-0.7,
        wartegg_correlations={16: 0.9, 13: 0.7, 10: 0.6},
        psychological_indicators=["caos", "distruzione", "rivelazione improvvisa"],
        neural_activation={"amygdala": 0.9, "insula": 0.8, "ACC": 0.7}
    ),
    
    17: TarotCardAnalysis(
        card_id=17,
        card_name="The Stars",
        italian_name="Le Stelle",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.5,
        symmetry_score=0.7,
        complexity_score=0.7,
        symbolic_density=0.8,
        emotional_tone=0.7,
        wartegg_correlations={17: 0.9, 2: 0.7, 9: 0.6},
        psychological_indicators=["speranza", "ispirazione", "guida spirituale"],
        neural_activation={"vmPFC": 0.7, "precuneus": 0.6, "insula": 0.5}
    ),
    
    18: TarotCardAnalysis(
        card_id=18,
        card_name="The Moon",
        italian_name="La Luna",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.6,
        pressure_intensity=0.5,
        symmetry_score=0.5,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=-0.3,
        wartegg_correlations={18: 0.9, 2: 0.7, 13: 0.6},
        psychological_indicators=["illusione", "subconscio", "intuizione notturna"],
        neural_activation={"insula": 0.8, "vmPFC": 0.7, "amygdala": 0.6}
    ),
    
    19: TarotCardAnalysis(
        card_id=19,
        card_name="The Sun",
        italian_name="Il Sole",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.FIGURATIVE,
        line_quality=0.8,
        pressure_intensity=0.7,
        symmetry_score=0.8,
        complexity_score=0.7,
        symbolic_density=0.7,
        emotional_tone=0.9,
        wartegg_correlations={19: 0.9, 6: 0.7, 14: 0.6},
        psychological_indicators=["gioia", "successo", "illuminazione"],
        neural_activation={"vmPFC": 0.8, "accumbens": 0.7, "insula": 0.6}
    ),
    
    20: TarotCardAnalysis(
        card_id=20,
        card_name="Judgement",
        italian_name="Il Giudizio",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.7,
        pressure_intensity=0.7,
        symmetry_score=0.7,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=0.5,
        wartegg_correlations={20: 0.9, 8: 0.7, 16: 0.6},
        psychological_indicators=["valutazione", "risveglio", "chiamata interiore"],
        neural_activation={"dlPFC": 0.7, "TPJ": 0.6, "ACC": 0.5}
    ),
    
    21: TarotCardAnalysis(
        card_id=21,
        card_name="The World",
        italian_name="Il Mondo",
        arcano_type=ArcanoType.MAGGIORE,
        drawing_style=DrawingStyle.SYMBOLIC,
        line_quality=0.8,
        pressure_intensity=0.6,
        symmetry_score=0.9,
        complexity_score=0.8,
        symbolic_density=0.9,
        emotional_tone=0.8,
        wartegg_correlations={21: 0.9, 16: 0.7, 10: 0.6},
        psychological_indicators=["completezza", "realizzazione", "unità"],
        neural_activation={"vmPFC": 0.8, "precuneus": 0.7, "insula": 0.6}
    )
}

# Funzione per integrare l'analisi dei Tarocchi nel sistema Wartegg
def integrate_tarot_analysis(wartegg_analysis: Dict[str, Any], tarot_cards: List[int] = None) -> Dict[str, Any]:
    """
    Integra l'analisi dei Tarocchi con i risultati del test Wartegg
    
    Args:
        wartegg_analysis: Risultati dell'analisi Wartegg
        tarot_cards: Lista di carte da analizzare (se None, analizza tutte)
    
    Returns:
        Dizionario con analisi integrata
    """
    if tarot_cards is None:
        tarot_cards = list(TAROT_CARD_ANALYSES.keys())
    
    # Analisi delle carte selezionate
    tarot_analyses = {}
    for card_id in tarot_cards:
        if card_id in TAROT_CARD_ANALYSES:
            tarot_analyses[card_id] = TAROT_CARD_ANALYSES[card_id].to_dict()
    
    # Correlazioni con i quadri Wartegg
    wartegg_correlations = {}
    for w_frame in range(1, 17):
        correlations = {}
        for card_id, card_analysis in TAROT_CARD_ANALYSES.items():
            if w_frame in card_analysis.wartegg_correlations:
                correlations[card_id] = card_analysis.wartegg_correlations[w_frame]
        if correlations:
            wartegg_correlations[w_frame] = correlations
    
    # Crea report integrato
    integrated_report = {
        "wartegg_analysis": wartegg_analysis,
        "tarot_analyses": tarot_analyses,
        "wartegg_tarot_correlations": wartegg_correlations,
        "integration_timestamp": datetime.now().isoformat(),
        "interpretation_notes": generate_integrated_interpretation(wartegg_analysis, tarot_analyses)
    }
    
    return integrated_report

def generate_integrated_interpretation(wartegg_analysis: Dict[str, Any], tarot_analyses: Dict[int, Any]) -> List[str]:
    """Genera note interpretative integrate"""
    notes = []
    
    # Analizza correlazioni emotive
    emotional_tones = [card['emotional_tone'] for card in tarot_analyses.values()]
    avg_emotional_tone = sum(emotional_tones) / len(emotional_tones)
    
    if avg_emotional_tone > 0.5:
        notes.append("Tono emotivo generale positivo: tendenza all'ottimismo e all'apertura")
    elif avg_emotional_tone < -0.3:
        notes.append("Tono emotivo generale negativo: possibili difficoltà emotive o conflitti interiori")
    else:
        notes.append("Tono emotivo bilanciato: equilibrio tra positività e consapevolezza delle sfide")
    
    # Analizza stili di disegno predominanti
    drawing_styles = {}
    for card in tarot_analyses.values():
        style = card['drawing_style']
        drawing_styles[style] = drawing_styles.get(style, 0) + 1
    
    predominant_style = max(drawing_styles.items(), key=lambda x: x[1])[0] if drawing_styles else "misto"
    notes.append(f"Stile grafico predominante: {predominant_style}")
    
    # Correlazioni con i risultati Wartegg
    if 'personality_profile' in wartegg_analysis:
        personality = wartegg_analysis['personality_profile']['big_five']
        
        # Correlazione tra apertura mentale e complessità simbolica
        complexity_scores = [card['complexity_score'] for card in tarot_analyses.values()]
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        if personality['openness'] > 0.7 and avg_complexity > 0.7:
            notes.append("Alta apertura mentale correlata con complessità simbolica: capacità di elaborazione astratta avanzata")
        elif personality['openness'] < 0.4 and avg_complexity > 0.7:
            notes.append("Bassa apertura mentale ma alta complessità simbolica: possibile conflitto tra concretezza e simbolismo")
    
    return notes

# Aggiornamento del codice di lettura Wartegg W-16
def enhanced_wartegg_analysis_with_tarot(test_data: Dict[int, Dict[str, Any]], 
                                       tarot_cards: List[int] = None) -> Dict[str, Any]:
    """
    Versione avanzata dell'analisi Wartegg con integrazione Tarocchi
    """
    # Esegui l'analisi Wartegg standard
    analyzer = WarteggRodellaUltimateAnalyzerV2()
    wartegg_results = analyzer.analyze_full_test_with_validation(test_data)
    
    # Integra con l'analisi dei Tarocchi
    integrated_results = integrate_tarot_analysis(wartegg_results, tarot_cards)
    
    return integrated_results

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica i dati di esempio (disegni Wartegg)
    sample_drawings = load_or_create_drawings()
    
    # Estrai features
    features = {}
    for qid, img_path in sample_drawings.items():
        features[qid] = extract_enhanced_features_single(img_path)
    
    # Analisi avanzata con integrazione Tarocchi
    test_data = {qid: {"features": feat} for qid, feat in features.items()}
    results = enhanced_wartegg_analysis_with_tarot(test_data)
    
    # Salva i risultati
    with open("wartegg_tarot_integrated_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Analisi completata con integrazione Tarocchi. Risultati salvati in 'wartegg_tarot_integrated_analysis.json'")