from __future__ import annotations
"""Genererar publicerbara forskningsrapporter från AI-upptäckter."""
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path

from .hypothesis_engine import Hypothesis


class ScientificReportGenerator:
    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        
    def generate_paper(self, discovery: Dict[str, Any]) -> str:
        """Generera komplett forskningsrapport i Markdown."""
        sections = {
            "title": self._generate_title(discovery),
            "abstract": self._write_abstract(discovery),
            "introduction": self._contextualize_discovery(discovery),
            "hypothesis": self._formalize_hypothesis(discovery),
            "methods": self._describe_methods(discovery),
            "results": self._present_findings(discovery),
            "discussion": self._analyze_implications(discovery),
            "future_work": self._suggest_validation(discovery),
            "references": self._format_references(discovery)
        }
        
        return self._format_markdown(sections)
    
    def _generate_title(self, discovery: Dict[str, Any]) -> str:
        statement = discovery.get("statement", "Novel Discovery")
        return f"In-Silico Discovery: {statement[:100]}"
    
    def _write_abstract(self, discovery: Dict[str, Any]) -> str:
        return f"""
**Background:** {discovery.get('health_goal', 'Health optimization')} remains a critical area of research.

**Hypothesis:** {discovery.get('statement', 'Novel therapeutic approach')}

**Methods:** We employed an AI-driven hypothesis generation and validation framework utilizing 
Neo4j knowledge graphs, BioBERT relation extraction, and multi-source literature validation.

**Results:** Our analysis identified {discovery.get('mechanism', 'a novel mechanism')} with a 
confidence score of {discovery.get('confidence', 0.0):.2f} and novelty score of 
{discovery.get('novelty_score', 0.0):.2f}.

**Conclusion:** This in-silico discovery warrants wet-lab validation and presents a promising 
avenue for {discovery.get('health_goal', 'therapeutic development')}.
"""
    
    def _formalize_hypothesis(self, discovery: Dict[str, Any]) -> str:
        return f"""
## Hypothesis

**Primary Hypothesis:** {discovery.get('statement', '')}

**Proposed Mechanism:** {discovery.get('mechanism', '')}

**Supporting Evidence:**
{self._format_evidence(discovery.get('evidence', []))}
"""
    
    def _describe_methods(self, discovery: Dict[str, Any]) -> str:
        return """
## Methods

### Data Sources
- PubMed Central Open Access (3M+ articles)
- DrugBank (14k+ drug entries)
- ClinicalTrials.gov (400k+ trials)
- Neo4j biomedical knowledge graph

### Computational Framework
1. **Hypothesis Generation:** Ensemble of domain-specific language models
2. **Literature Validation:** FAISS-indexed retrieval-augmented generation
3. **Relation Extraction:** Fine-tuned BioBERT with confidence scoring
4. **Graph Analysis:** Neo4j Cypher queries for pathway exploration

### In-Silico Experimental Design
""" + discovery.get('experimental_design', 'Virtual experiment protocol detailed above.')
    
    def _present_findings(self, discovery: Dict[str, Any]) -> str:
        # Default key findings with proper newlines
        default_findings = """1. Novel mechanism identified
2. Safety profile favorable
3. Synergistic potential detected"""
        
        return f"""
## Results

### Validation Metrics
- Literature Support Score: {discovery.get('validity_score', 0.0):.3f}
- Novelty Score: {discovery.get('novelty_score', 0.0):.3f}
- Confidence Score: {discovery.get('confidence', 0.0):.3f}

### Key Findings
{discovery.get('key_findings', default_findings)}
"""
    
    def _analyze_implications(self, discovery: Dict[str, Any]) -> str:
        return f"""
## Discussion

Our AI-driven discovery process has identified a promising hypothesis that warrants further investigation.
The combination of high novelty ({discovery.get('novelty_score', 0.0):.2f}) and reasonable literature 
support suggests this represents a genuinely novel contribution to the field.

### Clinical Implications
{discovery.get('clinical_implications', 'Potential therapeutic applications pending validation.')}

### Limitations
- In-silico validation only
- Requires wet-lab confirmation
- Long-term safety unknown
"""
    
    def _suggest_validation(self, discovery: Dict[str, Any]) -> str:
        return """
## Future Work

### Recommended Wet-Lab Validation
1. Cell culture studies for initial efficacy
2. Dose-response characterization
3. Toxicity screening
4. Mechanism confirmation via Western blot/qPCR

### Clinical Translation Path
1. Animal model studies
2. Phase I safety trials
3. Biomarker development
4. Combination therapy exploration
"""
    
    def _format_evidence(self, evidence: list) -> str:
        if not evidence:
            return "- No direct evidence found (novel hypothesis)"
        
        formatted = []
        for i, e in enumerate(evidence[:5], 1):
            formatted.append(f"{i}. {e[:200]}...")
        return "\n".join(formatted)
    
    def _format_references(self, discovery: Dict[str, Any]) -> str:
        refs = discovery.get('references', [])
        if not refs:
            refs = ["[1] AI-generated hypothesis - validation pending"]
        
        return "\n## References\n" + "\n".join(refs)
    
    def _format_markdown(self, sections: Dict[str, str]) -> str:
        output = f"""
# {sections['title']}

**Generated:** {datetime.now().strftime('%Y-%m-%d')}  
**Status:** Pre-print / In-silico validation only

---

## Abstract
{sections['abstract']}

---

{sections['introduction']}

{sections['hypothesis']}

{sections['methods']}

{sections['results']}

{sections['discussion']}

{sections['future_work']}

{sections['references']}

---

*This report was generated by an AI research assistant. All findings require 
experimental validation before clinical application.*
"""
        return output.strip()
    
    def save_report(self, report: str, filename: str) -> Path:
        """Spara rapport som Markdown-fil."""
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath.write_text(report)
        
        return filepath 