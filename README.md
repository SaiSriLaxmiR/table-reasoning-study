# Table Reasoning Study

> Do large language models actually read tables, or do they answer from memory?

A counterfactual perturbation study measuring how often LLMs ground their answers on provided table data versus relying on memorized knowledge. Built as a precursor to **TableGuard** — a runtime hallucination detector for table-grounded AI responses.

---

## Research Question

When an LLM is given a Wikipedia info-box and a question about it, does its answer change when the table's values are deliberately altered?

```
If the model reads the TABLE  → answer changes when table changes  ✓
If the model uses MEMORY      → answer stays the same despite change ✗
```

This is a **counterfactual perturbation test** — the same methodology used in InfoTabS (Gupta et al., 2020) for adversarial tabular NLI evaluation.

---

## Key Findings

| Finding | Result |
|---|---|
| Overall table grounding rate (PSR) | **67.6%** — model uses table 71/105 times |
| Named entity questions | **97.2%** — nearly always reads the table |
| Relational inference questions | **40.0%** — ignores table 60% of the time |
| Famous articles (Einstein, France) | **60.0%** PSR |
| Obscure articles (Eswatini, BRAC) | **77.8%** PSR |
| Familiarity gap | **17.8pp** — memorization competes with grounding |
| Memory override rate | **7.6%** — model actively corrects table from memory |

**The core finding:** LLaMA 3.3 70B grounds on Wikipedia tables only two-thirds of the time. Relational inference is the weakest point — the model ignores the table in 60% of relational questions. A 17.8 percentage point gap between famous and obscure subjects confirms that memorized knowledge actively competes with table grounding.

---

## Dataset

- **35 Wikipedia infoboxes** across 5 categories
- **3 questions per table** = 105 question-answer pairs
- **3 perturbation types** per question

| Category | Tables | Famous | Obscure |
|---|---|---|---|
| Scientists | 10 | Einstein, Curie, Bohr... | Chandrasekhar, Noether... |
| Countries | 9 | France, Brazil, Japan... | Bhutan, Vanuatu, Eswatini... |
| Films | 7 | The Godfather, Parasite... | Lagaan, Bicycle Thieves... |
| Organisations | 9 | Google, UNESCO, SpaceX... | Grameen Bank, BRAC... |

---

## Perturbation Types

Three types of table modifications, chosen to mirror InfoTabS's α1/α2/α3 test set design:

| Type | Example | PSR |
|---|---|---|
| **Numerical** | Children: 3 → 4 | 64.7% |
| **Named entity** | Director: Coppola → Scorsese | 97.2% |
| **Relational** | Yes/No fields, comparisons | 40.0% |

---

## Experimental Design

For each table + question pair, the model is queried under three conditions:

```
Condition 1: Original table + question   → should answer correctly
Condition 2: Perturbed table + question  → should change answer if using table
Condition 3: No table + question         → reveals memorized baseline
```

**Perturbation Sensitivity Rate (PSR):**
```
PSR = (answers that changed when table changed) / total × 100
```

---

## Pipeline

```
collect_tables.py      → 50 Wikipedia articles → 35 successful infoboxes
       ↓
generate_questions.py  → 3 questions per table via Groq LLaMA
       ↓
perturb_tables.py      → wrong values per question field via Groq
       ↓
evaluate_models.py     → 3 conditions × 105 questions → raw_results.json
       ↓
analyse_results.py     → PSR tables, familiarity gap, category breakdown
```

---

## Results

### Overall PSR
```
llama-3.3-70b: PSR = 67.6% (71/105 used table)
```

### By Question Type
```
Named entity:  97.2%   ← model reads table reliably for names
Numerical:     64.7%   ← partial grounding
Relational:    40.0%   ← model ignores table more than half the time
```

### By Familiarity
```
Famous articles:  60.0%
Obscure articles: 77.8%
Gap:             +17.8pp  ← memorization competes with table grounding
```

### By Category
```
Organisations: 85.2%
Scientists:    73.3%
Countries:     63.0%
Films:         42.9%
```

---

## Quickstart

```bash
git clone https://github.com/SaiSriLaxmiR/table-reasoning-study
cd table-reasoning-study
python3 -m venv venv
source venv/bin/activate
pip install requests langchain-groq langchain-core python-dotenv
```

Create `.env`:
```
GROQ_API_KEY=your-key-here
```

Run the full pipeline:
```bash
python collect_tables.py
python generate_questions.py
python perturb_tables.py
python evaluate_models.py
python analyse_results.py
```

---

## Project Structure

```
table-reasoning-study/
├── collect_tables.py       # Wikipedia API → infobox extraction
├── generate_questions.py   # LLM-generated questions per table
├── perturb_tables.py       # Counterfactual value generation
├── evaluate_models.py      # 3-condition model evaluation with resume
├── analyse_results.py      # PSR metrics and result tables
├── data/
│   ├── tables_raw.json
│   ├── tables_with_questions.json
│   └── tables_perturbed.json
└── results/
    ├── raw_results.json
    └── findings.csv
```

---

## Next: TableGuard

These findings motivate **TableGuard** — a runtime hallucination detector that uses perturbation sensitivity as a grounding signal for production AI systems. Instead of measuring the problem, TableGuard detects it in real time and returns a grounding certificate alongside every table-based answer.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B |
| Table source | Wikipedia Action API |
| Question generation | Groq LLaMA 3.3 70B |
| Perturbation generation | Groq LLaMA 3.3 70B |
| Framework | LangChain + python-dotenv |
| Output | JSON + CSV |

---

## License

MIT
