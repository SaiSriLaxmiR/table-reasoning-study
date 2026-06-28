"""
analyse_results.py
------------------
Compute all metrics and generate the research findings.

Metrics computed:
  1. Perturbation Sensitivity Rate (PSR) per model per question type
  2. PSR by familiarity (famous vs obscure articles)
  3. PSR by table category
  4. Memory override rate (how often model corrects the table from memory)
  5. Error analysis — what kinds of questions fool each model most

Run after evaluate_models.py:
    python analyse_results.py
"""

import json
import csv
from collections import defaultdict

def load_results():
    with open("results/raw_results.json") as f:
        return json.load(f)

def compute_psr(results: list, model: str,
                filter_fn=None) -> dict:
    """
    Compute Perturbation Sensitivity Rate for a model.
    filter_fn: optional lambda to filter results subset
    Returns {"psr": float, "n": int, "used_table": int}
    """
    subset = [r for r in results if filter_fn is None or filter_fn(r)]
    n = len(subset)
    if n == 0:
        return {"psr": None, "n": 0, "used_table": 0}
    used = sum(
        1 for r in subset
        if r["model_responses"].get(model, {}).get("used_table", False)
    )
    return {"psr": round(used / n * 100, 1), "n": n, "used_table": used}

def compute_memory_override_rate(results, model):
    """
    Memory override = model gave the ORIGINAL correct answer
    even when shown the PERTURBED table.
    i.e., it ignored the table and answered from memory.
    """
    n = 0
    overrides = 0
    for r in results:
        resp = r["model_responses"].get(model, {})
        pert_ans = resp.get("perturbed_answer", "").lower()
        orig_val = r.get("original_value", "").lower()
        if orig_val and pert_ans:
            n += 1
            # Model gave original (memorized) answer despite perturbed table
            if orig_val[:10] in pert_ans or pert_ans in orig_val[:20]:
                overrides += 1
    return round(overrides / n * 100, 1) if n else None

def analyse(results: list):
    models = list(results[0]["model_responses"].keys()) if results else []

    print("=" * 65)
    print("TABLE REASONING STUDY — RESULTS")
    print("=" * 65)
    print(f"Total questions: {len(results)}")
    print(f"Models evaluated: {models}")
    print()

    # ── Table 1: Overall PSR per model ──────────────────────────────────────
    print("TABLE 1 — Overall Perturbation Sensitivity Rate (PSR)")
    print("(PSR = % of time model's answer changed when table was perturbed)")
    print(f"\n{'Model':<25} {'PSR':>8} {'Used Table':>12} {'Total':>8}")
    print("-" * 57)
    for model in models:
        r = compute_psr(results, model)
        print(f"{model:<25} {str(r['psr'])+'%':>8} {r['used_table']:>12} {r['n']:>8}")

    # ── Table 2: PSR by question type ───────────────────────────────────────
    print("\nTABLE 2 — PSR by Question Type")
    print(f"\n{'Model':<25} {'Numerical':>12} {'Named Entity':>14} {'Relational':>12}")
    print("-" * 67)
    for model in models:
        num = compute_psr(results, model, lambda r: r["question_type"] == "numerical")
        ne  = compute_psr(results, model, lambda r: r["question_type"] == "named_entity")
        rel = compute_psr(results, model, lambda r: r["question_type"] == "relational")
        print(f"{model:<25} {str(num['psr'])+'%':>12} {str(ne['psr'])+'%':>14} {str(rel['psr'])+'%':>12}")

    # ── Table 3: PSR by familiarity ─────────────────────────────────────────
    print("\nTABLE 3 — PSR by Article Familiarity")
    print("(Famous = LLM likely memorized; Obscure = must use table)")
    print(f"\n{'Model':<25} {'Famous':>10} {'Obscure':>10} {'Difference':>12}")
    print("-" * 60)
    for model in models:
        fam = compute_psr(results, model, lambda r: r["familiarity"] == "famous")
        obs = compute_psr(results, model, lambda r: r["familiarity"] == "obscure")
        diff = round((obs["psr"] or 0) - (fam["psr"] or 0), 1)
        print(f"{model:<25} {str(fam['psr'])+'%':>10} {str(obs['psr'])+'%':>10} {str(diff)+'pp':>12}")
    print("  Interpretation: higher PSR for obscure = model uses table when it can't remember")

    # ── Table 4: PSR by category ─────────────────────────────────────────────
    print("\nTABLE 4 — PSR by Table Category")
    categories = list(set(r["category"] for r in results))
    header = f"{'Model':<25} " + " ".join(f"{c:>14}" for c in categories)
    print(f"\n{header}")
    print("-" * (25 + 15 * len(categories)))
    for model in models:
        row = f"{model:<25}"
        for cat in categories:
            r = compute_psr(results, model, lambda r: r["category"] == cat)
            row += f" {str(r['psr'])+'%':>14}"
        print(row)

    # ── Table 5: Memory override rate ────────────────────────────────────────
    print("\nTABLE 5 — Memory Override Rate")
    print("(% of time model gave the ORIGINAL answer despite perturbed table)")
    print(f"\n{'Model':<25} {'Override Rate':>15}")
    print("-" * 42)
    for model in models:
        rate = compute_memory_override_rate(results, model)
        print(f"{model:<25} {str(rate)+'%':>15}")

    # ── Save CSV for further analysis ────────────────────────────────────────
    rows = []
    for r in results:
        for model in models:
            resp = r["model_responses"].get(model, {})
            rows.append({
                "article":          r["article"],
                "category":         r["category"],
                "familiarity":      r["familiarity"],
                "question_type":    r["question_type"],
                "field":            r["field"],
                "model":            model,
                "correct_answer":   r["correct_answer"],
                "original_value":   r["original_value"],
                "perturbed_value":  r["perturbed_value"],
                "original_answer":  resp.get("original_answer", ""),
                "perturbed_answer": resp.get("perturbed_answer", ""),
                "memory_answer":    resp.get("memory_answer", ""),
                "answer_changed":   resp.get("answer_changed", False),
                "used_table":       resp.get("used_table", False),
            })

    with open("results/findings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Saved full results to results/findings.csv")
    print("  Open this in Excel or pandas for further analysis")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    results = load_results()
    analyse(results)
