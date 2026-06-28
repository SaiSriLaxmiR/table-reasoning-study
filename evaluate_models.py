"""
evaluate_models.py
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODELS = {
    "llama-3.3-70b": ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )
}

TABLE_PROMPT = ChatPromptTemplate.from_template("""
You are given a table. Answer the question using ONLY the information
in the table below. Do not use any outside knowledge.
Give a SHORT answer — just the value, no explanation.

Table:
{table_text}

Question: {question}

Answer:""")

MEMORY_PROMPT = ChatPromptTemplate.from_template("""
Answer this question from your own knowledge.
Give a SHORT answer — just the value, no explanation.

Question: {question}

Answer:""")


def format_table(fields: dict) -> str:
    rows = "\n".join(f"| {k:30s} | {v}" for k, v in fields.items())
    return f"| {'Field':30s} | Value\n|{'-'*31}|{'-'*20}\n{rows}"


def query_model(model, prompt_messages, sleep_after=3) -> str:
    """Query with exponential backoff on rate limit."""
    for attempt in range(4):
        try:
            response = model.invoke(prompt_messages)
            time.sleep(sleep_after)
            return response.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = [30, 60, 120, 180][attempt]
                print(f"\n    [429] waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


def answers_match(a1: str, a2: str) -> bool:
    a1 = a1.lower().strip(" .,'\"")
    a2 = a2.lower().strip(" .,'\"")
    if a1 == a2:
        return True
    if len(a1) > 3 and len(a2) > 3:
        if a1 in a2 or a2 in a1:
            return True
    return False


def load_checkpoint() -> tuple[list, set]:
    """Load existing results and return (results, done_keys)."""
    try:
        with open("results/raw_results.json") as f:
            existing = json.load(f)
        # Only keep clean rows (no ERROR)
        clean = [
            r for r in existing
            if not any(
                "ERROR" in str(v.get("original_answer", ""))
                for v in r["model_responses"].values()
            )
        ]
        done_keys = set(
            f"{r['article']}||{r['question']}" for r in clean
        )
        print(f"Resuming — {len(clean)} clean rows already done, "
              f"skipping {len(existing) - len(clean)} errored rows")
        return clean, done_keys
    except FileNotFoundError:
        return [], set()


def save_checkpoint(results: list):
    os.makedirs("results", exist_ok=True)
    with open("results/raw_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def evaluate_all(tables: list) -> list:
    results, done_keys = load_checkpoint()

    for table_idx, table in enumerate(tables):
        print(f"\n[{table_idx+1}/{len(tables)}] {table['article']} "
              f"({table['familiarity']})")

        for q_idx, q in enumerate(table["questions"]):
            key = f"{table['article']}||{q['question']}"

            if key in done_keys:
                print(f"  Q{q_idx+1}: SKIP (already done)")
                continue

            original_text  = format_table(table["fields"])
            perturbed_text = format_table(q["perturbed_fields"])

            row = {
                "article":       table["article"],
                "category":      table["category"],
                "familiarity":   table["familiarity"],
                "question":      q["question"],
                "question_type": q["type"],
                "field":         q["field"],
                "correct_answer":  q["answer"],
                "original_value":  q["original_value"],
                "perturbed_value": q["perturbed_value"],
                "model_responses": {}
            }

            for model_name, model in MODELS.items():
                # 3 calls per question — sleep 3s after each
                orig_ans = query_model(model, TABLE_PROMPT.format_messages(
                    table_text=original_text, question=q["question"]))

                pert_ans = query_model(model, TABLE_PROMPT.format_messages(
                    table_text=perturbed_text, question=q["question"]))

                mem_ans = query_model(model, MEMORY_PROMPT.format_messages(
                    question=q["question"]))

                # Skip if any call errored — will retry next run
                if any("ERROR" in a for a in [orig_ans, pert_ans, mem_ans]):
                    print(f"  Q{q_idx+1}: ERROR — skipping, will retry next run")
                    continue

                answer_changed = not answers_match(orig_ans, pert_ans)
                used_table     = answers_match(pert_ans, q["perturbed_value"])

                row["model_responses"][model_name] = {
                    "original_answer":  orig_ans,
                    "perturbed_answer": pert_ans,
                    "memory_answer":    mem_ans,
                    "answer_changed":   answer_changed,
                    "used_table":       used_table,
                }

                status = "TABLE ✓" if used_table else "MEMORY ✗"
                print(f"  [{model_name}] Q{q_idx+1}: {status} | "
                      f"orig='{orig_ans[:25]}' pert='{pert_ans[:25]}'")

            if row["model_responses"]:
                results.append(row)
                done_keys.add(key)
                # Save after every question — never lose progress
                save_checkpoint(results)

    return results


if __name__ == "__main__":
    with open("data/tables_perturbed.json") as f:
        tables = json.load(f)

    os.makedirs("results", exist_ok=True)

    print("Table Reasoning Evaluation")
    print("=" * 55)
    print(f"Tables: {len(tables)} | Models: {list(MODELS.keys())}")
    print("Saves after every question — safe to Ctrl+C and resume\n")

    results = evaluate_all(tables)

    print(f"\n✓ Total clean results: {len(results)}")

    # Summary
    print("\n── PSR Summary ─────────────────────────────────────")
    for model_name in MODELS:
        total = len(results)
        used  = sum(
            1 for r in results
            if r["model_responses"].get(model_name, {}).get("used_table", False)
        )
        psr = used / total * 100 if total else 0
        print(f"  {model_name:25s}: PSR = {psr:.1f}% ({used}/{total})")