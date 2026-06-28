"""
perturb_tables.py
-----------------
For each table+question pair, create a perturbed version
of the table where the answer field is changed to a wrong value.

Three perturbation strategies:
  Type 1 — numerical:     change a number to a different plausible number
  Type 2 — named_entity:  change a name/place to a different plausible one
  Type 3 — relational:    change a value that affects a derived answer

Run after generate_questions.py:
    python perturb_tables.py
"""

import os
import json
import time
import copy
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

PERTURB_PROMPT = ChatPromptTemplate.from_template("""
You are helping design a research experiment on table reasoning.

I have a table with this field:
  Field: {field}
  Current value: {value}
  Question type: {qtype}

Generate ONE plausible-but-wrong replacement value for this field.

Rules:
- Must be factually wrong (different from "{value}")
- Must be plausible — same type, similar format
- For numbers: change the value by 20-50% (e.g. 1952 → 1934)
- For names/places: use a real but incorrect name of the same type
- For years: change by 5-20 years
- Keep the same format (if original is a year, return a year)

Reply with ONLY the replacement value — no explanation, no quotes.
""")


def generate_perturbation(field: str, value: str, qtype: str) -> str:
    """Generate one plausible wrong value for a field."""
    response = llm.invoke(
        PERTURB_PROMPT.format_messages(
            field=field,
            value=value,
            qtype=qtype
        )
    )
    return response.content.strip()


def perturb_table_fields(table: dict, questions: list) -> dict:
    """
    For each question, create a perturbed copy of the table
    where that question's answer field has a wrong value.
    Returns the table with perturbed_questions added.
    """
    perturbed_qs = []

    for q in questions:
        field = q["field"]
        original_value = table["fields"].get(field, q["answer"])

        # Generate wrong value
        wrong_value = generate_perturbation(field, original_value, q["type"])

        # Build perturbed table
        perturbed_fields = copy.deepcopy(table["fields"])
        perturbed_fields[field] = wrong_value

        perturbed_qs.append({
            **q,
            "original_value":   original_value,
            "perturbed_value":  wrong_value,
            "perturbed_fields": perturbed_fields,
        })
        time.sleep(0.8)

    return perturbed_qs


if __name__ == "__main__":
    with open("data/tables_with_questions.json") as f:
        tables = json.load(f)

    print(f"Generating perturbations for {len(tables)} tables...")
    print("=" * 50)

    results = []
    for i, table in enumerate(tables):
        print(f"  [{i+1:2d}/{len(tables)}] {table['article']}...", end=" ")
        try:
            perturbed_qs = perturb_table_fields(table, table["questions"])
            results.append({
                **table,
                "questions": perturbed_qs
            })
            print(f"OK")
        except Exception as e:
            print(f"ERR — {e}")

    print(f"\n✓ Perturbed {len(results)} tables")

    with open("data/tables_perturbed.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✓ Saved to data/tables_perturbed.json")

    # Preview
    if results and results[0]["questions"]:
        t = results[0]
        q = t["questions"][0]
        print(f"\nSample perturbation — {t['article']}:")
        print(f"  Field:           {q['field']}")
        print(f"  Original value:  {q['original_value']}")
        print(f"  Perturbed value: {q['perturbed_value']}")
        print(f"  Question:        {q['question']}")
        print(f"  Correct answer:  {q['answer']}")
