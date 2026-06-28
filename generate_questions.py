"""
generate_questions.py
---------------------
For each collected table, generate 3 questions using Groq.
Each question must be answerable ONLY from the table —
not from general world knowledge.

Run after collect_tables.py:
    python generate_questions.py
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

QUESTION_PROMPT = ChatPromptTemplate.from_template("""
You are designing a research experiment on table reasoning.
Given a Wikipedia info-box table below, generate exactly 3 questions.

Rules:
1. Each question must be answerable using ONLY the table — not general knowledge
2. The answer must come from a SPECIFIC field in the table
3. The question should NOT mention the article name (test if model uses the table)
4. Vary question types: one numerical, one named-entity, one relational

Table about: {article} (but do NOT mention this in questions)
Category: {category}

Table fields:
{fields_text}

Reply with EXACTLY this format — nothing else:

Q1: <question>
A1: <exact value from table>
FIELD1: <which table field this comes from>
TYPE1: <numerical | named_entity | relational>

Q2: <question>
A2: <exact value from table>
FIELD2: <which table field this comes from>
TYPE2: <numerical | named_entity | relational>

Q3: <question>
A3: <exact value from table>
FIELD3: <which table field this comes from>
TYPE3: <numerical | named_entity | relational>
""")


def format_fields(fields: dict) -> str:
    return "\n".join(f"| {k:25s} | {v}" for k, v in fields.items())


def parse_questions(text: str) -> list:
    """Parse the LLM's structured question output."""
    import re
    questions = []
    for i in range(1, 4):
        q = re.search(rf'Q{i}:\s*(.+)', text)
        a = re.search(rf'A{i}:\s*(.+)', text)
        f = re.search(rf'FIELD{i}:\s*(.+)', text)
        t = re.search(rf'TYPE{i}:\s*(.+)', text)
        if q and a and f and t:
            questions.append({
                "question":    q.group(1).strip(),
                "answer":      a.group(1).strip(),
                "field":       f.group(1).strip(),
                "type":        t.group(1).strip().lower()
            })
    return questions


def generate_for_table(table: dict) -> list:
    fields_text = format_fields(table["fields"])
    response = llm.invoke(
        QUESTION_PROMPT.format_messages(
            article=table["article"],
            category=table["category"],
            fields_text=fields_text
        )
    )
    return parse_questions(response.content.strip())


if __name__ == "__main__":
    # Load tables
    with open("data/tables_raw.json") as f:
        tables = json.load(f)

    print(f"Generating questions for {len(tables)} tables...")
    print("=" * 50)

    results = []
    for i, table in enumerate(tables):
        print(f"  [{i+1:2d}/{len(tables)}] {table['article']}...", end=" ")
        try:
            questions = generate_for_table(table)
            if len(questions) < 2:
                print(f"SKIP (only {len(questions)} questions parsed)")
                continue
            results.append({
                **table,
                "questions": questions
            })
            print(f"OK ({len(questions)} questions)")
        except Exception as e:
            print(f"ERR — {e}")
        time.sleep(1.5)   # Groq rate limit

    print(f"\n✓ Generated questions for {len(results)} tables")

    # Save
    with open("data/tables_with_questions.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✓ Saved to data/tables_with_questions.json")

    # Preview
    if results:
        t = results[0]
        print(f"\nSample — {t['article']}:")
        for q in t["questions"]:
            print(f"  [{q['type']}] Q: {q['question']}")
            print(f"           A: {q['answer']}  (field: {q['field']})")
