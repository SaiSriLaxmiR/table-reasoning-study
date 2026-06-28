import re
import json
import time
import os
import requests

ARTICLES = [
    # SCIENTISTS
    ("Albert Einstein",                  "scientist",    "famous"),
    ("Marie Curie",                      "scientist",    "famous"),
    ("Niels Bohr",                       "scientist",    "famous"),
    ("Rosalind Franklin",                "scientist",    "famous"),
    ("Ada Lovelace",                     "scientist",    "famous"),
    ("Subrahmanyan Chandrasekhar",       "scientist",    "obscure"),
    ("Emmy Noether",                     "scientist",    "obscure"),
    ("Srinivasa Ramanujan",              "scientist",    "obscure"),
    ("Chien-Shiung Wu",                  "scientist",    "obscure"),
    ("Jagadish Chandra Bose",            "scientist",    "obscure"),

    # COUNTRIES
    ("France",                           "country",      "famous"),
    ("Brazil",                           "country",      "famous"),
    ("Japan",                            "country",      "famous"),
    ("Nigeria",                          "country",      "famous"),
    ("Canada",                           "country",      "famous"),
    ("Bhutan",                           "country",      "obscure"),
    ("Suriname",                         "country",      "obscure"),
    ("Vanuatu",                          "country",      "obscure"),
    ("Djibouti",                         "country",      "obscure"),
    ("Eswatini",                         "country",      "obscure"),

    # FILMS
    ("The Godfather",                    "film",         "famous"),
    ("Parasite (2019 film)",             "film",         "famous"),
    ("Dune (2021 film)",                 "film",         "famous"),
    ("Everything Everywhere All at Once","film",         "famous"),
    ("RRR (film)",                       "film",         "famous"),
    ("Lagaan",                           "film",         "obscure"),
    ("The Battle of Algiers",            "film",         "obscure"),
    ("Bicycle Thieves",                  "film",         "obscure"),
    ("Pan's Labyrinth",                  "film",         "obscure"),
    ("City of God (film)",               "film",         "obscure"),

    # CHEMICAL ELEMENTS
    ("Gold",                             "element",      "famous"),
    ("Carbon",                           "element",      "famous"),
    ("Hydrogen",                         "element",      "famous"),
    ("Uranium",                          "element",      "famous"),
    ("Silicon",                          "element",      "famous"),
    ("Osmium",                           "element",      "obscure"),
    ("Tellurium",                        "element",      "obscure"),
    ("Praseodymium",                     "element",      "obscure"),
    ("Bohrium",                          "element",      "obscure"),
    ("Roentgenium",                      "element",      "obscure"),

    # ORGANISATIONS
    ("Google",                           "organisation", "famous"),
    ("UNESCO",                           "organisation", "famous"),
    ("SpaceX",                           "organisation", "famous"),
    ("Amnesty International",            "organisation", "famous"),
    ("Infosys",                          "organisation", "famous"),
    ("Grameen Bank",                     "organisation", "obscure"),
    ("BRAC (organization)",              "organisation", "obscure"),
    ("Aga Khan Development Network",     "organisation", "obscure"),
    ("SEWA (trade union)",               "organisation", "obscure"),
    ("Pratham",                          "organisation", "obscure"),
]

HEADERS = {
    "User-Agent": "TableReasoningStudy/1.0 (academic research; contact: student@asu.edu)"
}


def fetch_wikitext(title: str) -> str:
    """
    Fetch raw wikitext of an article's lead section (section 0)
    using the Wikipedia Action API.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action":        "query",
        "titles":        title,
        "prop":          "revisions",
        "rvprop":        "content",
        "rvslots":       "main",
        "rvsection":     "0",
        "format":        "json",
        "formatversion": "2",
        "redirects":     "1",
    }
    resp = requests.get(url, params=params,
                        headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""
    revisions = pages[0].get("revisions", [])
    if not revisions:
        return ""
    return revisions[0].get("slots", {}).get("main", {}).get("content", "")


def parse_infobox(wikitext: str) -> dict:
    """
    Extract key-value pairs from a Wikipedia infobox in wikitext.
    Returns {field: cleaned_value}.
    """
    # Match infobox block — handles nested braces roughly
    match = re.search(
        r'\{\{[Ii]nfobox[^\n]*\n(.*?)(?=\n\}\}|\Z)',
        wikitext, re.DOTALL
    )
    if not match:
        return {}

    block = match.group(0)
    fields = {}

    for line in block.split("\n"):
        line = line.strip()
        if not (line.startswith("|") and "=" in line):
            continue

        key, _, value = line[1:].partition("=")
        key   = key.strip()
        value = value.strip()

        # Clean wikitext markup
        value = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', value)   # links
        value = re.sub(r'\{\{[Cc]onvert\|([0-9.,]+)[^}]*\}\}', r'\1', value)  # convert
        value = re.sub(r'\{\{[Bb]irth[ _]date[^}]*\|(\d{4})[^}]*\}\}', r'\1', value)
        value = re.sub(r'\{\{[Dd]eath[ _]date[^}]*\|(\d{4})[^}]*\}\}', r'\1', value)
        value = re.sub(r'\{\{[^}]+\}\}', '', value)                        # other templates
        value = re.sub(r'<[^>]+>',       '', value)                        # HTML tags
        value = re.sub(r"'{2,}",         '', value)                        # bold/italic
        value = re.sub(r'\s+',           ' ', value).strip(" ,;()")

        # Skip empty, comment, or very long values
        if not key or not value or len(value) > 150 or value.startswith("<!--"):
            continue

        fields[key] = value

    return fields


def collect_all() -> list:
    tables = []

    for title, category, familiarity in ARTICLES:
        print(f"  Fetching: {title:<45}", end=" ", flush=True)
        try:
            wikitext = fetch_wikitext(title)

            if not wikitext:
                print("SKIP — no content")
                continue

            fields = parse_infobox(wikitext)

            if len(fields) < 3:
                print(f"SKIP — only {len(fields)} fields parsed")
                continue

            tables.append({
                "article":     title,
                "category":    category,
                "familiarity": familiarity,
                "fields":      fields,
            })
            print(f"OK  ({len(fields)} fields)")

        except requests.exceptions.ConnectionError:
            print("ERR — no internet connection (are you online?)")
        except requests.exceptions.Timeout:
            print("ERR — request timed out")
        except Exception as e:
            print(f"ERR — {e}")

        time.sleep(0.5)   # polite rate limit for Wikipedia

    return tables


if __name__ == "__main__":
    print("Wikipedia Infobox Collector")
    print("=" * 60)

    tables = collect_all()

    print(f"\n{'='*60}")
    print(f"Collected {len(tables)} / {len(ARTICLES)} tables")

    # Save
    os.makedirs("data", exist_ok=True)
    out_path = "data/tables_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")

    # Preview first table
    if tables:
        t = tables[0]
        print(f"\nPreview — {t['article']} ({t['category']}, {t['familiarity']}):")
        for k, v in list(t["fields"].items())[:8]:
            print(f"  {k:30s} : {v}")