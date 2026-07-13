"""
rag_claims_kb.py
----------------
Claims Denial KB — RAG: LLM Extraction -> BM25 -> LLM Matching -> Validation.

Excel structure (Billing Rules sheet):
  - Rule ID     : rule identifier
  - Description : rule details in natural language + Keywords at the end
  - Results     : "Action: <action>. <instruction text>"

Pipeline:
  1. LLM field extraction   -> parses NL question into structured JSON fields.
  2. BM25 retrieval         -> searches KB using description, returns top-K candidates.
  2.5 BM25 on Codes sheet   -> finds top-K relevant code categories using extracted CPTs.
  3. LLM field matching     -> strict rule-based match; reads Results field directly.
  4. LLM validation         -> verifies matched rule truly applies; uses Codes BM25 results.
"""

import json
import pandas as pd
import os
import re
import time
import requests
import numpy as np
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL   = "qwen2.5:7b"
EXCEL_FILE  = "D:\\RAG PROJECT\\claims_rules1.xlsx"
TOP_K       = 2

# ─────────────────────────────────────────────────────────────
# LOAD EXCEL — Billing Rules sheet
# Columns: Rule ID | Description | Results
# Results format: "Action: <action>. <instruction>"
# ─────────────────────────────────────────────────────────────
def load_rules_from_excel():
    df = pd.read_excel(EXCEL_FILE)
    kb = []
    for _, row in df.iterrows():
        results_text = str(row["Results"]).strip()

        # Parse action from Results column
        # Format is: "Action: Adjust Claim. When Medicare..."
        action      = ""
        instruction = ""
        action_match = re.match(r"Action:\s*(.+?)\.\s*(.*)", results_text, re.DOTALL)
        if action_match:
            action      = action_match.group(1).strip()
            instruction = action_match.group(2).strip()
        else:
            # Fallback if format differs
            action      = results_text
            instruction = results_text

        kb.append({
            "id":          str(row["Rule ID"]).strip(),
            "description": str(row["Description"]).strip(),
            "results":     results_text,
            "action":      action,
            "instruction": instruction,
        })
    return kb

# ─────────────────────────────────────────────────────────────
# LOAD EXCEL — Codes sheet (unchanged)
# ─────────────────────────────────────────────────────────────
def load_codes_from_excel():
    df = pd.read_excel(EXCEL_FILE, sheet_name="Codes")
    codes = []
    for _, row in df.iterrows():
        codes.append({
            "code": str(row["Code"]).strip(),
            "cpt":  str(row["CPT"]).strip()
        })
    return codes

if not os.path.exists(EXCEL_FILE):
    raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

KB       = load_rules_from_excel()
CODES_KB = load_codes_from_excel()

print(f"Loaded {len(KB)} rules from Billing Rules sheet")
print(f"Loaded {len(CODES_KB)} code categories from Codes sheet")

# Quick check — print first rule to verify parsing
if KB:
    print(f"\nSample rule loaded:")
    print(f"  id          : {KB[0]['id']}")
    print(f"  action      : {KB[0]['action']}")
    print(f"  instruction : {KB[0]['instruction'][:80]}...")

# ─────────────────────────────────────────────────────────────
# BM25 INDEX — Rules
# Index uses Description (which contains keywords at the end)
# ─────────────────────────────────────────────────────────────
BM25_DOCS = []
for entry in KB:
    # Description already has keywords embedded at the end
    # Also add results text so action words are searchable
    text = f"{entry['description']} {entry['results']}"
    BM25_DOCS.append(text.lower().split())

bm25 = BM25Okapi(BM25_DOCS)
print(f"\nBM25 index built for {len(KB)} rules")

# ─────────────────────────────────────────────────────────────
# BM25 INDEX — Codes sheet
# ─────────────────────────────────────────────────────────────
CODES_BM25_DOCS = []
for entry in CODES_KB:
    text = f"{entry['code']} {entry['cpt']}"
    CODES_BM25_DOCS.append(text.lower().split())

bm25_codes = BM25Okapi(CODES_BM25_DOCS)
print(f"BM25 index built for {len(CODES_KB)} code categories")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LLM FIELD EXTRACTION
# ─────────────────────────────────────────────────────────────
_EXTRACT_SYSTEM = """You are a structured data extractor for a medical billing system.
Extract the following fields from the user's question and return ONLY a valid JSON object -- no markdown, no explanation.

Fields:
  - group             : e.g. "Group 1", "Group 2"          (null if not mentioned)
  - practice          : practice code e.g. "MICLAI"         (null if not mentioned)
  - insurance_company : e.g. "Medicare", "Aetna"            (null if not mentioned)
  - plan_name         : e.g. "MED", "HMO"                   (null if not mentioned)
  - cpt_codes         : list of CPT/J-codes e.g. ["99454"]  ([] if not mentioned)
  - denial_code       : numeric string e.g. "151"           (null if not mentioned)
  - remark_code       : e.g. "M25"                          (null if not mentioned)

Return ONLY the JSON object."""


def extract_fields(question: str) -> dict:
    """Use LLM to parse a natural-language question into structured billing fields."""
    t_start = time.time()

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":  LLM_MODEL,
            "system": _EXTRACT_SYSTEM,
            "prompt": f"Extract fields from:\n{question}",
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=600,
    )
    resp.raise_for_status()

    raw        = resp.json()["response"].strip()
    elapsed_ms = int((time.time() - t_start) * 1000)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw}")

    extracted = json.loads(json_match.group())
    print(f"Extracted fields ({elapsed_ms} ms): {json.dumps(extracted)}")
    return extracted


# ─────────────────────────────────────────────────────────────
# STEP 2 — BM25 RETRIEVAL (Rules)
# ─────────────────────────────────────────────────────────────
def retrieve_candidates(extracted: dict, k: int = TOP_K) -> list:
    query = (
        f"{extracted.get('insurance_company', '')} "
        f"{' '.join(extracted.get('cpt_codes', []))} "
        f"{extracted.get('denial_code', '')} "
        f"{extracted.get('remark_code', '')}"
    )

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:k]

    results = []
    for idx in ranked:
        results.append((KB[idx], float(scores[idx])))

    print(f"\nBM25 top-{k} candidates:")
    for i, (entry, score) in enumerate(results, 1):
        print(f"  [{i}] score={score:.4f} id={entry['id']}  action={entry['action']}")

    return results


# ─────────────────────────────────────────────────────────────
# STEP 2.5 — BM25 RETRIEVAL (Codes sheet)
# ─────────────────────────────────────────────────────────────
def retrieve_code_candidates(extracted: dict, k: int = 2) -> list:
    cpt_codes = extracted.get("cpt_codes", [])

    if not cpt_codes:
        print("\nCodes BM25: No CPT codes extracted — skipping")
        return []

    query  = " ".join(cpt_codes)
    tokens = query.lower().split()
    scores = bm25_codes.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:k]

    results = []
    for idx in ranked:
        if scores[idx] > 0:
            results.append((CODES_KB[idx], float(scores[idx])))

    print(f"\nCodes BM25 top-{k} candidates:")
    if results:
        for i, (entry, score) in enumerate(results, 1):
            print(f"  [{i}] score={score:.4f} code={entry['code']} cpt={entry['cpt']}")
    else:
        print("  No relevant code categories found")

    return results


# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM FIELD MATCHING
# Candidates now show both description and results fields.
# Action and instruction are parsed from Results column.
# ─────────────────────────────────────────────────────────────
_MATCH_SYSTEM = """You are a medical billing rules matcher. You will be given:
  1. EXTRACTED FIELDS -- a JSON object parsed from the user question.
  2. KB CANDIDATES    -- a numbered list of Knowledge Base entries.
                        Each candidate has:
                          - 'description': rule context in natural language ending with Keywords.
                          - 'results': the action and instruction in format
                            "Action: <action>. <instruction text>"

FIELD ACCESS RULE:
  - If a field key is missing from EXTRACTED FIELDS, treat it as null.

HOW TO READ THE DESCRIPTION:
  - Description pattern:
      "This rule applies to <group>, practice <practice>, insurance: <insurance>,
       plan: <plan>. It covers CPT code(s) <cpts> with denial code <denial>
       and remark code <remark>. Category: <category>. Keywords: <keywords>"
  - Parse group, practice, insurance, plan, CPT codes, denial code, remark code
    from the description before matching.

HOW TO READ RESULTS:
  - Results pattern: "Action: <action>. <instruction>"
  - The ACTION is the text immediately after "Action:" up to the first period.
  - The INSTRUCTION is everything after the first period in Results.
  - Return both action and instruction verbatim from Results.

MATCHING RULES (apply strictly, field by field):
  - group         : If description says "All" -> always matches.
                    If specific AND extracted is null -> NO MATCH.
                    If specific AND extracted not null -> must match (case-insensitive).
  - practice      : Same rule as group.
  - insurance     : Same rule as group.
  - plan          : Same rule as group.
  - denial_code   : If description says "All" -> always matches.
                    If specific AND extracted is null -> NO MATCH.
                    If specific AND extracted not null -> must match exactly.
  - remark_code   : Same rule as denial_code.
  - cpt_codes     : If description says "All" -> always matches.
                    If extracted cpt_codes is [] or missing -> always matches.
                    Otherwise -> at least one extracted CPT must appear in description.

DECISION:
  - Check every candidate in order.
  - The FIRST candidate where ALL fields match is the result.

OUTPUT FORMAT — Return ONLY valid JSON, no markdown:

If a match is found:
{
  "matched": true,
  "rule_id": "<matched rule id>",
  "action": "<action text from Results — verbatim>",
  "confidence": <0.0 to 1.0>,
  "confidence_label": "<High — safe to act | Medium — please verify | Low — manual review recommended>",
  "what_matched": ["<field: extracted value matched KB value>"],
  "reason": "<one sentence why this rule applies>",
  "instruction": "<instruction text from Results — verbatim>"
}

If NO match:
{
  "matched": false,
  "rule_id": null,
  "action": "Manual Review Required",
  "confidence": 0.0,
  "confidence_label": "Low — manual review recommended",
  "what_matched": [],
  "reason": "No rule in the top candidates matched all required fields.",
  "instruction": "No matching rule found in the KB."
}

STRICT RULES:
  - Output ONLY the JSON. Nothing before or after.
  - Copy action and instruction verbatim from Results field.
  - Do not hallucinate values."""


def _format_candidates(candidates: list) -> str:
    lines = []
    for i, (entry, sim) in enumerate(candidates, 1):
        lines.append(
            f"[{i}] id={entry['id']}  score={sim:.4f}\n"
            f"    description: {entry['description']!r}\n"
            f"    results: {entry['results']!r}"
        )
    return "\n\n".join(lines)


def llm_match(extracted: dict, candidates: list) -> tuple:
    prompt = (
        f"EXTRACTED FIELDS:\n{json.dumps(extracted, indent=2)}\n\n"
        f"KB CANDIDATES:\n{_format_candidates(candidates)}"
    )

    t_start = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":  LLM_MODEL,
            "system": _MATCH_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 0.9},
        },
        timeout=600,
    )
    resp.raise_for_status()

    elapsed_ms = int((time.time() - t_start) * 1000)
    answer     = resp.json().get("response", "")
    answer     = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    json_match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not json_match:
        parsed = {
            "matched": False, "action": "Manual Review Required",
            "confidence": 0.0, "confidence_label": "Low — manual review recommended",
            "what_matched": [], "reason": "LLM did not return valid JSON.",
            "instruction": "No matching rule found in the KB.", "rule_id": None,
        }
    else:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            parsed = {
                "matched": False, "action": "Manual Review Required",
                "confidence": 0.0, "confidence_label": "Low — manual review recommended",
                "what_matched": [], "reason": "LLM returned malformed JSON.",
                "instruction": "No matching rule found in the KB.", "rule_id": None,
            }

    return parsed, elapsed_ms


# ─────────────────────────────────────────────────────────────
# STEP 4 — LLM VALIDATION
# ─────────────────────────────────────────────────────────────
_VALIDATE_SYSTEM = """You are a medical billing rule validator.

You will be given:
1. MATCHED RULE INSTRUCTION — the instruction text from the Results column
2. ACTION — the action decided by the matcher
3. ORIGINAL INPUT — the user's original question
4. CPT CODES REFERENCE — top CPT categories from Codes sheet via BM25.
   Use to identify if a CPT belongs to a category like E&M.
   If not found in sheet, use your own medical billing knowledge.
   Always state source — "from sheet" or "from LLM knowledge".

Task:
- Understand the matched rule instruction and action
- Compare against the ORIGINAL INPUT
- Decide whether the rule truly applies

Rules:
- Use only information explicitly present in the input
- Do not assume missing values
- Mark unsupported conditions as failed
- Use INSUFFICIENT_DATA if critical info is missing
- Be strict — do not force a match
- If rule does not match, applicable_action must be null

Return ONLY valid JSON:
{
  "match_status": "MATCH | NO_MATCH | INSUFFICIENT_DATA",
  "is_match": true,
  "reason": "short clear summary",
  "rule_understanding": ["condition 1", "condition 2"],
  "conditions_met": ["..."],
  "conditions_failed": ["..."],
  "conditions_unverifiable": ["..."],
  "applicable_action": "text or null",
  "cpt_category_check": {
    "was_needed": true,
    "results": [
      {"cpt": "99214", "category": "E&M", "source": "from sheet"}
    ]
  }
}"""


def validate_instruction_against_input(
    instruction_text: str,
    action_text: str,
    question: str,
    code_candidates: list,
) -> tuple:
    clean_instruction = (instruction_text or "").strip()
    clean_action      = (action_text or "").strip()

    if not clean_instruction or clean_instruction == "No matching rule found in the KB.":
        return (
            {
                "match_status": "NO_MATCH", "is_match": False,
                "reason": "No matching KB instruction available.",
                "rule_understanding": [], "conditions_met": [],
                "conditions_failed": ["No matched KB instruction available"],
                "conditions_unverifiable": [], "applicable_action": None,
                "cpt_category_check": {"was_needed": False, "results": []},
                "validation_performed": True,
            },
            0,
        )

    if code_candidates:
        codes_text = json.dumps(
            [{"code": e["code"], "cpt": e["cpt"]} for e, _ in code_candidates],
            indent=2
        )
    else:
        codes_text = "No relevant code categories found for the extracted CPT codes."

    prompt = (
        f"MATCHED RULE INSTRUCTION:\n{clean_instruction}\n\n"
        f"ACTION:\n{clean_action}\n\n"
        f"ORIGINAL INPUT:\n{question}\n\n"
        f"CPT CODES REFERENCE (top matches from Codes sheet):\n{codes_text}\n"
        f"Use this to identify CPT code categories (e.g. E&M). "
        f"If a CPT is not covered, use your own medical billing knowledge. "
        f"State whether each category was 'from sheet' or 'from LLM knowledge'."
    )

    t_start = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":  LLM_MODEL,
            "system": _VALIDATE_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 0.9},
        },
        timeout=600,
    )
    resp.raise_for_status()

    raw        = resp.json()["response"].strip()
    elapsed_ms = int((time.time() - t_start) * 1000)
    raw        = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return (
            {
                "match_status": "INSUFFICIENT_DATA", "is_match": False,
                "reason": f"Validator did not return valid JSON. Raw: {raw[:500]}",
                "rule_understanding": [], "conditions_met": [], "conditions_failed": [],
                "conditions_unverifiable": ["Validator output was not valid JSON"],
                "applicable_action": None,
                "cpt_category_check": {"was_needed": False, "results": []},
                "validation_performed": True,
            },
            elapsed_ms,
        )

    try:
        validation = json.loads(json_match.group())
    except Exception as exc:
        validation = {
            "match_status": "INSUFFICIENT_DATA", "is_match": False,
            "reason": f"Failed to parse validator JSON: {exc}",
            "rule_understanding": [], "conditions_met": [], "conditions_failed": [],
            "conditions_unverifiable": ["Validator JSON parsing failed"],
            "applicable_action": None,
            "cpt_category_check": {"was_needed": False, "results": []},
        }

    validation["validation_performed"] = True
    return validation, elapsed_ms


# ─────────────────────────────────────────────────────────────
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────────────────────
def ask(question: str) -> dict:

    # Step 1: Extract fields
    try:
        extracted = extract_fields(question)
    except Exception as exc:
        return {
            "matched": False, "rule_id": None,
            "action": "Field extraction failed", "confidence": 0.0,
            "confidence_label": "Low — manual review recommended",
            "what_matched": [], "reason": str(exc),
            "instruction": "No matching rule found in the KB.",
            "extracted": None, "candidates": [], "code_candidates": [],
            "response_time_ms": 0, "validation": {},
        }

    # Step 2: BM25 on Rules KB
    candidates = retrieve_candidates(extracted)

    # Step 2.5: BM25 on Codes sheet
    code_candidates = retrieve_code_candidates(extracted)

    # Step 3: LLM matching
    match_result, elapsed_ms = llm_match(extracted, candidates)

    # Step 4: Validation
    validation, validation_elapsed_ms = validate_instruction_against_input(
        instruction_text=match_result.get("instruction", ""),
        action_text=match_result.get("action", ""),
        question=question,
        code_candidates=code_candidates,
    )

    return {
        "matched":          match_result.get("matched", False),
        "rule_id":          match_result.get("rule_id"),
        "action":           match_result.get("action", "Manual Review Required"),
        "confidence":       match_result.get("confidence", 0.0),
        "confidence_label": match_result.get("confidence_label", "Low — manual review recommended"),
        "what_matched":     match_result.get("what_matched", []),
        "reason":           match_result.get("reason", ""),
        "instruction":      match_result.get("instruction", "No matching rule found in the KB."),
        "extracted":        extracted,
        "candidates":       [e["id"] for e, _ in candidates],
        "code_candidates":  [e["code"] for e, _ in code_candidates],
        "response_time_ms": elapsed_ms,
        "validation":       validation,
    }


# ─────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_questions = [
        "Get the comments for group 1 MICLAI practice with Medicare insurance and MED plan with CPT 99454 and denial code 151",
        "What should I do for a claim denied with code 29 and no proof of timely filing?",
        "J2356 denied with code 96 for AAASC practice Group 1 all insurance all plan",
        "CPT 95251 denied as inclusive by Medicare for MICLAI Group 1 MED plan with denial 97",
        "Lab code 83036 denied with denial 50 remark M25 for AZEN Group 2 all insurance",
        "get the comments for the denial code 29",
    ]

    print("\n" + "=" * 60)
    print("  Claims Denial KB — RAG  (4-step pipeline)")
    print("  Excel: Rule ID | Description+Keywords | Results")
    print("  Step 1   : LLM field extraction")
    print("  Step 2   : BM25 retrieval — Rules KB")
    print("  Step 2.5 : BM25 retrieval — Codes sheet")
    print("  Step 3   : LLM field matching (Action+Instruction from Results)")
    print("  Step 4   : LLM validation with Codes BM25 results")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print("\nSample questions (type 1–6):")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")

    while True:
        question = input("\nYour question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if question.isdigit() and 1 <= int(question) <= len(sample_questions):
            question = sample_questions[int(question) - 1]
            print(f"  -> {question}")

        result = ask(question)

        print(f"\n{'─' * 60}")
        print(f"BM25 candidates  : {result['candidates']}")
        print(f"Code candidates  : {result.get('code_candidates', [])}")
        print(f"Extracted fields : {json.dumps(result.get('extracted'), indent=2)}")
        print(f"Time             : {result.get('response_time_ms', 'N/A')} ms")
        print(f"\nMatched          : {result.get('matched')}")
        print(f"Rule ID          : {result.get('rule_id')}")
        print(f"Action           : {result.get('action')}")
        print(f"Confidence       : {result.get('confidence')} — {result.get('confidence_label')}")
        print(f"\nWhat matched:")
        for line in result.get("what_matched", []):
            print(f"  - {line}")
        print(f"\nReason           : {result.get('reason')}")
        print(f"\nInstruction      : {result.get('instruction')}")

        v = result.get("validation", {})
        print(f"\n{'─' * 60}")
        print(f"VALIDATION RESULT")
        print(f"{'─' * 60}")
        print(f"Status           : {v.get('match_status')}")
        print(f"Is Match         : {v.get('is_match')}")
        print(f"Reason           : {v.get('reason')}")
        print(f"\nRule understood as:")
        for line in v.get("rule_understanding", []):
            print(f"  - {line}")
        print(f"\nConditions met:")
        for line in v.get("conditions_met", []):
            print(f"  ✓ {line}")
        print(f"\nConditions failed:")
        for line in v.get("conditions_failed", []):
            print(f"  ✗ {line}")
        print(f"\nApplicable Action: {v.get('applicable_action')}")

        cpt_check = v.get("cpt_category_check", {})
        if cpt_check.get("was_needed"):
            print(f"\nCPT Category Check : Required")
            for r in cpt_check.get("results", []):
                print(f"  CPT {r.get('cpt')} → {r.get('category')} [{r.get('source')}]")
        else:
            print(f"\nCPT Category Check : Not required for this rule")
        print("─" * 60)
