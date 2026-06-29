"""
rag_claims_kb.py
----------------
Claims Denial KB — RAG: LLM Extraction → Structured FAISS → LLM Matching.

Pipeline (revised):
  1. LLM field extraction   -> parses NL question into structured JSON fields FIRST.
  2. Structured FAISS query -> builds a rich text string from extracted fields
                               (same format as build_index.py chunks) and queries FAISS.
                               FAISS now compares like-for-like: structured vs structured.
  3. LLM field matching     -> strict rule-based match against FAISS candidates;
                               returns instruction VERBATIM.

WHY THIS IS BETTER THAN FAISS-ON-RAW-QUESTION
  The FAISS index stores structured chunks like:
      "Denial: 29 | Remark: All  ...  Rule: write off as past TFL"
  Querying with raw NL ("get the comments for denial code 29") creates a
  semantic mismatch — the query and the indexed chunks look nothing alike.
  By extracting fields first and rebuilding the same structured format,
  the query vector lands right next to the correct KB entry in embedding space.

Run build_index.py whenever the KB changes.
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
EXCEL_FILE  = "D:\RAG PROJECT\claims_rules1.xlsx"
TOP_K       = 2

def load_rules_from_excel():
    df = pd.read_excel(EXCEL_FILE)
    kb = []
    for _, row in df.iterrows():
        kb.append({
            "id":          str(row["Rule ID"]),
            "description": str(row["Description"]),
            "keywords":    str(row["Keywords"]).split("|")
        })
    return kb

if not os.path.exists(EXCEL_FILE):

    raise FileNotFoundError(
        f"Excel file not found: {EXCEL_FILE}"
    )

KB = load_rules_from_excel()

print(
    f"Loaded {len(KB)} rules from Excel"
)

# ==================================================
# BM25 INDEX
# ==================================================
BM25_DOCS = []
for entry in KB:
    text = f"{entry['description']} {' '.join(entry['keywords'])}"
    BM25_DOCS.append(text.lower().split())

bm25 = BM25Okapi(BM25_DOCS)
print(f"Loaded {len(KB)} KB rules")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LLM FIELD EXTRACTION  (runs FIRST now)
# Parses free-text question -> structured JSON fields.
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

    raw = resp.json()["response"].strip()
    elapsed_ms = int((time.time() - t_start) * 1000)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw}")

    extracted = json.loads(json_match.group())
    print(f"Extracted fields ({elapsed_ms} ms): {json.dumps(extracted)}")
    return extracted


# ─────────────────────────────────────────────────────────────
# STEP 2 — STRUCTURED FAISS QUERY
# Converts extracted fields into the same chunk format used by
# build_index.py so the query vector is comparable to stored vectors.
# ─────────────────────────────────────────────────────────────

def _build_query_chunk(extracted: dict) -> str:
    """
    Mirror the build_doc_chunk() format from build_index.py so the
    query embedding lands in the same region of the vector space as
    the indexed KB entries.

    Unknown fields use "All" (the KB wildcard) so they don't push the
    query vector away from entries that carry that wildcard.
    """
    cpt_str = ", ".join(extracted.get("cpt_codes") or []) or "All"
    return (
        f"Group     : {extracted.get('group') or 'All'}\n"
        f"Practice  : {extracted.get('practice') or 'All'}\n"
        f"Insurance : {extracted.get('insurance_company') or 'All'}\n"
        f"Plan      : {extracted.get('plan_name') or 'All'}\n"
        f"CPT Codes : {cpt_str}\n"
        f"Denial    : {extracted.get('denial_code') or 'All'} | "
        f"Remark: {extracted.get('remark_code') or 'All'}"
    )

def retrieve_candidates(extracted: dict, k: int = TOP_K):
    query = (
        f"{extracted.get('insurance_company','')} "
        f"{' '.join(extracted.get('cpt_codes', []))} "
        f"{extracted.get('denial_code','')} "
        f"{extracted.get('remark_code','')}"
    )

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:k]

    results = []
    for idx in ranked:
        results.append((KB[idx], float(scores[idx])))

    print(f"\nBM25 top-{k} candidates:")
    for i, (entry, score) in enumerate(results, 1):
        print(f"  [{i}] score={score:.4f} id={entry['id']}")

    return results


# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM FIELD MATCHING  (unchanged logic)
# LLM receives extracted JSON + FAISS candidates; returns instruction verbatim.
# ─────────────────────────────────────────────────────────────
_MATCH_SYSTEM = """You are a medical billing rules matcher. You will be given:
  1. EXTRACTED FIELDS -- a JSON object parsed from the user question.
  2. KB CANDIDATES    -- a numbered list of Knowledge Base entries.
                        Each candidate has a 'description' field written in natural language.
                        The description contains all rule details: group, practice, insurance,
                        plan, CPT codes, denial code, remark code, category, action, and instruction.
                        Read the description carefully to extract these values before matching.

FIELD ACCESS RULE:
  - If a field key is missing from EXTRACTED FIELDS, treat it exactly as if its value were null.

HOW TO READ THE DESCRIPTION:
  - The description follows this pattern:
      "This rule applies to <group>, practice <practice>, insurance: <insurance>,
       plan: <plan>. It covers CPT code(s) <cpts> with denial code <denial>
       and remark code <remark>. Category: <category>. Action: <action>. <instruction sentence>"
  - Parse each value from the description before applying matching rules.
  - The INSTRUCTION to return is the final sentence in the description (after "Action: ...").

MATCHING RULES (apply strictly, field by field):
  - group             : If description says "All" -> always matches.
                        If description has specific group AND extracted is null -> NO MATCH.
                        If description has specific group AND extracted is not null -> must match (case-insensitive).
  - practice          : Same rule as group.
  - insurance         : Same rule as group.
  - plan              : Same rule as group.
  - denial_code       : If description says "All" -> always matches.
                        If description has specific denial code AND extracted is null -> NO MATCH.
                        If description has specific denial code AND extracted is not null -> must match exactly.
  - remark_code       : Same rule as denial_code.
  - cpt_codes         : If description says "All" for CPT -> always matches.
                        If extracted cpt_codes is [] or missing -> always matches.
                        Otherwise -> at least one extracted CPT must appear in the description's CPT list.

DECISION:
  - Check every candidate in order.
  - The FIRST candidate where ALL fields satisfy the rules above is the match.

OUTPUT FORMAT:
  Return ONLY a valid JSON object — no markdown, no explanation outside the JSON.

  If a match is found:
  {
    "matched": true,
    "rule_id": "<the matched rule id>",
    "action": "<exact action field from the matched rule>",
    "confidence": <a float from 0.0 to 1.0 — how certain you are this is the right rule>,
    "confidence_label": "<one of: High — safe to act | Medium — please verify | Low — manual review recommended>",
    "what_matched": [
      "<field name>: extracted value matched KB value — e.g. denial_code: 16 matched 16>",
      "<one line per field that was checked and passed>"
    ],
    "reason": "<one plain-English sentence explaining why this rule applies to this specific claim>",
    "instruction": "<exact instruction field from the matched rule — verbatim, no changes>"
  }

  If NO candidate matches:
  {
    "matched": false,
    "rule_id": null,
    "action": "Manual Review Required",
    "confidence": 0.0,
    "confidence_label": "Low — manual review recommended",
    "what_matched": [],
    "reason": "No rule in the top candidates matched all required fields for this claim.",
    "instruction": "No matching rule found in the KB."
  }

STRICT OUTPUT RULES:
  - Output ONLY the JSON object. Nothing before it, nothing after it.
  - Do not paraphrase the instruction field — copy it verbatim.
  - Do not hallucinate field values."""


def _format_candidates(candidates: list[tuple[dict, float]]) -> str:
    lines = []
    for i, (entry, sim) in enumerate(candidates, 1):
        lines.append(
    f"[{i}] id={entry['id']}  score={sim:.4f}\n"
    f"    description: {entry['description']!r}"
)
    return "\n\n".join(lines)


def llm_match(
    extracted: dict,
    candidates: list[tuple[dict, float]],
) -> tuple[str, int]:

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

    # ── FULL DEBUG DUMP ──
    full_json = resp.json()
    answer = full_json.get("response", "")

    print(f"\n{'='*60}")
    print(f"[DEBUG] LLM MATCH RAW RESPONSE ({elapsed_ms} ms)")
    print(f"{'='*60}")
    print(f"  response length : {len(answer)}")
    print(f"  response repr   : {repr(answer[:2000])}")   # first 2000 chars
    print(f"  done            : {full_json.get('done')}")
    print(f"  done_reason     : {full_json.get('done_reason')}")
    print(f"  eval_count      : {full_json.get('eval_count')}")
    print(f"  prompt_eval_count: {full_json.get('prompt_eval_count')}")

    # Check for <think> blocks
    if "<think>" in answer:
        think_end = answer.find("</think>")
        if think_end != -1:
            after_think = answer[think_end + len("</think>"):]
            print(f"  <think> block   : YES (ends at char {think_end})")
            print(f"  after </think>  : {repr(after_think[:500])}")
        else:
            print(f"  <think> block   : UNCLOSED (no </think> found)")
    else:
        print(f"  <think> block   : NONE")
    print(f"{'='*60}\n")

    return answer, elapsed_ms

# 4th Step Validate the rule with input
_VALIDATE_SYSTEM = """You are a medical billing rule validator.

You will be given:
1. MATCHED RULE INSTRUCTION
2. ORIGINAL INPUT

Task:
- Understand the matched rule instruction
- Compare it directly against the ORIGINAL INPUT
- Decide whether the rule actually matches the data in the input

Rules:
- The ORIGINAL INPUT may contain natural language, JSON, or both
- Use only the information explicitly present in the input
- Do not assume missing values
- If a required condition is not clearly supported, mark it as failed
- If critical information is missing, use INSUFFICIENT_DATA
- Be strict. Do not force a match.
- If the rule contains multiple branches (such as Medicare vs non-Medicare), determine the applicable branch only if the main rule conditions match
- If the rule does not match, applicable_action must be null

Return ONLY valid JSON in this exact structure:
{
  "match_status": "MATCH | NO_MATCH | INSUFFICIENT_DATA",
  "is_match": true,
  "reason": "short clear summary",
  "rule_understanding": [
    "condition 1",
    "condition 2"
  ],
  "conditions_met": [
    "..."
  ],
  "conditions_failed": [
    "..."
  ],
  "conditions_unverifiable": [
    "..."
  ],
  "applicable_action": "text or null"
}"""

def validate_instruction_against_input(
    instruction_text: str,
    question: str,
) -> tuple[dict, int]:
    """
    STEP 4:
    Pass the matched instruction and the original input directly to the validator.
    The validator understands the rule and checks whether it matches the input.
    """
    clean_instruction = (instruction_text or "").strip()

    if not clean_instruction:
        return (
            {
                "match_status": "NO_MATCH",
                "is_match": False,
                "reason": "Matched instruction is empty.",
                "rule_understanding": [],
                "conditions_met": [],
                "conditions_failed": ["Matched instruction is empty"],
                "conditions_unverifiable": [],
                "applicable_action": None,
                "validation_performed": True,
            },
            0,
        )

    if clean_instruction == "No matching rule found in the KB.":
        return (
            {
                "match_status": "NO_MATCH",
                "is_match": False,
                "reason": "Step 3 returned no matching KB instruction.",
                "rule_understanding": [],
                "conditions_met": [],
                "conditions_failed": ["No matched KB instruction available"],
                "conditions_unverifiable": [],
                "applicable_action": None,
                "validation_performed": True,
            },
            0,
        )

    prompt = (
        f"MATCHED RULE INSTRUCTION:\n{clean_instruction}\n\n"
        f"ORIGINAL INPUT:\n{question}"
    )

    t_start = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": LLM_MODEL,
            "system": _VALIDATE_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()

    raw = resp.json()["response"].strip()
    elapsed_ms = int((time.time() - t_start) * 1000)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return (
            {
                "match_status": "INSUFFICIENT_DATA",
                "is_match": False,
                "reason": f"Validator did not return valid JSON. Raw: {raw[:1000]}",
                "rule_understanding": [],
                "conditions_met": [],
                "conditions_failed": [],
                "conditions_unverifiable": ["Validator output was not valid JSON"],
                "applicable_action": None,
                "validation_performed": True,
            },
            elapsed_ms,
        )

    try:
        validation = json.loads(json_match.group())
    except Exception as exc:
        validation = {
            "match_status": "INSUFFICIENT_DATA",
            "is_match": False,
            "reason": f"Failed to parse validator JSON: {exc}",
            "rule_understanding": [],
            "conditions_met": [],
            "conditions_failed": [],
            "conditions_unverifiable": ["Validator JSON parsing failed"],
            "applicable_action": None,
            "validation_performed": True,
        }

    validation["validation_performed"] = True
    return validation, elapsed_ms

# ─────────────────────────────────────────────────────────────
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    # Step 1: LLM extracts structured fields from natural language
    try:
        extracted = extract_fields(question)
    except Exception as exc:
        return {
            "answer":     f"Field extraction failed: {exc}",
            "extracted":  None,
            "candidates": [],
        }

    # Step 2: Build structured query chunk → BM search
    candidates = retrieve_candidates(extracted)

    # Step 3: LLM matches extracted fields against BM candidates
    answer, elapsed_ms = llm_match(extracted, candidates)


    validation, validation_elapsed_ms = validate_instruction_against_input(
            instruction_text=answer,
            question=question,
        )

    return {
        "answer":           answer,
        "extracted":        extracted,
        "candidates":       [e["id"] for e, _ in candidates],
        "response_time_ms": elapsed_ms,
        "validation": validation,

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
    print("  Claims Denial KB — RAG_EXP  (LLM-first pipeline)")
    print("  Step 1 : LLM field extraction   (NL → structured JSON)")
    print("  Step 2 : Structured BM query (JSON → chunk → vector)")
    print("  Step 3 : LLM field matching     (structured JSON vs candidates)")
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
        print(f"BM25 candidates : {result['candidates']}")
        print(f"Extracted fields : {json.dumps(result.get('extracted'), indent=2)}")
        print(f"Time             : {result.get('response_time_ms', 'N/A')} ms")
        print(f"\nInstruction:\n{result['answer']}")
        print("─" * 60)
