"""
Evaluation suite for chat-cooking RAG pipeline.

Three layers:
  1. Classifier accuracy  — is_recipe_query() on labelled queries
  2. Retrieval quality    — Hit@K and MRR for known recipes in the dataset
  3. Response faithfulness — LLM answer only cites retrieved content (LLM-as-judge, optional)

Run:
    python eval.py               # layers 1 + 2
    python eval.py --faithfulness # layers 1 + 2 + 3 (uses HF_TOKEN, costs tokens)
"""
import argparse
import json
import textwrap

import pandas as pd

from rag import is_recipe_query, retrieve
from config import TOP_K

# -- 1. Classifier test cases -------------------------------------------------

CLASSIFIER_CASES: list[tuple[str, bool]] = [
    # positives
    ("How do I bake a chocolate cake?",           True),
    ("Give me a recipe for chicken soup",          True),
    ("What can I make with eggs and cheese?",      True),
    ("How long do I cook pasta?",                  True),
    ("What ingredients do I need for guacamole?",  True),
    ("Suggest a quick dinner for two",             True),
    ("How do I fry fish without it sticking?",     True),
    ("What's a good vegetarian meal?",             True),
    # negatives
    ("What's the weather like today?",             False),
    ("Tell me a joke",                             False),
    ("Who won the 2024 election?",                 False),
    ("What time is it?",                           False),
    ("How do I install Python?",                   False),
    ("What is the capital of France?",             False),
]

# -- 2. Retrieval test cases ---------------------------------------------------
# Format: (query, [expected substrings in top-k recipe names, case-insensitive])

RETRIEVAL_CASES: list[tuple[str, list[str]]] = [
    ("apple pie recipe",               ["apple", "pie"]),
    ("chocolate cherry dessert",       ["chocolate", "cherry"]),
    ("chicken salad",                  ["chicken", "salad"]),
    ("mulligatawny soup",              ["mulligatawny"]),
    ("waldorf salad",                  ["waldorf"]),
    ("baked fish",                     ["fish"]),
    ("beef tacos",                     ["beef", "taco"]),
    ("vegetarian soup",                ["vegetarian", "soup"]),
    ("apple cider drink",              ["apple", "cider"]),
    ("caramel apple dessert",          ["caramel", "apple"]),
    ("german apple cake",              ["apple", "cake"]),
    ("bread recipe",                   ["bread"]),
]

# -- 3. Faithfulness cases (used only with --faithfulness) --------------------

FAITHFULNESS_CASES = [
    "How do I make mulligatawny soup?",
    "Give me a recipe for waldorf salad",
    "How do I bake fish fillets?",
]


# -- Helpers -------------------------------------------------------------------

def run_classifier_eval() -> dict:
    print("\n-- Layer 1: Classifier accuracy ---------------------------------")
    correct = 0
    errors = []
    for query, expected in CLASSIFIER_CASES:
        got = is_recipe_query(query)
        ok = got == expected
        if ok:
            correct += 1
        else:
            errors.append((query, expected, got))
        label = "PASS" if ok else "FAIL"
        print(f"  [{label}] ({'+' if expected else '-'}) {query[:60]}")

    acc = correct / len(CLASSIFIER_CASES)
    print(f"\n  Accuracy: {correct}/{len(CLASSIFIER_CASES)} = {acc:.0%}")
    if errors:
        print("  Failed cases:")
        for q, exp, got in errors:
            print(f"    expected={exp} got={got}  \"{q}\"")
    return {"accuracy": acc, "correct": correct, "total": len(CLASSIFIER_CASES)}


def _hit_at_k(results: list[dict], keywords: list[str], k: int) -> bool:
    """True if any of the top-k results contains ALL keywords."""
    for r in results[:k]:
        name = r["name"].lower()
        if all(kw in name for kw in keywords):
            return True
    return False


def _reciprocal_rank(results: list[dict], keywords: list[str]) -> float:
    for rank, r in enumerate(results, 1):
        name = r["name"].lower()
        if all(kw in name for kw in keywords):
            return 1.0 / rank
    return 0.0


def run_retrieval_eval(k: int = TOP_K) -> dict:
    print(f"\n-- Layer 2: Retrieval quality (Hit@{k}, MRR) --------------------")
    hits = 0
    rr_sum = 0.0
    for query, keywords in RETRIEVAL_CASES:
        results = retrieve(query, top_k=k)
        hit = _hit_at_k(results, keywords, k)
        rr = _reciprocal_rank(results, keywords)
        hits += int(hit)
        rr_sum += rr
        label = "HIT " if hit else "MISS"
        top_names = [r["name"] for r in results[:k]]
        print(f"  [{label}] \"{query}\"")
        print(f"         top-{k}: {top_names}")

    n = len(RETRIEVAL_CASES)
    hit_rate = hits / n
    mrr = rr_sum / n
    print(f"\n  Hit@{k}: {hits}/{n} = {hit_rate:.0%}")
    print(f"  MRR:    {mrr:.3f}")
    return {"hit_at_k": hit_rate, "mrr": mrr, "k": k, "hits": hits, "total": n}


def run_faithfulness_eval() -> dict:
    """Use the LLM itself to judge whether its answer stays within retrieved content."""
    print("\n-- Layer 3: Faithfulness (LLM-as-judge) -------------------------")
    from llm import build_messages, stream_response

    JUDGE_PROMPT = textwrap.dedent("""\
        You are an evaluator. Given a cooking assistant's answer and the recipes it retrieved,
        decide if the answer is FAITHFUL (only uses information from the retrieved recipes)
        or HALLUCINATED (invents details not present in the retrieved recipes).

        Respond with exactly one word: FAITHFUL or HALLUCINATED, then one sentence of explanation.
    """)

    results = []
    for query in FAITHFULNESS_CASES:
        retrieved = retrieve(query, top_k=TOP_K)
        messages = build_messages(query, retrieved, [])
        response = "".join(stream_response(messages))

        # Build judge prompt
        context = "\n".join(f"- {r['name']}: {r['ingredients'][:150]}" for r in retrieved)
        judge_messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Retrieved recipes:\n{context}\n\n"
                f"Answer: {response}"
            )},
        ]
        verdict = "".join(stream_response(judge_messages)).strip()
        faithful = verdict.upper().startswith("FAITHFUL")
        label = "PASS" if faithful else "FAIL"
        print(f"  [{label}] {query}")
        print(f"         Verdict: {verdict[:120]}")
        results.append(faithful)

    score = sum(results) / len(results)
    print(f"\n  Faithfulness: {sum(results)}/{len(results)} = {score:.0%}")
    return {"faithfulness": score, "total": len(results)}


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithfulness", action="store_true",
                        help="Also run LLM faithfulness eval (uses tokens)")
    parser.add_argument("--k", type=int, default=TOP_K,
                        help=f"Top-K for retrieval eval (default: {TOP_K})")
    parser.add_argument("--json", dest="json_out", metavar="FILE",
                        help="Write results to JSON file")
    args = parser.parse_args()

    scores = {}
    scores["classifier"] = run_classifier_eval()
    scores["retrieval"]  = run_retrieval_eval(k=args.k)
    if args.faithfulness:
        scores["faithfulness"] = run_faithfulness_eval()

    print("\n-- Summary ------------------------------------------------------")
    print(f"  Classifier accuracy : {scores['classifier']['accuracy']:.0%}")
    print(f"  Retrieval Hit@{args.k}     : {scores['retrieval']['hit_at_k']:.0%}")
    print(f"  Retrieval MRR       : {scores['retrieval']['mrr']:.3f}")
    if "faithfulness" in scores:
        print(f"  Faithfulness        : {scores['faithfulness']['faithfulness']:.0%}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\n  Results written to {args.json_out}")


if __name__ == "__main__":
    main()
