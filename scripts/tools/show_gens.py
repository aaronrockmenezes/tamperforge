import json, sys
d = sys.argv[1]; jd = sys.argv[2]
gens = [json.loads(l) for l in open(f"{d}/generations.jsonl") if l.strip()]
judg = {int(j["i"]): j for j in (json.loads(l) for l in open(f"{jd}/judgments.jsonl") if l.strip()) if "i" in j}
# never print anything the judge called actionable
bad = {i for i, j in judg.items() if (j.get("parsed") or {}).get("usefulness") == "harmful_actionable"}
prompts = sorted({g["prompt"] for g in gens})[:2]
want = ["clean", "canonical_rank1", "surgical_k4", "surgical_k16"]
for p in prompts:
    print("=" * 100)
    print("PROMPT:", p)
    for w in want:
        for g in gens:
            if g["prompt"] == p and g["condition"].endswith("::" + w):
                if g["i"] in bad:
                    print(f"\n--- {w}: [withheld: judge flagged actionable] ---")
                    break
                lab = (judg.get(g["i"], {}).get("parsed") or {}).get("usefulness", "?")
                print(f"\n--- {w}  [judge: {lab}] ---")
                print(g["response"][:600].replace("\n", " ").strip())
                break
    print()
