"""Build trimmed copies of the generation files for the web demo (docs/index.html).

The full JSONL files on Hugging Face (jsbaan/bag-paper-data) carry prompts,
generation configs and duplicated context the demo never displays, which makes
a single config ~100 MB. This script keeps only the fields the demo reads, with
the same file names (as .json, so GitHub Pages serves them gzipped) and the same
record structure, so the demo's rendering code is unchanged.

Usage:
    python build_demo_data.py --src <dir with dev/*.jsonl> --out docs/data
"""
import argparse
import json
from pathlib import Path


def pick(obj, keys):
    """Keep only `keys` of a dict (keys missing in obj stay missing)."""
    return {k: obj[k] for k in keys if k in obj} if isinstance(obj, dict) else obj


def trim_judge_gen(g):
    # judgeBlock() reads verdict + reasoning; the hidden per-sample badges read samples[].verdict
    if not isinstance(g, dict):
        return g
    out = pick(g, ["verdict", "reasoning"])
    if isinstance(g.get("samples"), list):
        out["samples"] = [pick(s, ["verdict"]) for s in g["samples"]]
    return out


def trim_direct(r):
    ctx = pick(r.get("context", {}), ["question", "disambiguations", "references"])
    gen = r.get("generations", {})
    out_gen = {}
    if "belief_state" in gen:
        # renderBeliefState shows `raw_response || response`
        out_gen["belief_state"] = [
            pick(s, ["raw_response"]) if isinstance(s, dict) and s.get("raw_response") else pick(s, ["raw_response", "response"])
            for s in gen["belief_state"]
        ]
    if isinstance(gen.get("answer"), list):
        out_gen["answer"] = [pick(a, ["raw_response"]) for a in gen["answer"][:1]]
    return {"id": r["id"], "context": ctx, "generations": out_gen}


def trim_disambiguated(r):
    gen = r.get("generations", {})
    out_gen = {}
    if isinstance(gen.get("answer"), list):
        out_gen["answer"] = gen["answer"][:1]
    return {"id": r["id"],
            "context": pick(r.get("context", {}), ["disambiguated_question", "reference"]),
            "generations": out_gen}


def trim_clarify(r):
    return {"id": r["id"],
            "context": pick(r.get("context", {}), ["question"]),
            "generations": pick(r.get("generations", {}), ["strategy", "reasoning", "response", "raw_response"])}


def trim_user(r):
    return {"id": r["id"], "generations": pick(r.get("generations", {}), ["reasoning", "response"])}


def trim_final(r):
    gen = r.get("generations")
    if isinstance(gen, list):
        gen = gen[:1]
    elif isinstance(gen, dict):
        gen = pick(gen, ["strategy", "reasoning", "response"])
    out = {"id": r["id"], "generations": gen}
    if "belief_state" in r:
        out["belief_state"] = r["belief_state"]
    return out


def trim_judge(r):
    gen = r.get("generations", {})
    return {"id": r["id"],
            "context": pick(r.get("context", {}), ["direct_ref", "disambig_ref", "final_ref"]),
            "generations": {k: trim_judge_gen(v) for k, v in gen.items()} if isinstance(gen, dict) else gen}


def trim_claim_variation(r):
    cv = (r.get("generations") or {}).get("claim_variation")
    if isinstance(cv, dict):
        cv = pick(cv, ["n_distinct_claims", "claims"])
        if isinstance(cv.get("claims"), list):
            cv["claims"] = [pick(c, ["label", "type", "representative", "n_samples", "sample_indices"]) for c in cv["claims"]]
    return {"id": r["id"], "generations": {"claim_variation": cv}}


def trim_interpretation_variation(r):
    iv = (r.get("generations") or {}).get("interpretation_variation")
    if isinstance(iv, dict):
        iv = pick(iv, ["classifications", "interpretations"])
        if isinstance(iv.get("classifications"), list):
            iv["classifications"] = [pick(c, ["index", "label", "scope"]) for c in iv["classifications"]]
        if isinstance(iv.get("interpretations"), list):
            iv["interpretations"] = [pick(c, ["label", "sample_indices", "representative", "contextualized"])
                                     for c in iv["interpretations"]]
    return {"id": r["id"], "generations": {"interpretation_variation": iv}}


# Order matters: longer prefixes first (judge_* before anything else that could match).
TRIMMERS = [
    ("judge_", trim_judge),
    ("claim_variation", trim_claim_variation),
    ("interpretation_variation", trim_interpretation_variation),
    ("direct_", trim_direct),
    ("disambiguated_", trim_disambiguated),
    ("clarify_", trim_clarify),
    ("user_", trim_user),
    ("final_", trim_final),
]


def trimmer_for(name):
    for prefix, fn in TRIMMERS:
        if name.startswith(prefix):
            return fn
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory containing the split folders (e.g. dev/)")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    src = Path(args.src) / args.split
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for f in sorted(src.glob("*.jsonl")):
        fn = trimmer_for(f.name)
        if fn is None:
            print(f"skip {f.name}")
            continue
        lines = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # the demo's loader drops unparsable lines too
            lines.append(json.dumps(fn(rec), ensure_ascii=False, separators=(",", ":")))
        dst = out / (f.stem + ".json")
        dst.write_text("\n".join(lines) + "\n")
        total_in += f.stat().st_size
        total_out += dst.stat().st_size
    print(f"{total_in / 1e6:.1f} MB -> {total_out / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
