"""Utilities for belief-state diversity analysis (companion to paper §7 "Faithfulness").

Covers two variation signals computed over the K belief-state samples, only one of
which is reported in the paper:

  - claim_variation (-> H(claim)):
        This IS the "semantic entropy" of paper §7: "We cluster generations on the
        main answer they assert and compute semantic entropy (Kuhn et al., 2023b)".
        `claim_entropy()` clusters samples by asserted claim (`type == "content"`)
        and takes Shannon entropy over cluster sizes. This is the signal behind
        Figure 5 and the §7 claim that BAG's strategy choice tracks belief-state
        entropy while prompt-only SAG does not.

  - interpretation_variation (-> H(interpretation)):
        NOT reported anywhere in the paper, despite giving this notebook its name.
        It is a narrower, code-only signal added post-submission to probe the
        distinction §6.2 draws in prose but never quantifies: variation because
        samples assume different *explicit scopes* (e.g. "in the US, ..." vs "in
        Canada, ...", the contextualisation cues discussed in §6.2 and shown lost
        under brevity prompts in §6.3/Appendix Table 4) vs. variation that is just
        claim-level disagreement (hallucination, §7/§8). Unlike H(claim), it is
        NOT entropy over N distinct interpretation clusters — `interpretation_entropy()`
        is binary entropy over contextualised-vs-not sample counts. Treat it as an
        exploratory routing diagnostic (see the Q1-Q4 quadrant analysis and the
        H(claim) vs H(interpretation) scatter in evaluate_faithfulness.ipynb), not
        as a paper-validated metric.
"""

import sys
sys.path.insert(0, "..")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from dataloader import load_jsonl
from config import build_output_fname
from nb_utils import _prettify_reasoner_prompt, _prettify_model

sns.set_theme()


def _prettify_rp(rp):
    """Map internal reasoner-prompt keys to display names: belief* → BAG*, prompt → SAG."""
    if rp.startswith('belief'):
        return 'BAG' + rp[len('belief'):]
    if rp == 'prompt':
        return 'SAG'
    return rp


# ── Data loading ──────────────────────────────────────────────────────────────

def load_variation_data(config):
    """Load claim_variation, interpretation_variation, direct, and reasoner files.

    Returns:
        cv_data:       {(am, dp, bs): {id: item}}  — claim_variation
        iv_data:       {(am, dp, bs): {id: item}}  — interpretation_variation
        direct_data:   {(am, dp, bs): {id: item}}
        reasoner_data: {(am, dp, bs, rp): [items]}
        missing:       list of paths not found
    """
    missing = []
    cv_data, iv_data, direct_data, reasoner_data = {}, {}, {}, {}

    for am in config["assistant_models"]:
        for dp in config["direct_prompts"]:
            for bs in config["belief_samplings"]:
                key = (am, dp, bs)

                for mode, store in [("claim_variation", cv_data), ("interpretation_variation", iv_data)]:
                    items = _load(mode, am, dp, bs, config, missing)
                    store[key] = {item["id"]: item for item in items}

                direct_items = _load("direct", am, dp, bs, config, missing)
                direct_data[key] = {item["id"]: item for item in direct_items}

                for rp in config["reasoner_prompts"]:
                    reasoner_data[(am, dp, bs, rp)] = _load("clarify", am, dp, bs, config, missing, rp=rp)

    if missing:
        print(f"\n{len(missing)} file(s) not found:")
        for f in missing:
            print(f"  MISSING: {f}")

    return cv_data, iv_data, direct_data, reasoner_data, missing


def load_cvf_data(config):
    """Load claim_variation_final data for belief reasoner prompts.

    Returns:
        cvf_data: {(am, dp, bs, rp): {id: item}}  — claim_variation_final
        missing:  list of paths not found
    """
    missing = []
    cvf_data = {}
    for am in config["assistant_models"]:
        for dp in config["direct_prompts"]:
            for bs in config["belief_samplings"]:
                for rp in config["reasoner_prompts"]:
                    if "belief" in rp:
                        items = _load("claim_variation_final", am, dp, bs, config, missing,
                                      rp=rp, um=config.get("user_model"))
                        cvf_data[(am, dp, bs, rp)] = {item["id"]: item for item in items}
    if missing:
        print(f"\n{len(missing)} cvf file(s) not found:")
        for f in missing:
            print(f"  MISSING: {f}")
    return cvf_data, missing


def load_direct_judge(config):
    """Load direct-step judge verdicts.

    Returns {(am, dp, bs): {id: {"direct": verdict, "direct_anyref": verdict}}}
    where verdict is 0, 1, or -1 (unparseable).
    """
    judge_data = {}
    for am in config["assistant_models"]:
        for dp in config["direct_prompts"]:
            for bs in config["belief_samplings"]:
                fname = build_output_fname(
                    "judge", assistant_model=am, direct_prompt=dp, belief_sampling=bs,
                    seed=config["seed"], judge_model=config["judge_model"], judge_branch="direct",
                )
                path = f"{_data_path(config)}/{fname}.jsonl"
                try:
                    items = load_jsonl(path)
                    judge_data[(am, dp, bs)] = {
                        item["id"]: {
                            branch: ((item.get("generations") or {}).get(branch) or {}).get("verdict", -1)
                            for branch in ("direct", "direct_anyref")
                        }
                        for item in items
                    }
                except FileNotFoundError:
                    print(f"  MISSING: {path}")
                    judge_data[(am, dp, bs)] = {}
    return judge_data


def _data_path(config):
    split = config.get("split", "train")
    return config.get("data_path") or (
        "../data/generations" if split == "train" else f"../data/generations/{split}"
    )


def _load(mode, am, dp, bs, config, missing, rp=None, um=None):
    fname = build_output_fname(
        mode, assistant_model=am, direct_prompt=dp, belief_sampling=bs,
        seed=config["seed"], judge_model=config.get("judge_model"), reasoner_prompt=rp,
        user_model=um,
    )
    path = f"{_data_path(config)}/{fname}.jsonl"
    try:
        return load_jsonl(path)
    except FileNotFoundError:
        missing.append(path)
        return []


# ── Entropy helpers ───────────────────────────────────────────────────────────

def _entropy(counts):
    """Shannon entropy in bits from a list/array of cluster sample counts."""
    counts = np.array([c for c in counts if c > 0], dtype=float)
    if counts.sum() == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def claim_entropy(cv_gen):
    """H(claim) in bits — the paper's §7 "semantic entropy": entropy over
    type=content claim clusters. nan if no data."""
    if not cv_gen:
        return float("nan")
    content_counts = [c["n_samples"] for c in (cv_gen.get("claims") or []) if c.get("type") == "content"]
    return _entropy(content_counts) if content_counts else float("nan")


def interpretation_entropy(iv_gen):
    """H(interpretation) in bits — NOT in the paper (see module docstring).
    Binary entropy over contextualised vs uncontextualised sample counts, not
    entropy over distinct interpretation clusters like `claim_entropy`.

    High entropy = roughly half the samples are contextualised, low = one side dominates.
    Returns nan if no data.
    """
    if not iv_gen:
        return float("nan")
    n_ctx   = iv_gen.get("n_contextualised", 0) or 0
    n_unctx = iv_gen.get("n_uncontextualised", 0) or 0
    if n_ctx + n_unctx == 0:
        return float("nan")
    return _entropy([n_ctx, n_unctx])


def _belief_lengths(direct_item):
    """Word counts for each belief-state sample in a direct-answer item."""
    belief_state = (direct_item.get("generations") or {}).get("belief_state") or []
    texts = [
        (g.get("response") or g.get("raw_response") or "") if isinstance(g, dict) else (g or "")
        for g in belief_state
    ]
    return [len(t.split()) for t in texts]


# ── DataFrame builder ─────────────────────────────────────────────────────────

def build_variation_df(cv_data, iv_data, direct_data, reasoner_data, config, cvf_data=None):
    """Build a per-item × per-reasoner-prompt analysis DataFrame.

    Columns:
        assistant_model, direct_prompt, belief_sampling, reasoner_prompt, item_id,
        strategy,
        n_distinct_claims, n_content_cv, h_claim, h_claim_final, n_clarifying_cv, n_refusing_cv,
        n_contextualised, h_interpretation,
        n_uncontextualised, n_clarifying_iv, n_refusing_iv,
        mean_belief_len, std_belief_len
    """
    rows = []
    for am in config["assistant_models"]:
        for dp in config["direct_prompts"]:
            for bs in config["belief_samplings"]:
                key = (am, dp, bs)
                for rp in config["reasoner_prompts"]:
                    for r_item in reasoner_data.get((am, dp, bs, rp), []):
                        iid = r_item["id"]
                        cv_item = cv_data[key].get(iid)
                        iv_item = iv_data[key].get(iid)
                        if cv_item is None and iv_item is None:
                            continue

                        cv_gen = (cv_item.get("generations") or {}).get("claim_variation") or {} if cv_item else {}
                        iv_gen = (iv_item.get("generations") or {}).get("interpretation_variation") or {} if iv_item else {}

                        cvf_item = (cvf_data or {}).get((am, dp, bs, rp), {}).get(iid) if cvf_data else None
                        cvf_gen  = (cvf_item.get("generations") or {}).get("claim_variation") or {} if cvf_item else {}

                        lengths = _belief_lengths(direct_data[key][iid]) if iid in direct_data[key] else []

                        rows.append({
                            "assistant_model":            am,
                            "direct_prompt":              dp,
                            "belief_sampling":            bs,
                            "reasoner_prompt":            rp,
                            "item_id":                    iid,
                            "strategy":                   (r_item.get("generations") or {}).get("strategy"),
                            "n_distinct_claims":          cv_gen.get("n_distinct_claims", -1),
                            "n_content_cv":               sum(c["n_samples"] for c in (cv_gen.get("claims") or []) if c.get("type") == "content"),
                            "h_claim":                    claim_entropy(cv_gen),
                            "h_claim_final":              claim_entropy(cvf_gen),
                            "n_clarifying_cv":            cv_gen.get("n_clarifying", 0),
                            "n_refusing_cv":              cv_gen.get("n_refusing", 0),
                            "n_contextualised":   iv_gen.get("n_contextualised", -1),
                            "h_interpretation":   interpretation_entropy(iv_gen),
                            "n_uncontextualised": iv_gen.get("n_uncontextualised", 0),
                            "n_clarifying_iv":    iv_gen.get("n_clarifying", 0),
                            "n_refusing_iv":      iv_gen.get("n_refusing", 0),
                            "mean_belief_len":            float(np.mean(lengths)) if lengths else float("nan"),
                            "std_belief_len":             float(np.std(lengths)) if len(lengths) > 1 else float("nan"),
                        })

    return pd.DataFrame(rows)


def print_data_sanity_check(judge_data, df):
    """Print parse-failure counts for both judges — only lines with errors are shown."""
    judge_errors = [
        f"  {am}/{dp}/{bs}: {n}/{len(v)} unparseable"
        for (am, dp, bs), v in judge_data.items()
        if (n := sum(1 for x in v.values() if x.get("direct", -1) == -1)) > 0
    ]
    if judge_errors:
        print("=== Direct answer judge: unparseable verdicts ===")
        print("\n".join(judge_errors))

    n_rps = df["reasoner_prompt"].nunique()
    n_claim = (df.n_distinct_claims == -1).sum() // n_rps
    n_interp = (df.n_contextualised == 0).sum() // n_rps
    if n_claim or n_interp:
        print("=== Variation judges: unparseable outputs ===")
        if n_claim:
            print(f"  Claim parse errors      (n_distinct_claims=-1): {n_claim}")
        if n_interp:
            print(f"  Ctx parse warnings (n_contextualised=0): {n_interp}")


# ── Faithfulness AUROC ───────────────────────────────────────────────────────

def print_faithfulness_auroc(df_view, rps):
    """Print AUROC of entropy signals as predictors of routing strategy.

    Three scores per reasoner prompt:
      AUROC(H(interp)          → clarify): high H(interp) predicts clarify.
      AUROC(H(claim)           → abstain): high H(claim)  predicts abstain.
      AUROC(-(H(claim)+H(interp)) → direct): low total entropy (high consistency)
                                             predicts direct answer.

    'prompt' (no belief state) should be ~0.5; 'belief*' should be higher if
    the belief-state entropy actually drives the routing decision.
    """
    rows = []
    for rp in rps:
        sub = df_view[df_view.reasoner_prompt == rp].dropna(
            subset=["h_interpretation", "h_claim", "strategy"]
        )
        y_clarify = (sub["strategy"] == "clarification_question").astype(int)
        y_abstain  = (sub["strategy"] == "abstain").astype(int)
        y_direct   = (sub["strategy"] == "direct_answer").astype(int)
        consistency = -(sub["h_claim"] + sub["h_interpretation"])

        auc_clarify = (
            roc_auc_score(y_clarify, sub["h_interpretation"])
            if y_clarify.sum() > 0 else float("nan")
        )
        auc_abstain = (
            roc_auc_score(y_abstain, sub["h_claim"])
            if y_abstain.sum() > 0 else float("nan")
        )
        auc_direct = (
            roc_auc_score(y_direct, consistency)
            if y_direct.sum() > 0 else float("nan")
        )
        rows.append({
            "reasoner_prompt":                   rp,
            "AUROC H(interp)→clarify":          round(auc_clarify, 3),
            "AUROC H(claim)→abstain":           round(auc_abstain, 3),
            "AUROC consistency→direct":         round(auc_direct,  3),
        })

    df_auc = pd.DataFrame(rows).set_index("reasoner_prompt")
    print("=== Faithfulness AUROC  (0.5 = chance, 1.0 = perfect) ===")
    print(df_auc.to_string())


_JUDGE_METRICS = ("direct", "direct_anyref")

def print_quality_auroc(df_view, rps, judge_data, metric="direct"):
    """Print AUROC of entropy signals as predictors of direct-answer quality.

    metric: 'direct'        — verdict against the single canonical reference.
            'direct_anyref' — verdict correct if it matches any reference answer.

    Quality is independent of the reasoner prompt, so this deduplicates to one
    row per item using the first reasoner prompt.

    Two scores:
      AUROC(-H(claim)  → correct): low claim entropy predicts a correct answer.
      AUROC(-H(interp) → correct): low interp entropy predicts a correct answer.

    Both should be above 0.5 if the signals capture genuine answer difficulty.
    """
    if metric not in _JUDGE_METRICS:
        raise ValueError(f"metric must be one of {_JUDGE_METRICS}; got {metric!r}")

    sub = df_view[df_view.reasoner_prompt == rps[0]].dropna(
        subset=["h_claim", "h_interpretation"]
    ).copy()

    sub["verdict"] = sub.apply(
        lambda r: judge_data.get(
            (r.assistant_model, r.direct_prompt, r.belief_sampling), {}
        ).get(r.item_id, {}).get(metric, -1),
        axis=1,
    )
    sub = sub[sub["verdict"] >= 0]

    y = sub["verdict"].astype(int)
    rows = []
    for col, label in [("h_claim", "H(claim)"), ("h_interpretation", "H(interp)")]:
        auc = roc_auc_score(y, -sub[col]) if y.nunique() > 1 else float("nan")
        rows.append({"signal": f"-{label} → correct", "AUROC": round(auc, 3)})

    print(f"=== Quality AUROC  (n={len(sub)}, 0.5 = chance, 1.0 = perfect) ===")
    print(pd.DataFrame(rows).set_index("signal").to_string())


# ── Plot helpers ──────────────────────────────────────────────────────────────

STRATEGY_ORDER   = ["direct_answer", "clarification_question", "abstain"]
STRATEGY_PALETTE = {"direct_answer": "#2196F3", "clarification_question": "#4CAF50", "abstain": "#FF9800", "error": "#F44336"}
STRATEGY_LABELS  = {"direct_answer": "Direct", "clarification_question": "Clarify", "abstain": "Abstain", "error": "Error"}


def make_view(df_valid, config, model=None, dp=None, settings=None):
    """Return (df_view, run_label) for one or more model/dp settings.

    Single setting (original behaviour):
        make_view(df_valid, config, model="qwen3-14b", dp="concise")

    Multiple settings — pass a list of (model, dp) or (model, dp, [rp, ...]) tuples:
        make_view(df_valid, config, settings=[
            ("gemini-2.5-flash", "vanilla"),
            ("olmo2-13b-instruct", "concise", ["prompt", "belief"]),
        ])
    Rows from all settings are concatenated; an optional third element filters
    which reasoner_prompts are kept for that setting.
    """
    if settings is not None:
        parts = []
        for entry in settings:
            m, d, *rest = entry
            rps_filter = rest[0] if rest else None
            chunk = df_valid[(df_valid.assistant_model == m) & (df_valid.direct_prompt == d)].copy()
            if rps_filter is not None:
                chunk = chunk[chunk["reasoner_prompt"].isin(rps_filter)]
            chunk["setting"] = f"{m}/{d}"
            parts.append(chunk)
        df_view   = pd.concat(parts, ignore_index=True)
        run_label = "  |  ".join(f"{m}/{d}" for m, d, *_ in settings) + f"  |  seed: {config['seed']}  |  split: {config.get('split', '?')}"
    else:
        model     = model or config["assistant_models"][0]
        dp        = dp    or config["direct_prompts"][0]
        df_view   = df_valid[(df_valid.assistant_model == model) & (df_valid.direct_prompt == dp)].copy()
        run_label = f"{model}  |  dp: {dp}  |  seed: {config['seed']}  |  split: {config.get('split', '?')}"
    return df_view, run_label


def rp_axes(rps, **kwargs):
    """Return (fig, axes_list) for a single row of per-reasoner-prompt subplots."""
    fig, axes = plt.subplots(1, len(rps), **kwargs)
    return fig, [axes] if len(rps) == 1 else list(axes)


def kde_by_strategy(ax, df, col, strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    for strat in strategy_order:
        vals = df[df.strategy == strat][col].dropna().values
        if len(vals):
            sns.kdeplot(vals, ax=ax, label=strat, color=palette[strat], fill=True, alpha=0.3)


# ── Signal correlation analysis ──────────────────────────────────────────────

def plot_signal_correlation(df, config, rp0=None, k=10):
    """Heatmap of n_distinct_claims × n_contextualised per model.

    Summary numbers per model:
      - KL(marginal_claims ∥ marginal_interp): similar marginal shape → low value
      - Spearman r: item-level rank correlation → low = fire on different items
    """
    if rp0 is None:
        rp0 = config["reasoner_prompts"][0]
    models = config["assistant_models"]
    _dedup = df[df.reasoner_prompt == rp0].dropna(
        subset=["n_distinct_claims", "n_contextualised"]
    )

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    axes = np.atleast_1d(axes)

    for ax, am in zip(axes, models):
        sub = _dedup[_dedup.assistant_model == am]
        nc = sub["n_distinct_claims"].clip(upper=k).astype(int)
        ni = sub["n_contextualised"].clip(upper=k).astype(int)

        r, _ = spearmanr(nc, ni)

        bins = np.arange(0, k + 2)
        p_nc = np.histogram(nc, bins=bins, density=True)[0] + 1e-10
        p_ni = np.histogram(ni, bins=bins, density=True)[0] + 1e-10
        p_nc /= p_nc.sum()
        p_ni /= p_ni.sum()
        kl = float(np.sum(p_nc * np.log(p_nc / p_ni)))

        counts = (sub.assign(nc=nc, ni=ni)
                     .groupby(["nc", "ni"]).size().unstack(fill_value=0))
        counts = counts.reindex(index=range(0, k+1), columns=range(0, k+1), fill_value=0)

        sns.heatmap(counts, ax=ax, cmap="Blues", linewidths=0.3, linecolor="#444",
                    cbar_kws={"label": "# items"}, annot=True, fmt="d", annot_kws={"size": 7})
        ax.set(title=f"{am}\nKL(claims||ctx)={kl:.2f}  |  Spearman r={r:.2f}",
               xlabel="n contextualised", ylabel="n distinct claims")
        ax.invert_yaxis()

    fig.suptitle(
        f"Joint distribution: n_distinct_claims × n_contextualised  |  seed: {config['seed']}  |  split: {config.get('split', '?')}\n"
        "Low KL(claims||ctx) -> similar marginals;  low r -> fire on different items",
        y=1.04,
    )
    plt.tight_layout()
    plt.show()


# ── Overview: distributions ───────────────────────────────────────────────────

def plot_variation_overview(df, config, rp0=None, metric="h", cols=None, dp=None):
    """Per-model histograms of claim and interpretation diversity.

    rp0:    which reasoner_prompt to use for deduplication (default: first in config).
    metric: 'h' — entropy rows (H(claim), H(interpretation)).
            'n' — count rows (n_distinct_claims, n_contextualised).
    cols:   optional list of column names to show, e.g. ["h_claim"].
            Overrides metric when provided.
    dp:     optional direct_prompt filter, e.g. "vanilla" or "concise".
    """
    if rp0 is None:
        rp0 = config["reasoner_prompts"][0]
    _df = df[df.reasoner_prompt == rp0].copy()
    if dp is not None:
        _df = _df[_df.direct_prompt == dp]
    models = config["assistant_models"]

    _all_rows = [
        ("n_distinct_claims", "#5C85D6", "n distinct claims",    "n"),
        ("h_claim",           "#5C85D6", "H(claim)  bits",       "h"),
        ("n_contextualised",  "#E8A838", "n contextualised",     "n"),
        ("h_interpretation",  "#E8A838", "H(interpretation) bits", "h"),
    ]
    if cols is not None:
        active_rows = [(col_name, color, xlabel, kind)
                       for col_name, color, xlabel, kind in _all_rows if col_name in cols]
    else:
        active_rows = [(col_name, color, xlabel, kind)
                       for col_name, color, xlabel, kind in _all_rows if kind == metric]
    n_metrics = len(active_rows)

    # Pre-compute global max for discrete (n) rows so bins are identical across models.
    global_max = {}
    for col_name, _color, _xlabel, kind in active_rows:
        if kind == "n":
            vals = _df[col_name].dropna()
            global_max[col_name] = int(vals.max()) if len(vals) else 1

    MAX_COLS = 3
    n_cols = min(len(models), MAX_COLS)
    n_model_rows = math.ceil(len(models) / n_cols)
    total_rows = n_metrics * n_model_rows

    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * total_rows),
        squeeze=False,
    )

    # Hide unused cells in the last model-row of each metric strip.
    last_row_models = len(models) - (n_model_rows - 1) * n_cols
    if last_row_models < n_cols:
        for m in range(n_metrics):
            grid_row = m * n_model_rows + (n_model_rows - 1)
            for extra_col in range(last_row_models, n_cols):
                axes[grid_row, extra_col].set_visible(False)

    for i, am in enumerate(models):
        model_subrow = i // n_cols
        model_subcol = i % n_cols
        sub = _df[_df.assistant_model == am]
        for m, (col_name, color, xlabel, kind) in enumerate(active_rows):
            grid_row = m * n_model_rows + model_subrow
            ax = axes[grid_row, model_subcol]
            if kind == "n":
                vals = sub[col_name].dropna()
                max_val = global_max[col_name]
                ax.hist(vals, bins=range(0, max_val + 2), color=color, edgecolor="white", linewidth=0.5, align="left")
            else:
                vals = sub[col_name].dropna().values
                if len(vals):
                    sns.histplot(vals, ax=ax, color=color, bins=20, kde=True)
            ax.set(xlabel=xlabel)
            if m == 0:
                ax.set_title(am)
            if model_subcol == 0:
                ax.set_ylabel("# items")

    # Unify x and y limits within each metric strip.
    for m in range(n_metrics):
        strip_axes = [
            axes[m * n_model_rows + model_subrow, model_subcol]
            for model_subrow in range(n_model_rows)
            for model_subcol in range(n_cols)
            if axes[m * n_model_rows + model_subrow, model_subcol].get_visible()
        ]
        x0 = min(ax.get_xlim()[0] for ax in strip_axes)
        x1 = max(ax.get_xlim()[1] for ax in strip_axes)
        y0 = min(ax.get_ylim()[0] for ax in strip_axes)
        y1 = max(ax.get_ylim()[1] for ax in strip_axes)
        for ax in strip_axes:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

    # fig.suptitle(
    #     f"Belief-state diversity distributions  |  seed: {config['seed']}  |  split: {config.get('split', '?')}", y=1.01,
    # )
    plt.tight_layout()
    plt.show()


def plot_n_contextualised_dist(df, config, rp0=None, k=10, dp=None):
    """Bar charts of n_contextualised distribution (0..k) per model × direct_prompt.

    Rows = direct_prompts, columns = models.  Each bar shows the fraction of questions
    with exactly that many contextualised belief-state samples.  Since n_contextualised
    is independent of reasoner_prompt, rows are deduplicated by rp0.
    dp: optional filter to a single direct_prompt value, e.g. "vanilla" or "concise".
    """
    if rp0 is None:
        rp0 = config["reasoner_prompts"][0]
    _df = df[(df.reasoner_prompt == rp0) & (df.n_contextualised >= 0)].copy()
    models = config["assistant_models"]
    dps    = [dp] if dp is not None else config["direct_prompts"]

    fig, axes = plt.subplots(
        len(dps), len(models),
        figsize=(4 * len(models), 3 * len(dps)),
        sharex=True, sharey=True,
    )
    axes = np.array(axes).reshape(len(dps), len(models))

    bins = np.arange(0, k + 2)
    for r, dp in enumerate(dps):
        dp_df = _df[_df.direct_prompt == dp]
        for c, am in enumerate(models):
            ax = axes[r, c]
            vals = dp_df[dp_df.assistant_model == am]["n_contextualised"].dropna()
            counts, _ = np.histogram(vals, bins=bins)
            fracs = counts / counts.sum() if counts.sum() > 0 else counts
            ax.bar(np.arange(k + 1), fracs, color="#E8A838", edgecolor="white", linewidth=0.4)
            ax.set_xlim(-0.5, k + 0.5)
            ax.set_xticks(range(0, k + 1, 2))
            if r == 0:
                ax.set_title(am, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{dp}\nfraction", fontsize=8)

    fig.supxlabel("n contextualised (out of K=10)", y=-0.01)
    fig.suptitle(
        f"Distribution of n_contextualised per question  |  seed: {config['seed']}  |  split: {config.get('split', '?')}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_natural_routing(df, config, rp0=None, mode="mean_fraction", judge="iv",
                         strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    """Stacked bar: how often models naturally clarify / abstain / answer per model.

    'Natural' routing comes from variation-judge annotations, not the BAG reasoner.

    judge:
      "iv"   — interpretation-variation judge.
               K = n_contextualised + n_uncontextualised + n_clarifying_iv + n_refusing_iv.
      "cv"   — claim-variation judge.
               K = n_distinct_claims + n_clarifying_cv + n_refusing_cv.
      "both" — two rows: IV on top, CV on bottom.
    mode:
      "mean_fraction"  — per item compute n_clarifying/K and n_refusing/K, then
                         average across items.  For constant K this equals pooling
                         all samples (same result).
      "at_least_one"   — fraction of *questions* that have ≥1 clarify or ≥1 refuse
                         sample anywhere in their belief state.  More sensitive when
                         clarifying is rare but non-zero.
    """
    if rp0 is None:
        rp0 = config["reasoner_prompts"][0]
    _df = df[df.reasoner_prompt == rp0].copy()

    def _add_p_cols(d, src):
        if src == "iv":
            d["_K"] = (d["n_contextualised"].clip(lower=0)
                       + d["n_uncontextualised"].clip(lower=0)
                       + d["n_clarifying_iv"].clip(lower=0)
                       + d["n_refusing_iv"].clip(lower=0))
            d = d[d["_K"] > 0]
            if mode == "at_least_one":
                d["_p_clarify"] = (d["n_clarifying_iv"].clip(lower=0) >= 1).astype(float)
                d["_p_refuse"]  = (d["n_refusing_iv"].clip(lower=0)  >= 1).astype(float)
                d["_p_direct"]  = ((d["_p_clarify"] == 0) & (d["_p_refuse"] == 0)).astype(float)
            else:
                d["_p_clarify"] = d["n_clarifying_iv"].clip(lower=0) / d["_K"]
                d["_p_refuse"]  = d["n_refusing_iv"].clip(lower=0)  / d["_K"]
                d["_p_direct"]  = (d["n_contextualised"].clip(lower=0) + d["n_uncontextualised"].clip(lower=0)) / d["_K"]
        else:  # cv
            d["_K"] = (d["n_content_cv"].clip(lower=0)
                       + d["n_clarifying_cv"].clip(lower=0)
                       + d["n_refusing_cv"].clip(lower=0))
            d = d[d["_K"] > 0]
            if mode == "at_least_one":
                d["_p_clarify"] = (d["n_clarifying_cv"].clip(lower=0) >= 1).astype(float)
                d["_p_refuse"]  = (d["n_refusing_cv"].clip(lower=0)  >= 1).astype(float)
                d["_p_direct"]  = ((d["_p_clarify"] == 0) & (d["_p_refuse"] == 0)).astype(float)
            else:
                d["_p_clarify"] = d["n_clarifying_cv"].clip(lower=0) / d["_K"]
                d["_p_refuse"]  = d["n_refusing_cv"].clip(lower=0)  / d["_K"]
                d["_p_direct"]  = d["n_content_cv"].clip(lower=0) / d["_K"]
        return d

    judges = ["iv", "cv"] if judge == "both" else [judge]
    judge_labels = {"iv": "interpretation-variation judge", "cv": "claim-variation judge"}
    ylabel = "fraction of questions with ≥1 sample" if mode == "at_least_one" else "mean fraction of K samples"

    models = config["assistant_models"]
    dps    = config["direct_prompts"]
    n_rows = len(judges)

    fig, axes = plt.subplots(n_rows, len(dps),
                             figsize=(5 * len(dps), 4 * n_rows),
                             sharey=True, squeeze=False)

    strategy_cols = [
        ("direct_answer",          "_p_direct",  "direct"),
        ("clarification_question", "_p_clarify", "clarify"),
        ("abstain",                "_p_refuse",  "abstain / refuse"),
    ]
    x = np.arange(len(models))

    for row, src in enumerate(judges):
        src_df = _add_p_cols(_df.copy(), src)
        for col_idx, dp in enumerate(dps):
            ax = axes[row][col_idx]
            dp_df = src_df[src_df.direct_prompt == dp]
            means = dp_df.groupby("assistant_model")[[c for _, c, _ in strategy_cols]].mean()
            means = means.reindex(models)

            bottom = np.zeros(len(models))
            for strat_key, pcol, label in strategy_cols:
                vals = means[pcol].fillna(0).values
                ax.bar(x, vals, bottom=bottom, color=palette[strat_key], label=label, width=0.6)
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            title = dp if n_rows == 1 else (f"{dp}\n({judge_labels[src]})" if col_idx == 0 else dp)
            ax.set_title(title)

        axes[row][0].set_ylabel(f"{judge_labels[src]}\n\n{ylabel}" if n_rows > 1 else ylabel)

    axes[0][-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Natural routing from variation-judge annotations (no BAG reasoner)"
        f"  |  mode={mode}  |  seed: {config['seed']}  |  split: {config.get('split', '?')}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    for src in judges:
        src_df = _add_p_cols(_df.copy(), src)
        rows = []
        for dp in dps:
            dp_df = src_df[src_df.direct_prompt == dp]
            means = dp_df.groupby("assistant_model")[[c for _, c, _ in strategy_cols]].mean()
            means = means.reindex(models)
            for am in models:
                rows.append({
                    "model": am,
                    "direct_prompt": dp,
                    "clarify %": round(means.loc[am, "_p_clarify"] * 100, 1),
                    "abstain/refuse %": round(means.loc[am, "_p_refuse"] * 100, 1),
                })
        tbl = pd.DataFrame(rows).set_index(["model", "direct_prompt"])
        print(f"\n{judge_labels[src]}  |  mode={mode}")
        print(tbl.to_string())


# ── Strategy composition histograms ──────────────────────────────────────────

def plot_strategy_composition(df_view, col, run_label, k=10,
                               fill=True,
                               strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    """Stacked fill histogram: proportion of each routing strategy per x-value.

    col:  any of 'n_distinct_claims', 'n_contextualised', 'h_claim', 'h_interpretation'.
          Columns starting with 'n_' are treated as discrete counts (clipped to k);
          columns starting with 'h_' are treated as continuous entropy values (20 bins).
    k:    number of belief samples — discrete counts above k are clipped into the k bin.
    fill: True → normalise each bin to 1 (proportion view, hides bar size).
          False → stacked counts (bar height reflects how many items fall in that bin).
    """
    _col_map = {
        "n_claims": ("n_distinct_claims", "# distinct claims",    "Strategy composition by n_distinct_claims\n(BAG: n=1 → mostly direct; higher → more clarify/abstain)"),
        "h_claims": ("h_claim",           "Semantic entropy (bits)", "Strategy composition by H(claim)\n(BAG: low H → direct; high H → clarify/abstain)"),
        "n_interp": ("n_contextualised",  "# contextualised",     "Strategy composition by n_contextualised\n(BAG: high n_contextualised → more clarify)"),
        "h_interp": ("h_interpretation",  "H(interpretation) bits", "Strategy composition by H(interpretation)\n(BAG: high H → clarify)"),
    }
    if col not in _col_map:
        raise ValueError(f"col must be one of {list(_col_map)}; got {col!r}")
    col, xlabel, title = _col_map[col]
    discrete = col.startswith("n_")

    _df = df_view.dropna(subset=["strategy"]).copy()
    if discrete:
        _df[col] = _df[col].clip(upper=k)
        hist_kwargs = dict(binwidth=1, discrete=True)
        xlim = (0, k + 1)
    else:
        hist_kwargs = dict(bins=20)
        xlim = None

    n_rps = _df["reasoner_prompt"].nunique()
    multi_setting = "setting" in _df.columns and _df["setting"].nunique() > 1
    multiple = "fill" if fill else "stack"
    ylabel = "" if fill else "# items"

    rp_unique = list(dict.fromkeys(_df["reasoner_prompt"]))
    pretty_titles = {rp: _prettify_reasoner_prompt(rp) for rp in rp_unique}
    col_order = [pretty_titles[rp] for rp in rp_unique]
    _df = _df.copy()
    _df["reasoner_prompt"] = _df["reasoner_prompt"].map(pretty_titles)

    def _prettify_setting(s):
        model, dp = s.split("/", 1)
        label = _prettify_model(model)
        if dp != "vanilla":
            label += f" / {dp}"
        return label

    if multi_setting:
        row_order = list(dict.fromkeys(_df["setting"]))
        g = sns.FacetGrid(_df, row="setting", col="reasoner_prompt",
                          height=2.6, aspect=1.94, sharey="row", sharex=True,
                          row_order=row_order, col_order=col_order)
        g.set_titles(template="")
        for ax, name in zip(g.axes[0], g.col_names):
            ax.set_title(name, size=17, fontweight="bold")
    else:
        g = sns.FacetGrid(_df, col="reasoner_prompt", col_wrap=n_rps,
                          height=3.0, aspect=1.49, sharey=False,
                          col_order=col_order)
        g.set_titles(col_template="{col_name}", size=17, fontweight="bold")

    g.map_dataframe(sns.histplot, x=col, hue="strategy", hue_order=strategy_order,
                    palette=palette, multiple=multiple, stat="count", **hist_kwargs)
    if xlim:
        g.set(xlim=xlim)

    if multi_setting:
        g.set_axis_labels(xlabel, "", fontsize=16)
        for ax, name in zip(g.axes[:, 0], g.row_names):
            ax.set_ylabel(_prettify_setting(name), fontsize=20, fontweight="bold", labelpad=5)
    else:
        g.set_axis_labels(xlabel, ylabel, fontsize=20)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=15)
    # g.figure.suptitle(f"{title}  |  {run_label}", fontsize=17)
    g.figure.tight_layout(rect=[0, 0, 1.0, 0.93])
    pretty_strategy_labels = [STRATEGY_LABELS.get(s, s) for s in strategy_order]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[s]) for s in strategy_order]
    legend_ax = g.axes[0, -1] if g.axes.ndim == 2 else g.axes.flat[-1]
    legend_ax.legend(handles, pretty_strategy_labels, title="Strategy", fontsize=13, title_fontsize=14,
                     loc="lower right", borderaxespad=0.5)
    plt.show()


# ── Per-strategy variation plots ──────────────────────────────────────────────

def plot_variation_by_strategy(df_view, rps, run_label,
                                kind="violin", bw_adjust=1.0, metric="h",
                                strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    """Per-strategy distribution of claim and interpretation variation.

    kind:      'scaled_kde'  — overlaid KDEs scaled by n so area ∝ count (default).
               'violin'      — KDE violin (each strategy normalised independently).
               'boxen'       — letter-value plot (no KDE).
    bw_adjust: KDE bandwidth multiplier. Lower = less smoothing. Default 1.0.
    metric:    'h' — entropy (H(claim) / H(interpretation) in bits).
               'n' — distinct count (n_distinct_claims / n_contextualised).
    """
    from scipy.stats import gaussian_kde

    _metric_map = {
        "h":       [("h_claim",                  "H(claim)  bits"),
                    ("h_interpretation",           "H(interpretation)  bits")],
        "h_claim": [("h_claim",                  "H(claim)  bits")],
        "h_interp":[("h_interpretation",           "H(interpretation)  bits")],
        "n":       [("n_distinct_claims", "# distinct claims"),
                    ("n_contextualised",  "# contextualised")],
    }
    if metric not in _metric_map:
        raise ValueError(f"metric must be 'h' or 'n'; got {metric!r}")
    metrics = _metric_map[metric]
    tick_labels = ["direct", "clarify", "abstain"]
    n_rows, n_cols = len(metrics), len(rps)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
        sharex="row", sharey="row" if kind == "scaled_kde" else True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r, (col, xlabel) in enumerate(metrics):
        for c, rp in enumerate(rps):
            ax = axes[r, c]
            sub = df_view[df_view.reasoner_prompt == rp].dropna(subset=[col])
            if kind == "scaled_kde":
                x_max = sub[col].quantile(0.999)
                xs = np.linspace(0, x_max, 300)
                for strat in strategy_order:
                    vals = sub[sub.strategy == strat][col].dropna().values
                    if len(vals) < 2:
                        continue
                    bw = vals.std() * len(vals) ** -0.2 * bw_adjust
                    ys = gaussian_kde(vals, bw_method=bw)(xs) * len(vals)
                    ax.fill_between(xs, ys, alpha=0.35, color=palette[strat])
                    ax.plot(xs, ys, color=palette[strat], linewidth=1, label=strat)
                ax.set_ylabel("# items  (approx.)" if c == 0 else "")
            elif kind == "violin":
                sns.violinplot(
                    data=sub, x=col, y="strategy",
                    order=strategy_order, hue="strategy", hue_order=strategy_order,
                    palette=palette, legend=False, inner="quartile", cut=0,
                    bw_adjust=bw_adjust, density_norm="count", ax=ax,
                )
                ax.yaxis.set_major_locator(plt.FixedLocator(range(len(strategy_order))))
                ax.set_yticklabels(tick_labels if c == 0 else [])
                if c == 0:
                    ax.set_ylabel(xlabel, fontsize=11, labelpad=8)
            else:
                sns.boxenplot(
                    data=sub, x=col, y="strategy",
                    order=strategy_order, hue="strategy", hue_order=strategy_order,
                    palette=palette, legend=False, ax=ax,
                )
                ax.yaxis.set_major_locator(plt.FixedLocator(range(len(strategy_order))))
                ax.set_yticklabels(tick_labels if c == 0 else [])
            if r == 0:
                ax.set_title(_prettify_rp(rp))
            ax.set_xlabel(xlabel)

    if kind == "scaled_kde":
        handles = [plt.Line2D([0], [0], color=palette[s], linewidth=2) for s in strategy_order]
        fig.legend(handles, strategy_order, loc="upper right", fontsize=9, framealpha=0.3)

    fig.suptitle(run_label)
    plt.tight_layout()
    plt.show()


# ── Scatter: H(claim) vs H(interpretation) ───────────────────────────────────

_SCATTER_COLS = {
    "h": {
        "x": ("h_interpretation",          "H(interpretation)  bits"),
        "y": ("h_claim",                   "H(claim)  bits"),
        "title_y": "H(claim)", "title_x": "H(interpretation)",
    },
    "n": {
        "x": ("n_contextualised",  "# contextualised"),
        "y": ("n_distinct_claims", "# distinct claims"),
        "title_y": "n(claim)", "title_x": "n(contextualised)",
    },
    "binary": {
        "x": ("_interp_binary", "any contextualised (0 = none, 1 = ≥1)"),
        "y": ("h_claim",        "H(claim)  bits"),
        "title_y": "H(claim)", "title_x": "contextualised binary",
    },
}


def plot_entropy_scatter(df_view, rps, run_label,
                          strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE,
                          alpha=0.5, s=30, mode="scatter", ncols=None, metric="h"):
    """Scatter or KDE contour plot of claim vs interpretation diversity by routing strategy.

    metric='h':      entropy axes — H(claim) vs H(interpretation).
    metric='n':      count axes  — n_distinct_claims vs n_contextualised.
    metric='binary': H(claim) vs binary interp axis (0 = no interpretations, 1 = ≥1).
                     Scatter points are jittered on x; x-ticks set to 0 and 1.

    mode='scatter': coloured dot per question.
    mode='kde':     filled 2-D KDE contour per strategy — reveals where each
                    strategy concentrates without overplotting.

    One subplot per reasoner prompt, arranged in a grid (ncols columns).

    Hypothesis:
      bottom-left  (low claim, low interp)  → direct answer
      right        (high interp)            → clarify
      top-left     (high claim, low interp) → abstain
    """
    if metric not in _SCATTER_COLS:
        raise ValueError(f"metric must be one of {list(_SCATTER_COLS)}; got {metric!r}")

    cols = _SCATTER_COLS[metric]
    x_col, x_label = cols["x"]
    y_col, y_label = cols["y"]

    df_sc = df_view.dropna(subset=["n_contextualised", y_col]).copy()
    if metric == "binary":
        df_sc["_interp_binary"] = (df_sc["n_contextualised"] > 0).astype(int)
    else:
        df_sc = df_sc.dropna(subset=[x_col])

    n = len(rps)
    if ncols is None:
        ncols = n
    nrows = math.ceil(n / ncols)
    fig, ax_grid = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5.0 * nrows),
        sharey=True, sharex=(metric != "binary"),
        squeeze=False,
    )
    axes_flat = [ax_grid[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    axes = axes_flat[:n]

    for idx, (ax, rp) in enumerate(zip(axes, rps)):
        sub = df_sc[df_sc.reasoner_prompt == rp]
        for strat in strategy_order:
            pts = sub[sub.strategy == strat]
            if pts.empty:
                continue
            if mode == "kde":
                sns.kdeplot(
                    x=pts[x_col], y=pts[y_col],
                    ax=ax, color=palette[strat], label=strat,
                    fill=True, alpha=0.25, levels=5, warn_singular=False,
                )
                sns.kdeplot(
                    x=pts[x_col], y=pts[y_col],
                    ax=ax, color=palette[strat],
                    fill=False, alpha=0.7, levels=5, linewidths=1.2, warn_singular=False,
                )
            else:
                x_vals = pts[x_col]
                if metric == "binary":
                    rng = np.random.default_rng(seed=42)
                    x_vals = x_vals + rng.uniform(-0.08, 0.08, size=len(x_vals))
                ax.scatter(
                    x_vals, pts[y_col],
                    c=palette[strat], label=strat, alpha=alpha, s=s, linewidths=0,
                )

        # Highlight n_interp=0 vs n_interp≥1 boundary (discrete count axis only).
        if metric == "n":
            ax.axvline(0.5, color="#888", linestyle="--", linewidth=1.2, zorder=3)
            ax.axvspan(-0.5, 0.5, color="#888", alpha=0.07, zorder=0)
            ax.text(0.02, 0.97, "n=0", fontsize=8, color="#666",
                    transform=ax.transAxes, va="top", ha="left")
        elif metric == "binary":
            ax.axvline(0.5, color="#888", linestyle="--", linewidth=1.2, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["none", "≥1"])
            ax.set_xlim(-0.4, 1.4)

        col_idx = idx % ncols
        ax.set_title(_prettify_rp(rp), fontsize=13, pad=6)
        ax.set_xlabel(x_label, fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=11)
        ax.tick_params(labelsize=10)

        ty, tx = cols["title_y"], cols["title_x"]
        kw = dict(fontsize=9, color="#aaa", ha="center", va="center",
                  transform=ax.transAxes)
        ax.text(0.22, 0.22, f"low {ty}\nlow {tx}\n→ direct",   **kw)
        ax.text(0.78, 0.22, f"high {tx}\n→ clarify",            **kw)
        ax.text(0.22, 0.78, f"high {ty}\nlow {tx}\n→ abstain", **kw)

    axes[0].legend(fontsize=10, loc="upper right")
    fig.suptitle(
        f"{y_label} vs {x_label} by routing strategy  |  {run_label}\n"
        f"BAG hypothesis: bottom-left → direct  ·  right → clarify  ·  top-left → abstain",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.show()


# ── Calibration: entropy → routing probability ────────────────────────────────

def plot_calibration(df_view, rps, run_label, n_bins=5,
                     strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    """Calibration plots: does each entropy signal monotonically predict routing?

    Left panel:  binned H(interpretation) → P(clarify)
    Right panel: binned H(claim)          → P(abstain)
    One line per reasoner prompt. `prompt` baseline should be flat (no belief state).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (xcol, target, xlabel) in zip(axes, [
        ("h_interpretation", "clarification_question", "H(interpretation)  bits"),
        ("h_claim",          "abstain",                "H(claim)  bits"),
    ]):
        for rp in rps:
            sub = df_view[df_view.reasoner_prompt == rp].dropna(subset=[xcol, "strategy"]).copy()
            if len(sub) < n_bins * 2:
                continue
            sub["_bin"] = pd.qcut(sub[xcol], q=n_bins, duplicates="drop")
            binned = (
                sub.groupby("_bin", observed=True)
                   .apply(lambda g: (g.strategy == target).mean(), include_groups=False)
            )
            mids = [iv.mid for iv in binned.index]
            ls = "--" if rp == "prompt" else "-"
            ax.plot(mids, binned.values, marker="o", label=_prettify_rp(rp),
                    linestyle=ls, alpha=0.85, linewidth=1.5, markersize=4)

        ax.set(xlabel=xlabel, ylabel=f"P({target.replace('_', ' ')})", ylim=(0, 1))
        ax.axhline(0.5, color="#ddd", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Calibration: entropy → routing probability  |  {run_label}\n"
        "Dashed = SAG baseline (no belief state) — should be flat if BAG is faithful",
        y=1.04,
    )
    plt.tight_layout()
    plt.show()


# ── Entropy CDFs by strategy ──────────────────────────────────────────────────

def plot_entropy_cdfs(df_view, rps, run_label,
                      metric="both",
                      strategy_order=STRATEGY_ORDER, palette=STRATEGY_PALETTE):
    """Empirical CDFs of entropy signals per routing strategy, one subplot per reasoner prompt.

    metric: 'h_claim'  — H(claim) only.
            'h_interp' — H(interpretation) only.
            'both'     — both signals, one figure each (default).
    """
    _cols = {
        "h_claim":  [("h_claim",          "H(claim)  bits")],
        "h_interp": [("h_interpretation", "H(interpretation)  bits")],
        "both":     [("h_claim",          "H(claim)  bits"),
                     ("h_interpretation", "H(interpretation)  bits")],
    }
    if metric not in _cols:
        raise ValueError(f"metric must be one of {list(_cols)}; got {metric!r}")

    for col, xlabel in _cols[metric]:
        df_ec = df_view.dropna(subset=[col]).copy()
        fig, axes = rp_axes(rps, figsize=(5 * len(rps), 4), sharey=True)
        for ax, rp in zip(axes, rps):
            sub = df_ec[df_ec.reasoner_prompt == rp]
            for strat in strategy_order:
                vals = np.sort(sub[sub.strategy == strat][col].dropna().values)
                if len(vals):
                    ax.step(vals, np.arange(1, len(vals) + 1) / len(vals),
                            label=strat, color=palette[strat], where="post")
            ax.set(title=_prettify_rp(rp), xlabel=xlabel, ylabel="CDF")
            ax.legend(fontsize=8)
        fig.suptitle(f"CDF of {col} by strategy  |  {run_label}", y=1.02)
        plt.tight_layout()
        plt.show()


# ── Entropy reduction after clarification ─────────────────────────────────────

def print_entropy_reduction(df, config, dp="vanilla", h_min=0.01):
    """Print mean ΔH(claim) and % entropy reduction for items routed to clarify.

    Only items with strategy == 'clarification_question' and h_claim > h_min are
    included in the percentage table to avoid division by near-zero entropy.
    """
    rps = config["reasoner_prompts"]

    df = df.copy()
    df["delta_h_claim"] = df["h_claim_final"] - df["h_claim"]
    clarified = df[
        df["delta_h_claim"].notna() &
        (df["direct_prompt"] == dp) &
        (df["strategy"] == "clarification_question")
    ]
    print(f"Items with delta_h_claim ({dp}): {len(clarified.drop_duplicates('item_id'))} unique items\n")

    tbl = (
        clarified
        .groupby(["assistant_model", "reasoner_prompt"])["delta_h_claim"]
        .agg(["mean"])
        .round(3)
    )
    tbl.columns = ["mean"]
    tbl = tbl.unstack("reasoner_prompt")
    tbl.columns = [f"{stat} / {rp}" for stat, rp in tbl.columns]
    bag_cols = [c for c in tbl.columns if "belief" in c]
    tbl["mean / BAG (avg)"] = tbl[bag_cols].mean(axis=1).round(3)
    print(tbl.to_string())

    clarified_pct = clarified[clarified["h_claim"] > h_min].copy()
    clarified_pct["pct_reduction"] = -clarified_pct["delta_h_claim"] / clarified_pct["h_claim"] * 100
    tbl_pct = (
        clarified_pct
        .groupby(["assistant_model", "reasoner_prompt"])["pct_reduction"]
        .agg(["mean"])
        .round(1)
    )
    tbl_pct.columns = ["mean"]
    tbl_pct = tbl_pct.unstack("reasoner_prompt")
    tbl_pct.columns = [f"{stat} / {rp}" for stat, rp in tbl_pct.columns]
    bag_cols_pct = [c for c in tbl_pct.columns if "belief" in c]
    tbl_pct["mean / BAG (avg)"] = tbl_pct[bag_cols_pct].mean(axis=1).round(1)
    print("\n% entropy reduction  (positive = reduced):")
    print(tbl_pct.to_string())

    return clarified
