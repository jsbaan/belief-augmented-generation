"""Shared notebook utilities for pipeline evaluation notebooks."""

import collections
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── General utilities ─────────────────────────────────────────────────────────

def nested_dict():
    """Recursive defaultdict factory for arbitrarily nested metric storage."""
    return collections.defaultdict(nested_dict)


def sample_is_valid(item):
    """Return False for thinking-model items cut off before </think> (raw_response is None)."""
    gens = item['generations']
    sample = (gens.get('answer') or gens.get('samples') or [{}])[0]  # older think-model runs used 'samples' instead of 'answer'
    return not isinstance(sample, dict) or sample.get('raw_response') is not None


def masked_mean(values, mask):
    """Mean of values where mask is True, skipping None; returns nan if empty."""
    filtered = [v for v, m in zip(values, mask) if m and v is not None]
    return np.mean(filtered) if filtered else np.nan


_MISSING = object()

def _get(d, *keys, default=''):
    """Safely traverse nested dicts; returns default if any key is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, _MISSING)
        if d is _MISSING:
            return default
    return d


# ── Ref consistency ───────────────────────────────────────────────────────────

def _ref_disambig(item):
    ctx = item.get('context')
    if ctx and ctx.get('reference') is not None:
        return str(ctx['reference'])
    return None

def _ref_user(item):
    ctx = item.get('context')
    if ctx and ctx.get('reference') is not None:
        return str(ctx['reference'][0])
    return None

def _ref_judge(item):
    ctx = item.get('context')
    if ctx and ctx.get('final_ref') is not None:
        return str(ctx['final_ref'])
    return None

def _pct_unstable(ref_by_id):
    n_total    = len(ref_by_id)
    n_unstable = sum(1 for refs in ref_by_id.values() if len(refs) > 1)
    return n_total, n_unstable

def _show_unstable(label, ref_by_id):
    n_total, n_unstable = _pct_unstable(ref_by_id)
    print(f"  {label}: {n_unstable / n_total * 100:.1f}% unstable  ({n_unstable}/{n_total})")
    for iid, refs in list((k, v) for k, v in ref_by_id.items() if len(v) > 1)[:3]:
        print(f"    item {iid}: {refs}")

def check_direct_judge_ref_consistency(outputs, config):
    """Check that direct_judge and disambig_judge use the same ref for each item."""
    print("── direct_judge vs disambig_judge ref consistency ────────────────────")
    for model in config['assistant_models']:
        for dp in config['direct_prompts']:
            for bs in config['belief_samplings']:
                o = outputs[model][dp][bs]
                dj = o.get('direct_judge', [])
                dgj = o.get('disambig_judge', [])
                if not dj or not dgj:
                    continue
                dj_refs  = {item['id']: str(item['context']['direct_ref'])
                            for item in dj if (item.get('context') or {}).get('direct_ref') is not None}
                dgj_refs = {item['id']: str((item.get('context') or {}).get('disambig_ref'))
                            for item in dgj if (item.get('context') or {}).get('disambig_ref') is not None}
                ref_by_id = {iid: {dj_refs[iid], dgj_refs[iid]}
                             for iid in dj_refs if iid in dgj_refs}
                if ref_by_id:
                    _show_unstable(f"{model}/{dp}/{bs}", ref_by_id)


def check_output_completeness(outputs):
    """Check all output files have the same length and aligned question IDs."""
    lengths = {}
    id_orders = {}  # fname -> list of ids

    def _collect(fname, lst):
        lengths[fname] = len(lst)
        if lst and isinstance(lst[0], dict) and "id" in lst[0]:
            id_orders[fname] = [item["id"] for item in lst]

    for model, dps in outputs.items():
        for dp, bss in dps.items():
            for bs, o in bss.items():
                prefix = f"{model}/{dp}/{bs}"
                for step in ("direct", "disambiguated", "direct_judge", "disambig_judge", "belief_judge"):
                    lst = o.get(step)
                    if lst is not None:
                        _collect(f"{prefix}/{step}", lst)
                for rp, lst in o.get("reasoner", {}).items():
                    _collect(f"{prefix}/reasoner[{rp}]", lst)
                for rp, lst in o.get("user", {}).items():
                    _collect(f"{prefix}/user[{rp}]", lst)
                for rp, fp_dict in o.get("final", {}).items():
                    for fp, lst in fp_dict.items():
                        _collect(f"{prefix}/final[{rp}][{fp}]", lst)
                for rp, lst in (o.get("clarify_judge") or {}).items():
                    _collect(f"{prefix}/clarify_judge[{rp}]", lst)
                for rp, fp_dict in (o.get("final_judge") or {}).items():
                    for fp, lst in fp_dict.items():
                        _collect(f"{prefix}/final_judge[{rp}][{fp}]", lst)

    if not lengths:
        print("No outputs found.")
        return

    max_len = max(lengths.values())
    shorter = {fname: n for fname, n in lengths.items() if n < max_len}

    print(f"Expected length: {max_len}")
    if shorter:
        print(f"WARNING — {len(shorter)} file(s) are shorter:")
        for fname, n in shorter.items():
            print(f"  {fname}: {n}")
    else:
        print("All files have consistent lengths.")

    if id_orders:
        ref_fname, ref_ids = next(iter(id_orders.items()))
        misaligned = {
            fname: ids for fname, ids in id_orders.items()
            if ids != ref_ids[:len(ids)]
        }
        if misaligned:
            print(f"\nWARNING — {len(misaligned)} file(s) have misaligned question IDs (vs {ref_fname}):")
            for fname in misaligned:
                print(f"  {fname}")
        else:
            print("All question ID orders are aligned.")


def check_ref_consistency(outputs):
    """Print reference consistency checks across disambig / user / final_judge."""
    print("── disambig vs user ──────────────────────────────────────────────────")
    for model, dps in outputs.items():
        for dp, bss in dps.items():
            for bs, o in bss.items():
                d_refs = {item['id']: _ref_disambig(item)
                          for item in o.get('disambiguated', []) if _ref_disambig(item)}
                ref_by_id = {}
                for rp, items in o.get('user', {}).items():
                    for item in items:
                        r = _ref_user(item)
                        if r and item['id'] in d_refs:
                            ref_by_id.setdefault(item['id'], set()).update([d_refs[item['id']], r])
                _show_unstable(f"{model}/{dp}/{bs}", ref_by_id)

    print("\n── user vs final_judge ───────────────────────────────────────────────")
    for model, dps in outputs.items():
        for dp, bss in dps.items():
            for bs, o in bss.items():
                u_refs = {}
                for rp, items in o.get('user', {}).items():
                    for item in items:
                        r = _ref_user(item)
                        if r:
                            u_refs[item['id']] = r
                ref_by_id = {}
                for rp, fp_dict in (o.get('final_judge') or {}).items():
                    for fp, items in fp_dict.items():
                        for item in items:
                            r = _ref_judge(item)
                            if r and item['id'] in u_refs:
                                ref_by_id.setdefault(item['id'], set()).update([u_refs[item['id']], r])
                if ref_by_id:
                    _show_unstable(f"{model}/{dp}/{bs}", ref_by_id)

    print("\n── across runs (user refs, all models / direct_prompts / reasoner_prompts) ──")
    ref_by_id = {}
    for model, dps in outputs.items():
        for dp, bss in dps.items():
            for bs, o in bss.items():
                for rp, items in o.get('user', {}).items():
                    for item in items:
                        r = _ref_user(item)
                        if r:
                            ref_by_id.setdefault(item['id'], set()).add(r)
    _show_unstable("all", ref_by_id)


# ── Length analysis ───────────────────────────────────────────────────────────

def _lens(items, extract_fn):
    return [len(t.split()) for item in items for t in [extract_fn(item)] if t]

def _extract_direct_gen(item):
    ans = (item.get('generations') or {}).get('answer') or []
    if not ans:
        return ''
    g = ans[0]
    return g['raw_response'] if isinstance(g, dict) else g


def _extract_final_gen(item):
    """Extract plain text from a final item (list=prompt format, dict=prompt1/belief format)."""
    gens = item.get('generations')
    if isinstance(gens, list):
        return gens[0] if gens else ''
    if isinstance(gens, dict):
        return gens.get('response') or '' if gens.get('strategy') == 'direct_answer' else ''
    return ''

def plot_length_distributions(outputs, config, brevity=None, split_belief=True, split_final=True):
    """Violin plot of generation length by pipeline step."""
    if brevity is None:
        brevity = config['direct_prompts'][0]
    models        = config['assistant_models']
    belief_prompts = [rp for rp in config['reasoner_prompts'] if rp != 'prompt']
    final_prompts  = config.get('final_prompts', ['prompt'])

    assert brevity in outputs[models[0]], \
        f"direct_prompt '{brevity}' not loaded; available: {list(outputs[models[0]])}"

    ncols = 2
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, am in zip(axes, models):
        belief_keys = belief_prompts if split_belief else ['reasoner\n(belief*)']
        final_keys  = [f'final\n({fp})' if fp != 'prompt' else 'final' for fp in final_prompts] if split_final else ['final']
        data = {'direct': [], 'reasoner\n(prompt)': [], **{k: [] for k in belief_keys},
                **{k: [] for k in final_keys}}
        for bs in config['belief_samplings']:
            out = outputs[am][brevity][bs]
            data['direct'] += _lens(out['direct'],
                lambda x: (x['generations'].get('answer') or [{}])[0].get('raw_response', ''))
            if 'prompt' in out['reasoner']:
                data['reasoner\n(prompt)'] += _lens(
                    out['reasoner']['prompt'],
                    lambda x: x.get('generations', {}).get('response', ''))
            for rp in belief_prompts:
                if rp in out['reasoner']:
                    key = rp if split_belief else 'reasoner\n(belief*)'
                    data[key] += _lens(
                        out['reasoner'][rp],
                        lambda x: x.get('generations', {}).get('response', ''))
            for rp in config['reasoner_prompts']:
                for fp in final_prompts:
                    if rp in out.get('final', {}) and fp in out['final'][rp]:
                        key = (f'final\n({fp})' if fp != 'prompt' else 'final') if split_final else 'final'
                        data[key] += _lens(out['final'][rp][fp], _extract_final_gen)

        labels = list(data.keys())
        parts  = ax.violinplot([data[k] for k in labels], positions=range(len(labels)),
                               showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(am, fontsize=13)
        ax.set_ylabel('Generation length (words)', fontsize=11)
        ax.tick_params(axis='y', labelsize=10)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle(f'Generation length distributions ({brevity}, all models) | split={config.get("split", "?")} | seed={config["seed"]}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

def print_truncation_rates(outputs, config, brevity=None):
    """Print no-terminal-punctuation truncation rates for direct and reasoner steps."""
    if brevity is None:
        brevity = config['direct_prompts'][0]

    print(f"Direct answer truncation rate ({brevity}, no terminal punctuation):")
    for am in config['assistant_models']:
        rates = []
        for bs in config['belief_samplings']:
            texts = [(item['generations'].get('answer') or [{}])[0].get('raw_response', '')
                     for item in outputs[am][brevity][bs]['direct']]
            texts = [t for t in texts if t]
            rates.append(sum(1 for t in texts if t[-1] not in '.!?"\'') / len(texts) * 100)
        print(f"  {am}: {sum(rates)/len(rates):.1f}%")

    print()
    print("Reasoner-direct truncation rate (DIRECT_ANSWER responses, no terminal punctuation):")
    for am in config['assistant_models']:
        for rp in config['reasoner_prompts']:
            responses = []
            for bs in config['belief_samplings']:
                for dp in config['direct_prompts']:
                    for item in outputs[am][dp][bs]['reasoner'].get(rp, []):
                        if item.get('generations', {}).get('strategy') == 'direct_answer':
                            r = item['generations'].get('response') or ''
                            if r:
                                responses.append(r)
            if not responses:
                continue
            rate = sum(1 for r in responses if r[-1] not in '.!?"\'') / len(responses) * 100
            print(f"  {am} | {rp}: {rate:.1f}%  (n={len(responses)})")


# ── Strategy distribution ─────────────────────────────────────────────────────

def plot_strategy_distribution(outputs, config, final_prompt=None, brevities=None, models=None):
    """Horizontal stacked bar chart of reasoner strategy distribution.

    Args:
        final_prompt: Which final step to use when splitting clarify→abstain.
            Defaults to config['final_prompts'][0] (typically 'prompt', forced answer).
            Pass 'prompt1' or 'belief' to show a separate clarify→abs segment for
            examples where the final step abstained after the clarification turn.
        brevities: List of direct_prompts (brevity values) to show, e.g. ['vanilla'].
            Defaults to all direct_prompts in config.
        models: List of assistant_models to show, e.g. ['qwen3-14b', 'gemini'].
            Defaults to all assistant_models in config.
    """
    if final_prompt is None:
        final_prompt = config.get('final_prompts', ['prompt'])[0]

    direct_prompts  = [dp for dp in config['direct_prompts']   if brevities is None or dp in brevities]
    assistant_models = [am for am in config['assistant_models'] if models is None or any(m in am for m in models)]

    show_clarify_abstain = final_prompt != 'prompt'

    if show_clarify_abstain:
        found = any(
            _resolve_fp(rp, final_prompt, config) in outputs[am][dp][bs].get('final', {}).get(rp, {})
            for am in assistant_models
            for dp in direct_prompts
            for bs in config['belief_samplings']
            for rp in config['reasoner_prompts']
        )
        if not found:
            print(f"Warning: final_prompt='{final_prompt}' not found in outputs — falling back to 'prompt' (no clarify→abs split).")
            show_clarify_abstain = False

    if show_clarify_abstain:
        strategy_order  = ['clarification_question', 'clarify_abstain', 'direct_answer', 'abstain', 'None']
        strategy_labels = ['clarify→ans', 'clarify→abs', 'direct', 'abstain', 'error']
        strategy_colors = ['#4CAF50', '#009688', '#2196F3', '#FF9800', '#F44336']
    else:
        strategy_order  = ['clarification_question', 'direct_answer', 'abstain', 'None']
        strategy_labels = ['clarify', 'direct', 'abstain', 'error']
        strategy_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

    rows = {}
    for am in assistant_models:
        for rp in config['reasoner_prompts']:
            for bs in config['belief_samplings']:
                for dp in direct_prompts:
                    counts = collections.Counter(
                        c['generations']['strategy'] or 'None'
                        for c in outputs[am][dp][bs]['reasoner'][rp]
                    )
                    if show_clarify_abstain:
                        final_items = outputs[am][dp][bs].get('final', {}).get(rp, {}).get(_resolve_fp(rp, final_prompt, config), [])
                        n_clarify_abstain = sum(
                            1 for item in final_items
                            if isinstance(item.get('generations'), dict) and item['generations'].get('strategy') == 'abstain'
                        )
                        counts['clarification_question'] -= n_clarify_abstain
                        counts['clarify_abstain'] = n_clarify_abstain
                    rows[f"{am}_{dp}_{bs}_{rp}"] = counts

    n_models   = len(assistant_models)
    n_brevitys = len(direct_prompts)
    fig, axes  = plt.subplots(
        n_brevitys, n_models,
        figsize=(4 * n_models, 0.62 * len(config['reasoner_prompts']) * n_brevitys + 1.9),
        sharey='row', sharex=True, squeeze=False,
    )

    for row, dp in enumerate(direct_prompts):
        for col, am in enumerate(assistant_models):
            ax = axes[row, col]
            bs = config['belief_samplings'][0]
            proportions = {s: [] for s in strategy_order}
            for rp in config['reasoner_prompts']:
                counts = rows[f"{am}_{dp}_{bs}_{rp}"]
                total  = sum(counts.values())
                for s in strategy_order:
                    proportions[s].append(counts.get(s, 0) / total * 100)

            y    = np.arange(len(config['reasoner_prompts']))
            left = np.zeros(len(y))
            for s, label, color in zip(strategy_order, strategy_labels, strategy_colors):
                vals = np.array(proportions[s])
                hatch = '//' if s == 'clarify_abstain' else None
                ax.barh(y, vals, left=left, color=color, alpha=0.85, hatch=hatch,
                        label=label if (row == 0 and col == 0) else None)
                for i, (v, l) in enumerate(zip(vals, left)):
                    if v > 8:
                        ax.text(l + v / 2, i, f'{v:.0f}%', ha='center', va='center',
                                fontsize=9, color='white', fontweight='bold')
                left += vals

            ax.set_yticks(y)
            ax.set_xlim(0, 100)
            ax.tick_params(axis='x', labelsize=15)
            if row == 0:
                ax.set_title(_prettify_model(am), fontsize=20, fontweight='bold')
            if col == 0 and n_brevitys > 1:
                ax.set_ylabel(_prettify_direct_prompt(dp), fontsize=18)
            if row == n_brevitys - 1:
                ax.set_xlabel('% of examples', fontsize=17)

    def _rp_label(rp):
        base = _prettify_reasoner_prompt(rp)
        if not show_clarify_abstain:
            return base
        fp = _resolve_fp(rp, final_prompt, config)
        has_data = fp != 'prompt' and any(
            fp in outputs[am][dp][bs].get('final', {}).get(rp, {})
            for am in assistant_models
            for dp in direct_prompts
            for bs in config['belief_samplings']
        )
        return base + '+' if has_data else base

    for row in range(n_brevitys):
        axes[row, 0].set_yticklabels([_rp_label(rp) for rp in config['reasoner_prompts']], fontsize=19, fontweight='bold')
        axes[row, 0].tick_params(axis='y', labelleft=True)

    fp_label = f' | final={final_prompt}' if show_clarify_abstain else ''
    fig.legend(loc='lower center', ncol=len(strategy_order), prop={'size': 19, 'weight': 'bold'}, bbox_to_anchor=(0.5, -0.04))
    # plt.suptitle(f'Strategy distribution by model, direct prompt, and reasoner{fp_label} | split={config.get("split", "?")} | seed={config["seed"]}',
    #              fontsize=15)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()


# ── Summary tables ───────────────────────────────────────────────────────────

def style_table(df, caption="", gradient_by="global"):
    """Style the pipeline summary table.

    gradient_by:
      "global" (default) — single colour scale across all cells.
      "model"            — each model group gets its own scale; reveals within-model col trends.
    """
    level1 = [c[1] for c in df.columns]
    has_disambig = 'disambig 1' in level1

    table_styles = [{"selector": "th.col_heading", "props": [("text-align", "center"), ("white-space", "normal"),
                                                               ("word-break", "break-word"), ("max-width", "60px")]}]
    styler = df.style.format("{:.1f}").set_caption(caption)

    if has_disambig:
        sep_idx = level1.index('disambig 1') + 1
        sep_col = df.columns[sep_idx]
        styler = styler.apply(lambda col: ['border-left: 2px solid #555'] * len(col), subset=[sep_col], axis=0)
        table_styles.append({"selector": f"th.col_heading.level1:nth-child({sep_idx + 3})",
                              "props": [("border-left", "2px solid #555")]})

    styler = styler.set_table_styles(table_styles)
    if gradient_by == "global":
        vmin, vmax = df.min().min(), df.max().max()
        styler = styler.background_gradient(cmap="RdYlGn", vmin=vmin, vmax=vmax)
    else:
        for model in df.index.get_level_values(0).unique():
            model_data = df.loc[model]
            vmin, vmax = model_data.min().min(), model_data.max().max()
            styler = styler.background_gradient(
                cmap="RdYlGn",
                subset=(pd.IndexSlice[model, :], df.columns),
                vmin=vmin, vmax=vmax,
            )
    return styler


def _build_pipeline_df(metrics, config, metric_type, ref_type, warn=False):
    """Build the pipeline summary DataFrame (and counts); shared by display and latex functions."""
    j = metric_type == "judge"
    key_default         = "direct_judge"                        if j else "direct_answer_randomref"
    key_default_any     = "direct_judge_anyref"                 if j else "direct_answer_anyref"
    key_belief          = "belief_judge"                        if j else "direct_belief_randomref"
    key_disambig        = "full_disambig_upperbound_judge"       if j else "full_disambig_upperbound"
    key_disambig_any    = "full_disambig_upperbound_judge_anyref" if j else "full_disambig_upperbound_anyref"
    key_interactive     = ("full_interactive_judge_anyref" if ref_type == "any" else "full_interactive_judge") if j else "full_interactive"

    filter_none = lambda l: [elem for elem in l if elem is not None]
    safe_mean   = lambda l: np.mean(l) if l else np.nan
    summary_table = collections.defaultdict(dict)
    counts_table  = collections.defaultdict(dict)

    for assistant_model in config['assistant_models']:
        for direct_prompt in config['direct_prompts']:
            for belief_sampling in config['belief_samplings']:
                row_key = (assistant_model, direct_prompt)
                for reasoner_prompt in config['reasoner_prompts']:
                    for final_prompt in config.get('final_prompts', ['prompt']):
                        if not _valid_fp(reasoner_prompt, final_prompt):
                            continue
                        m = metrics[assistant_model][direct_prompt][belief_sampling][reasoner_prompt][final_prompt]
                        fp_part = "+" if final_prompt != 'prompt' else ""
                        col = f"{_prettify_reasoner_prompt(reasoner_prompt)}{fp_part} {ref_type}"
                        summary_table[row_key]["Standard"]          = safe_mean(filter_none(m[key_default]))
                        summary_table[row_key]["Disambig"]          = safe_mean(filter_none(m[key_disambig]))
                        summary_table[row_key]["Standard (any)"]    = safe_mean(filter_none(m[key_default_any]))
                        summary_table[row_key]["Disambig (any)"]    = safe_mean(filter_none(m[key_disambig_any]))
                        summary_table[row_key]["#words"] = safe_mean(filter_none(m["direct_answer_length"]))

                        # summary_table[row_key]["Belief"]           = safe_mean(filter_none(m[key_belief]))
                        summary_table[row_key][col]                 = safe_mean(filter_none(m[key_interactive]))
                        counts_table[row_key]["Standard"]           = sum(1 for v in m[key_default]      if v is not None)
                        counts_table[row_key]["Disambig"]           = sum(1 for v in m[key_disambig]     if v is not None)
                        counts_table[row_key]["Standard (any)"]     = sum(1 for v in m[key_default_any]  if v is not None)
                        counts_table[row_key]["Disambig (any)"]     = sum(1 for v in m[key_disambig_any] if v is not None)
                        # counts_table[row_key]["Belief"]            = sum(1 for v in m[key_belief]        if v is not None)
                        counts_table[row_key][col]                  = sum(1 for v in m[key_interactive]  if v is not None)
                        if warn:
                            if metric_type == "rouge":
                                diff = np.mean(filter_none(m["full_baseline"])) - np.mean(m["direct_answer_randomref"])
                                if abs(diff) > 0.01:
                                    print(f"WARNING! Full baseline is {diff*100:.1f}% different than direct_answer {row_key}_{reasoner_prompt}/{final_prompt} with lengths {len(filter_none(m['full_baseline']))}")
                            _key_interactive_1 = "full_interactive_judge" if j else "full_interactive"
                            n_interactive = sum(1 for v in m[_key_interactive_1] if v is not None)
                            if n_interactive < 100:
                                print(f"WARNING! Only {n_interactive} examples have interactive output for {row_key}_{reasoner_prompt}/{final_prompt} — pipeline likely incomplete, metrics unreliable")

    df = pd.DataFrame(summary_table).T.round(3) * 100
    if '#words' in df.columns:
        df['#words'] = df['#words'] / 100
    df.index = pd.MultiIndex.from_tuples([(_prettify_model(m), _prettify_direct_prompt(dp)) for m, dp in df.index])
    df_counts = pd.DataFrame(counts_table).T
    df_counts.index = pd.MultiIndex.from_tuples([(_prettify_model(m), _prettify_direct_prompt(dp)) for m, dp in df_counts.index])
    return df, df_counts


def build_and_display_metrics_tables(metrics, config, metric_type, ref_type, gradient_by="global", direct_prompt=None, final_variant="all", include_words_col=False):
    from IPython.display import display
    df, df_counts = _build_pipeline_df(metrics, config, metric_type, ref_type, warn=True)

    if direct_prompt is not None:
        pretty = _prettify_direct_prompt(direct_prompt)
        df        = df.loc[df.index.get_level_values(1) == pretty]
        df_counts = df_counts.loc[df_counts.index.get_level_values(1) == pretty]

    _DIRECT_COLS_1   = {"Standard", "Disambig", "Belief"}
    _DIRECT_COLS_ANY = {"Standard (any)", "Disambig (any)"}
    _DIRECT_COLS     = _DIRECT_COLS_1 | _DIRECT_COLS_ANY

    if include_words_col:
        _DIRECT_COLS_1 = _DIRECT_COLS_1 | {"#words"}
        _DIRECT_COLS   = _DIRECT_COLS_1 | _DIRECT_COLS_ANY
    elif '#words' in df.columns:
        df = df.drop(columns=['#words'])

    if final_variant == "+":
        drop = [c for c in df.columns if c not in _DIRECT_COLS and "+" not in c]
        df        = df.drop(columns=drop)
        df_counts = df_counts.drop(columns=drop)
    elif final_variant == "prompt":
        drop = [c for c in df.columns if c not in _DIRECT_COLS and "+" in c]
        df        = df.drop(columns=drop)
        df_counts = df_counts.drop(columns=drop)

    def _col_to_tuple(c):
        if c in _DIRECT_COLS_1:
            return ("Direct Generation", "1 intent", c)
        if c in _DIRECT_COLS_ANY:
            return ("Direct Generation", "any intent", c.replace(" (any)", ""))
        return ("Strategy-Augmented Generation", "1 intent", c.removesuffix(" 1"))

    df_display = df.copy()
    df_display.columns = pd.MultiIndex.from_tuples([_col_to_tuple(c) for c in df.columns])
    df_display.index.names = [None, 'brevity']
    if df_display.index.get_level_values('brevity').nunique() == 1:
        df_display  = df_display.droplevel('brevity')
        df_counts   = df_counts.droplevel(1)
    caption = f"Summary | {metric_type} | split={config.get('split', '?')} | seed={config['seed']} | evergreen={config['filter_evergreen']} | brevity: {direct_prompt}"
    if direct_prompt is not None:
        caption += f" | {direct_prompt}"
    words_tuple = ("Direct Generation", "1 intent", "#words")
    no_grad = (words_tuple,) if include_words_col and words_tuple in df_display.columns else ()
    display(style_direct_table(df_display, caption=caption, gradient_by=gradient_by, no_gradient_cols=no_grad))

    df_counts.columns = pd.MultiIndex.from_tuples([("n non-None", c) for c in df_counts.columns])
    display(df_counts.style
        .set_caption("Non-None count per cell (should be ~N for all columns)")
        .set_table_styles([
            {"selector": "th.col_heading", "props": [("text-align", "center"), ("white-space", "normal"),
                                                      ("word-break", "break-word"), ("max-width", "60px")]},
        ])
    )


# ── Pipeline errors ───────────────────────────────────────────────────────────

def pipeline_error_summary(outputs, config, collapse=None):
    """Return a DataFrame summarising user-leak and pipeline completeness metrics.

    Columns and their denominators:
      user %              — user has ref / total user responses
      user & !cq | flip % — user has ref & CQ doesn't & flip / total flips
      user_errors %       — user entries with no response / n_user_entries
      missing_user %      — clarify examples with no user entry / n_clarified
      missing_final %     — user entries with no final output / n_user_entries

    collapse: None → one row per config (default)
              "model" → one row per assistant model
              "full"  → single row across all models
    """
    from evaluation_utils import information_leak, _judge_verdict, evaluate_rouge

    _idk_re = re.compile(r"i\s*don.?t\s*know|idk|no\s*idea", re.IGNORECASE)
    raw = {}  # key → collections.Counter of raw counts
    for am in config['assistant_models']:
        for dp in config['direct_prompts']:
            for rp in config['reasoner_prompts']:
                for bs in config['belief_samplings']:
                    o = outputs[am][dp][bs]
                    reasoner_output    = o['reasoner'][rp]
                    user_output        = o['user'][rp]
                    final_output       = next(iter(o['final'][rp].values()), [])
                    direct_judge_items = o.get('direct_judge') or []
                    fp_dict            = (o.get('final_judge') or {}).get(rp, {})
                    final_judge_items  = next(iter(fp_dict.values()), []) if fp_dict else []

                    leak_metrics = information_leak(user_output, final_output)
                    c = leak_metrics['count']

                    n_flips = n_flip_and_leak = 0
                    for i, u in enumerate(user_output):
                        dj = _judge_verdict(direct_judge_items[i] if i < len(direct_judge_items) else None, 'direct')
                        fj = _judge_verdict(final_judge_items[i]  if i < len(final_judge_items)  else None, 'final')
                        is_flip = dj == 0 and fj == 1
                        if is_flip:
                            n_flips += 1
                        if not (u and u.get('generations') and u['generations'].get('response')):
                            continue
                        if not is_flip:
                            continue
                        ref  = u['context']['reference']
                        cq   = u['context']['clarification_q']
                        resp = u['generations']['response']
                        if max(evaluate_rouge(ref, resp, 'recall')) >= 1 and max(evaluate_rouge(ref, cq, 'recall')) == 0:
                            n_flip_and_leak += 1

                    n_total         = len(reasoner_output)
                    n_clarified     = sum(1 for r in reasoner_output if r['generations']['strategy'] == 'clarification_question')
                    n_user_entries  = sum(1 for u in user_output if u and u.get('context'))
                    n_user_response = sum(1 for u in user_output if u and u.get('generations') and u['generations'].get('response'))
                    n_idk           = sum(1 for u in user_output if u and u.get('generations') and u['generations'].get('response') and _idk_re.search(u['generations']['response']))
                    n_final         = sum(1 for f in final_output if f and f.get('generations'))

                    key = "all" if collapse == "full" else am if collapse == "model" else f"{am}_{dp}_{bs}_{rp}"
                    r = raw.setdefault(key, collections.Counter())
                    r['user']           += c['user']
                    r['user_only']      += c['user_only']
                    r['n_flips']        += n_flips
                    r['flip_and_leak']  += n_flip_and_leak
                    r['total']          += c['total']
                    r['user_errors']    += n_user_entries - n_user_response
                    r['n_idk']          += n_idk
                    r['n_missing_user'] += n_clarified    - n_user_entries
                    r['n_missing_final']+= n_user_entries - n_final
                    r['n_clarified']    += n_clarified
                    r['n_user_entries'] += n_user_entries
                    r['n_total']        += n_total

    def _pct(num, denom):
        return round(num / denom * 100, 1) if denom else 0.0

    rows = {
        key: (
            r['n_total'],
            _pct(r['user'],          r['total']),
            _pct(r['user_only'],     r['total']),
            _pct(r['flip_and_leak'], r['n_flips']),
            _pct(r['user_errors'],    r['n_user_entries']),
            _pct(r['n_idk'],          r['n_user_entries'] - r['user_errors']),
            _pct(r['n_missing_user'], r['n_clarified']),
            _pct(r['n_missing_final'],r['n_user_entries']),
        )
        for key, r in raw.items()
    }
    return pd.DataFrame(rows, index=[
        "n_total", "user %", "user & !cq %", "user & !cq | flip %",
        "user_errors %", "idk %", "missing_user %", "missing_final %",
    ]).T


def leak_net_flip_overlap(metrics, outputs, config, ref_type="1", final_prompt=None):
    """For each config, compute what fraction of judge net-flips co-occur with a user leak.

    A net-flip: strategy=clarify, direct_judge=0, final_judge=1.
    A leak:     user ROUGE-L recall ≥ 1 against reference AND CQ ROUGE-L recall = 0.

    Returns a DataFrame indexed by (model, reasoner_prompt) with columns:
        n_net_flip, n_leak_and_flip, leak_pct_of_flips
    """
    from evaluation_utils import evaluate_rouge, _judge_verdict

    _any = ref_type == "any"
    _key_final  = "final_judge_anyref"  if _any else "final_judge"
    _key_direct = "direct_judge_anyref" if _any else "direct_judge"

    rows = {}
    for am in config['assistant_models']:
        for dp in config['direct_prompts']:
            for bs in config['belief_samplings']:
                for rp in config['reasoner_prompts']:
                    fp = _resolve_fp(rp, final_prompt or config.get('final_prompts', ['prompt'])[0], config)
                    m  = metrics[am][dp][bs][rp][fp]
                    user_items = outputs[am][dp][bs]['user'][rp]

                    n_total    = len(m['reasoner_strategy'])
                    n_net_flip = 0
                    n_leak_and_flip = 0

                    for i in range(n_total):
                        if m['reasoner_strategy'][i] != 'clarification_question':
                            continue
                        dj = m[_key_direct][i] if i < len(m.get(_key_direct, [])) else None
                        fj = m[_key_final][i]  if i < len(m.get(_key_final,  [])) else None
                        if dj != 0 or fj != 1:
                            continue
                        n_net_flip += 1

                        u = user_items[i] if i < len(user_items) else None
                        if not (u and u.get('generations') and u['generations'].get('response')):
                            continue
                        ref = u['context']['reference']
                        cq  = u['context']['clarification_q']
                        resp = u['generations']['response']
                        cq_rouge   = max(evaluate_rouge(ref, cq,   'recall'))
                        user_rouge = max(evaluate_rouge(ref, resp, 'recall'))
                        if user_rouge >= 1 and cq_rouge == 0:
                            n_leak_and_flip += 1

                    rows[(_prettify_model(am), _prettify_reasoner_prompt(rp))] = {
                        'n_net_flip':        n_net_flip,
                        'n_leak_and_flip':   n_leak_and_flip,
                        'leak % of flips':   round(n_leak_and_flip / n_net_flip * 100, 1) if n_net_flip else float('nan'),
                    }

    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=['model', 'reasoner'])
    df[['n_net_flip', 'n_leak_and_flip']] = df[['n_net_flip', 'n_leak_and_flip']].astype(int)
    return df


# ── Routing profile & clarification effectiveness ─────────────────────────────

def _add_model_borders(df):
    """Style DataFrame: thick bottom border on the last row of each model group."""
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    models = df.index.get_level_values(0).unique()
    for model in models[:-1]:
        last_rp = df.loc[model].index[-1]
        styles.loc[(model, last_rp)] = 'border-bottom: 2px solid #555555'
    return styles


def _routing_profile_rows(metrics, config, metric_type, ref_type, settings_list, block1_cols, settings_passed=False, avg_bag_variants=False):
    """Compute routing profile row data (shared by display and latex functions)."""
    _j        = metric_type == "judge"
    _any      = ref_type == "any"
    _key_base = ("direct_judge_anyref" if _any else "direct_judge") if _j else "direct_answer_randomref"

    rows = {}
    accum = {}       # row_key -> list of row dicts, used when avg_bag_variants=True
    key_order = []   # preserves insertion order for correct per-model grouping
    for assistant_model, dp in settings_list:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                fp = config.get('final_prompts', ['prompt'])[0]
                m = metrics[assistant_model][dp][belief_sampling][reasoner_prompt][fp]
                n_total = len(m['final'])

                is_clarify = [m['reasoner_strategy'][i] == 'clarification_question' for i in range(n_total)]
                is_direct  = [m['reasoner_strategy'][i] == 'direct_answer'          for i in range(n_total)]
                is_abstain = [m['reasoner_strategy'][i] == 'abstain'                for i in range(n_total)]
                is_ambig   = [m['num_disambigs'][i] > 0                             for i in range(n_total)]

                n_c = sum(is_clarify)
                n_d = sum(is_direct)
                n_a = sum(is_abstain)

                def subset_base(mask):
                    vals = [m[_key_base][i] for i in range(n_total) if mask[i] and m[_key_base][i] is not None]
                    return (sum(vals) / len(vals) * 100) if vals else float('nan')

                def pct_ambig(mask):
                    n = sum(mask)
                    if n == 0: return float('nan')
                    return sum(1 for i in range(n_total) if mask[i] and is_ambig[i]) / n * 100

                dp_suffix = f"_{dp}" if settings_passed else ""
                fp_suffix = "+" if fp != 'prompt' else ""
                is_bag    = reasoner_prompt.startswith('belief')

                if avg_bag_variants and is_bag:
                    rp_key = f"belief_avg{fp_suffix}{dp_suffix}"
                else:
                    rp_key = f"{reasoner_prompt}{fp_suffix}{dp_suffix}"
                routing_row_key = (assistant_model, rp_key)

                row = {}
                if 'routing' in block1_cols:
                    row.update({
                        'clarify %': round(n_c / n_total * 100, 1),
                        'direct %':  round(n_d / n_total * 100, 1),
                        'abstain %': round(n_a / n_total * 100, 1),
                    })
                if 'quality' in block1_cols:
                    row.update({
                        'acc(C)': round(subset_base(is_clarify), 1),
                        'acc(D)': round(subset_base(is_direct),  1),
                        'acc(A)': round(subset_base(is_abstain), 1),
                    })
                if 'ambig' in block1_cols:
                    row.update({
                        'amb(C)': round(pct_ambig(is_clarify), 1),
                        'amb(D)': round(pct_ambig(is_direct),  1),
                        'amb(A)': round(pct_ambig(is_abstain), 1),
                    })

                if avg_bag_variants and is_bag:
                    if routing_row_key not in accum:
                        key_order.append(routing_row_key)
                    accum.setdefault(routing_row_key, []).append(row)
                elif routing_row_key not in rows:
                    key_order.append(routing_row_key)
                    rows[routing_row_key] = row

    if avg_bag_variants:
        for key, row_list in accum.items():
            avg_row = {}
            for col in row_list[0]:
                vals = [r[col] for r in row_list if not np.isnan(r[col])]
                avg_row[col] = round(sum(vals) / len(vals), 1) if vals else float('nan')
            rows[key] = avg_row
        rows = {k: rows[k] for k in key_order}
    return rows


def plot_routing_profile(metrics, config, metric_type, direct_prompt=None, ref_type="1", settings=None, block1_cols=('routing', 'quality', 'ambig'), gradient_by="row", avg_bag_variants=False):
    """Display Block 1 (routing profile) table.

    Pass settings=[(model, direct_prompt), ...] to compare specific (model, dp) pairs.
    Pass block1_cols to select which column groups to show:
      'routing' -> clarify/direct/abstain %
      'quality' -> acc(C)/acc(D)/acc(A) (mean baseline score on each action subset)
      'ambig'   -> ambig(C)/direct/abstain
    gradient_by: "row" (default) | "model" | "global"
    avg_bag_variants: if True, average metrics across all belief_samplings per row
    """
    from IPython.display import display
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]
    if settings is not None:
        settings_list = settings
    else:
        settings_list = [(mdl, direct_prompt) for mdl in config['assistant_models']]

    routing_profile_rows = _routing_profile_rows(metrics, config, metric_type, ref_type, settings_list, block1_cols, settings_passed=settings is not None, avg_bag_variants=avg_bag_variants)

    df_routing_profile = pd.DataFrame(routing_profile_rows).T
    df_routing_profile.index = pd.MultiIndex.from_tuples(
        [(_prettify_model(m), _prettify_reasoner_prompt(rp)) for m, rp in df_routing_profile.index],
        names=["model", "reasoner"],
    )

    fmt_routing = {}
    if 'routing' in block1_cols:
        fmt_routing.update({'clarify %': '{:.1f}', 'direct %': '{:.1f}', 'abstain %': '{:.1f}'})
    if 'quality' in block1_cols:
        fmt_routing.update({'acc(C)': '{:.1f}', 'acc(D)': '{:.1f}', 'acc(A)': '{:.1f}'})
    if 'ambig' in block1_cols:
        fmt_routing.update({'amb(C)': '{:.1f}', 'amb(D)': '{:.1f}', 'amb(A)': '{:.1f}'})

    def _grad(styler, df, cols, cmap):
        if gradient_by == "row":
            return styler.background_gradient(subset=cols, cmap=cmap, axis=1)
        if gradient_by == "global":
            return styler.background_gradient(subset=cols, cmap=cmap, axis=0)
        for model in df.index.get_level_values(0).unique():
            styler = styler.background_gradient(
                subset=(pd.IndexSlice[model, :], cols), cmap=cmap, axis=0,
            )
        return styler

    styled_b1 = df_routing_profile.style.format(fmt_routing)
    if 'routing' in block1_cols:
        styled_b1 = _grad(styled_b1, df_routing_profile, ['clarify %', 'direct %', 'abstain %'], 'Blues')
    if 'quality' in block1_cols:
        styled_b1 = _grad(styled_b1, df_routing_profile, ['acc(C)', 'acc(D)', 'acc(A)'], 'RdYlGn')
    if 'ambig' in block1_cols:
        styled_b1 = _grad(styled_b1, df_routing_profile, ['amb(C)', 'amb(D)', 'amb(A)'], 'Blues')
    styled_b1 = styled_b1.apply(_add_model_borders, axis=None)
    display(styled_b1.set_caption(f"Block 1 — Routing profile | {metric_type} | split={config.get('split', '?')} | direct_prompt={direct_prompt if settings is None else 'see row'} | seed={config['seed']}"))


def plot_clarify_effectiveness(metrics, config, metric_type, direct_prompt=None, ref_type="1", final_prompt=None, settings=None):
    """Display Block 2 (clarification effectiveness) table.

    Pass settings=[(model, direct_prompt), ...] to compare specific (model, dp) pairs.
    """
    from IPython.display import display
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]
    if final_prompt is None:
        final_prompt = config.get('final_prompts', ['prompt'])[0]
    if settings is not None:
        settings_list = settings
    else:
        settings_list = [(mdl, direct_prompt) for mdl in config['assistant_models']]

    _j         = metric_type == "judge"
    _any       = ref_type == "any"
    _key_final = "final_judge"                                                  if _j else "final"
    _key_cbase = ("direct_judge_anyref" if _any else "direct_judge")           if _j else "direct_answer_clarifyref"

    clarify_effect_rows = {}

    for assistant_model, dp in settings_list:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                for fp in [_resolve_fp(reasoner_prompt, final_prompt, config)]:
                    m = metrics[assistant_model][dp][belief_sampling][reasoner_prompt][fp]
                    n_total = len(m['final'])

                    is_clarify = [m['reasoner_strategy'][i] == 'clarification_question' for i in range(n_total)]

                    dp_suffix = f"_{dp}" if settings is not None else ""
                    fp_suffix = "+" if fp != 'prompt' else ""
                    row_key   = (assistant_model, f"{reasoner_prompt}{fp_suffix}{dp_suffix}")

                    all_clarify = [
                        (m[_key_final][i], m[_key_cbase][i])
                        for i in range(n_total)
                        if is_clarify[i] and m[_key_cbase][i] is not None
                    ]
                    pairs     = [(fi, bi) for fi, bi in all_clarify if fi is not None]
                    abstained = [(fi, bi) for fi, bi in all_clarify if fi is None]

                    wrong_total    = [bi for fi, bi in all_clarify if bi == 0]
                    right_total    = [bi for fi, bi in all_clarify if bi == 1]
                    wrong_answered = [(fi, bi) for fi, bi in pairs if bi == 0]
                    right_answered = [(fi, bi) for fi, bi in pairs if bi == 1]

                    recovery   = sum(fi for fi, bi in wrong_answered)     / len(wrong_total) * 100 if wrong_total else float('nan')
                    regression = sum(1 - fi for fi, bi in right_answered) / len(right_total) * 100 if right_total else float('nan')
                    net_flip   = sum(fi - bi for fi, bi in pairs) / len(all_clarify) * 100 if all_clarify else float('nan')

                    row = {
                        'n_clarify_judged':  len(pairs),
                        'n_wrong_base':      len(wrong_total),
                        'n_right_base':      len(right_total),
                        'recovery rate %':   round(recovery, 1),
                        'regression rate %': round(regression, 1),
                        'net flip /N %':     round(net_flip, 1),
                    }
                    if fp != 'prompt':
                        abstain_wrong_pct = sum(1 for fi, bi in abstained if bi == 0) / len(wrong_total) * 100 if wrong_total else float('nan')
                        abstain_right_pct = sum(1 for fi, bi in abstained if bi == 1) / len(right_total) * 100 if right_total else float('nan')
                        row['abstain|wrong %'] = round(abstain_wrong_pct, 1)
                        row['abstain|right %'] = round(abstain_right_pct, 1)
                    clarify_effect_rows[row_key] = row

    df_clarify_effect = pd.DataFrame(clarify_effect_rows).T
    df_clarify_effect.index = pd.MultiIndex.from_tuples(
        [(_prettify_model(m), _prettify_reasoner_prompt(rp)) for m, rp in df_clarify_effect.index],
        names=["model", "reasoner"],
    )

    int_cols_effect = ['n_clarify_judged', 'n_wrong_base', 'n_right_base']
    fmt_effect = {c: '{:.0f}' for c in int_cols_effect}
    fmt_effect.update({'recovery rate %': '{:.1f}', 'regression rate %': '{:.1f}',
                       'net flip /N %':   '{:.1f}'})
    show_abstain_cols = final_prompt != 'prompt'
    if show_abstain_cols:
        fmt_effect.update({'abstain|wrong %': '{:.1f}', 'abstain|right %': '{:.1f}'})

    styled = (df_clarify_effect.style
        .format(fmt_effect)
        .background_gradient(subset=['recovery rate %'],   cmap='RdYlGn')
        .background_gradient(subset=['regression rate %'], cmap='RdYlGn_r')
        .background_gradient(subset=['net flip /N %'],     cmap='RdYlGn', vmin=-15, vmax=15)
        .set_caption(f"Block 2 — Clarification effectiveness | {metric_type} | split={config.get('split', '?')} | direct_prompt={direct_prompt if settings is None else 'see row'} | final_prompt={final_prompt} | seed={config['seed']} (conditioned on clarify; fixed denominators)"))
    if show_abstain_cols:
        styled = (styled
            .background_gradient(subset=['abstain|wrong %'], cmap='RdYlGn_r')
            .background_gradient(subset=['abstain|right %'], cmap='RdYlGn_r'))
    styled = styled.apply(_add_model_borders, axis=None)
    display(styled)


# ── Flip distribution ────────────────────────────────────────────────────────

def plot_flip_distribution(metrics, config, metric_type, direct_prompt=None, ref_type="1", final_prompt=None, model_display=None, compact=False):
    """Diverging bar chart: +flip (recovery) vs −flip (regression) for clarify examples.

    compact=True renders a 2×2 subplot grid sized for a 1-column paper figure.
    """
    if model_display is None:
        model_display = {}
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]
    if final_prompt is None:
        final_prompt = config.get('final_prompts', ['prompt'])[0]

    _j       = metric_type == "judge"
    _any     = ref_type == "any"
    key_final = "final_judge" if _j else "final"
    key_base  = ("direct_judge_anyref" if _any else "direct_judge") if _j else "direct_answer_clarifyref"

    rows = {}
    for assistant_model in config['assistant_models']:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                fp = _resolve_fp(reasoner_prompt, final_prompt, config)
                m = metrics[assistant_model][direct_prompt][belief_sampling][reasoner_prompt][fp]
                n_total = len(m['reasoner_strategy'])

                answered = [
                    (m[key_final][i], m[key_base][i])
                    for i in range(n_total)
                    if m['reasoner_strategy'][i] == 'clarification_question'
                    and m[key_base][i]  is not None
                    and m[key_final][i] is not None
                ]
                if not answered:
                    continue

                n     = len(answered)
                n_pos = sum(1 for f, b in answered if f > b)
                n_neg = sum(1 for f, b in answered if f < b)
                rows[(assistant_model, reasoner_prompt)] = {
                    'recovery':   n_pos / n * 100,
                    'regression': n_neg / n * 100,
                    'net':        (n_pos - n_neg) / n * 100,
                    'n':          n,
                }

    models           = config['assistant_models']
    reasoner_prompts = list(reversed(config['reasoner_prompts']))  # top = last rp

    if compact:
        _plot_flip_distribution_compact(rows, models, reasoner_prompts, model_display, metric_type, direct_prompt, final_prompt, config)
        return

    gap = 1.2
    bar_h = 0.6

    # assign y positions: one contiguous strip per model, separated by gap
    y_pos = {}
    group_centers = {}
    sep_lines = []
    current_y = 0
    for i, model in enumerate(models):
        ys = []
        for rp in reasoner_prompts:
            y_pos[(model, rp)] = current_y
            ys.append(current_y)
            current_y += 1
        group_centers[model] = np.mean(ys)
        if i < len(models) - 1:
            sep_lines.append(current_y - 0.5 + gap / 2)
        current_y += gap

    total_h = max(4, len(models) * (len(reasoner_prompts) * 0.42 + 0.5))
    fig, ax = plt.subplots(figsize=(7, total_h))

    added_labels = set()
    for (model, rp), y in y_pos.items():
        rec = rows.get((model, rp), {}).get('recovery',   0.0)
        reg = rows.get((model, rp), {}).get('regression', 0.0)
        net = rows.get((model, rp), {}).get('net',        0.0)
        kw_rec = dict(color='#4CAF50', alpha=0.85)
        kw_reg = dict(color='#F44336', alpha=0.85)
        if '+flip' not in added_labels:
            kw_rec['label'] = '+flip (recovery)'
            kw_reg['label'] = '−flip (regression)'
            added_labels.update(['+flip', '−flip'])
        ax.barh(y,  rec,  height=bar_h, **kw_rec)
        ax.barh(y, -reg,  height=bar_h, **kw_reg)
        sc_kw = dict(color='black', marker='|', s=200, zorder=5)
        if 'net' not in added_labels:
            sc_kw['label'] = 'net flip'
            added_labels.add('net')
        ax.scatter(net, y, **sc_kw)

    for sep in sep_lines:
        ax.axhline(sep, color='gray', linewidth=0.6, linestyle='-', alpha=0.4)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    all_ys     = [y_pos[(model, rp)] for model in models for rp in reasoner_prompts]
    all_labels = [
        _prettify_reasoner_prompt(rp) + ('+' if _resolve_fp(rp, final_prompt, config) != 'prompt' else '')
        for _ in models for rp in reasoner_prompts
    ]
    ax.set_yticks(all_ys)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.set_xlabel('% of answered clarify examples')

    trans = ax.get_yaxis_transform()
    for model, cy in group_centers.items():
        ax.text(1.02, cy, model_display.get(model, _prettify_model(model)),
                transform=trans, va='center', ha='left',
                fontsize=11, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10)
    ax.set_title(
        f'Flip distribution | {metric_type} | split={config.get("split", "?")} | dp={direct_prompt} | fp={final_prompt} | seed={config["seed"]}',
        fontsize=11,
    )
    plt.tight_layout()


def _plot_flip_distribution_compact(rows, models, reasoner_prompts, model_display, metric_type, direct_prompt, final_prompt, config):
    """2×2 subplot grid version of plot_flip_distribution for 1-column papers."""
    import math
    n_cols = 2
    n_rows = math.ceil(len(models) / n_cols)
    bar_h  = 0.45
    fs_tick = 8.5
    fs_label = 8.5
    fs_title = 9.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, n_rows * (len(reasoner_prompts) * 0.32 + 0.6)),
                              sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    ys     = list(range(len(reasoner_prompts)))
    labels = [
        _prettify_reasoner_prompt(rp, short=True) + ('+' if _resolve_fp(rp, final_prompt, config) != 'prompt' else '')
        for rp in reasoner_prompts
    ]

    # global x limits for shared scale
    all_vals = []
    for model in models:
        for rp in reasoner_prompts:
            d = rows.get((model, rp), {})
            all_vals += [d.get('recovery', 0), -d.get('regression', 0), d.get('net', 0)]
    xmax = max(abs(v) for v in all_vals) if all_vals else 30
    xmax = math.ceil(xmax / 5) * 5 + 5

    added_labels: set = set()
    for idx, model in enumerate(models):
        ax = axes[idx // n_cols, idx % n_cols]
        for y, rp in zip(ys, reasoner_prompts):
            rec = rows.get((model, rp), {}).get('recovery',   0.0)
            reg = rows.get((model, rp), {}).get('regression', 0.0)
            net = rows.get((model, rp), {}).get('net',        0.0)
            kw_rec = dict(color='#4CAF50', alpha=0.85)
            kw_reg = dict(color='#F44336', alpha=0.85)
            if '+flip' not in added_labels:
                kw_rec['label'] = '+flip (recovery)'
                kw_reg['label'] = '−flip (regression)'
                added_labels.update(['+flip', '−flip'])
            ax.barh(y,  rec,  height=bar_h, **kw_rec)
            ax.barh(y, -reg,  height=bar_h, **kw_reg)
            sc_kw = dict(color='black', marker='|', s=80, zorder=5, linewidths=0.8)
            if 'net' not in added_labels:
                sc_kw['label'] = 'net flip'
                added_labels.add('net')
            ax.scatter(net, y, **sc_kw)

        ax.axvline(0, color='black', linewidth=0.6, linestyle='--')
        ax.set_xlim(-xmax, xmax)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=fs_tick)
        ax.tick_params(axis='x', labelsize=fs_tick)
        ax.set_title(f"{model_display.get(model, _prettify_model(model))} ({direct_prompt})", fontsize=fs_title, fontweight='bold', pad=3)

    # hide unused subplots
    for idx in range(len(models), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # shared x-axis label on bottom row
    for col in range(n_cols):
        axes[-1, col].set_xlabel('% of clarify examples', fontsize=fs_label)

    # single legend in bottom-right subplot (or first visible)
    legend_ax = axes[(len(models) - 1) // n_cols, (len(models) - 1) % n_cols]
    legend_ax.legend(loc='lower right', fontsize=fs_tick, framealpha=0.8)

    plt.tight_layout(h_pad=0.8, w_pad=0.5)


# ── Intent recovery ───────────────────────────────────────────────────────────

def plot_intent_recovery(metrics, config, metric_type, direct_prompt=None):
    """Display intent recovery scores for the ambiguous + clarify subset."""
    from IPython.display import display
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]

    _j                 = metric_type == "judge"
    _key_final_intent  = "final_judge"         if _j else "final"
    _key_direct_1ref   = "direct_judge"        if _j else "direct_answer_randomref"
    _key_direct_anyref = "direct_judge_anyref" if _j else "direct_answer_anyref"
    _key_disambig_int  = "disambig_judge"      if _j else "disambig_answer"

    intent_rows = {}
    for assistant_model in config['assistant_models']:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                for final_prompt in config.get('final_prompts', ['prompt']):
                    if not _valid_fp(reasoner_prompt, final_prompt):
                        continue
                    m = metrics[assistant_model][direct_prompt][belief_sampling][reasoner_prompt][final_prompt]
                    n_total = len(m['reasoner_strategy'])

                    is_clarify = [s == 'clarification_question' for s in m['reasoner_strategy']]
                    is_ambig   = [n > 0 for n in m['num_disambigs']]

                    idx = [i for i in range(n_total)
                           if is_clarify[i] and is_ambig[i]
                           and m[_key_final_intent][i] is not None
                           and m[_key_direct_1ref][i]  is not None]

                    if not idx:
                        continue

                    final_scores  = [m[_key_final_intent][i]  for i in idx]
                    direct_1ref   = [m[_key_direct_1ref][i]   for i in idx]
                    direct_anyref = [m[_key_direct_anyref][i] for i in idx if m[_key_direct_anyref][i] is not None]
                    disambig_sc   = [m[_key_disambig_int][i]  for i in idx if m[_key_disambig_int][i]  is not None]

                    s = 100
                    mean_final  = np.mean(final_scores)  if final_scores  else float('nan')
                    mean_1ref   = np.mean(direct_1ref)   if direct_1ref   else float('nan')
                    mean_anyref = np.mean(direct_anyref) if direct_anyref else float('nan')
                    mean_disamb = np.mean(disambig_sc)   if disambig_sc   else float('nan')

                    intent_rows[f"{_prettify_model(assistant_model)}_{_prettify_reasoner_prompt(reasoner_prompt)}_{final_prompt}"] = {
                        "n":                 len(idx),
                        "direct (1 ref)":    round(mean_1ref   * s, 1),
                        "direct (any ref)":  round(mean_anyref * s, 1),
                        "disambig oracle":   round(mean_disamb * s, 1),
                        "final (after CQ)":  round(mean_final  * s, 1),
                        "Δ vs 1ref":         round((mean_final - mean_1ref)   * s, 1),
                        "Δ vs any ref":      round((mean_final - mean_anyref) * s, 1),
                    }

    df_intent = pd.DataFrame(intent_rows).T.apply(pd.to_numeric, errors='coerce')
    score_cols = [c for c in df_intent.columns if c != 'n']
    display(df_intent.style
        .format('{:.1f}', subset=score_cols)
        .format('{:.0f}', subset=['n'])
        .background_gradient(subset=['direct (1 ref)', 'direct (any ref)', 'disambig oracle', 'final (after CQ)'],
                             cmap='RdYlGn', axis=1)
        .background_gradient(subset=['Δ vs 1ref', 'Δ vs any ref'], cmap='RdYlGn', vmin=-10, vmax=10)
        .set_caption(f"Intent recovery — ambiguous + clarify subset | {metric_type} | split={config.get('split', '?')} | direct_prompt={direct_prompt} | seed={config['seed']}"))


# ── Cross-prompt comparison ───────────────────────────────────────────────────

def plot_cross_prompt_comparison(metrics, config, metric_type, direct_prompt=None):
    """Display Block 3: final judge score of each row prompt on shared clarified examples per model."""
    from IPython.display import display
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]

    _key_final        = "final_judge" if metric_type == "judge" else "final"
    prompts           = config['reasoner_prompts']
    cross_final_prompt = config.get('final_prompts', ['prompt'])[0]

    for assistant_model in config['assistant_models']:
        belief_sampling = config['belief_samplings'][0]

        clarify_data = {}
        for rp in prompts:
            m = metrics[assistant_model][direct_prompt][belief_sampling][rp][cross_final_prompt]
            clarify_data[rp] = {
                i: m[_key_final][i]
                for i in range(len(m['reasoner_strategy']))
                if m['reasoner_strategy'][i] == 'clarification_question' and m[_key_final][i] is not None
            }

        score_mat = pd.DataFrame(index=prompts, columns=prompts, dtype=float)
        n_mat     = pd.DataFrame(index=prompts, columns=prompts, dtype=int)

        for p_row in prompts:
            for p_col in prompts:
                shared = set(clarify_data[p_row]) & set(clarify_data[p_col])
                n_mat.loc[p_row, p_col] = len(shared)
                if shared:
                    vals = [clarify_data[p_row][i] for i in shared]
                    score_mat.loc[p_row, p_col] = round(sum(vals) / len(vals) * 100, 1)

        rename = {p: _prettify_reasoner_prompt(p) for p in prompts}
        score_mat = score_mat.rename(index=rename, columns=rename)
        n_mat     = n_mat.rename(index=rename, columns=rename)
        pretty_model = _prettify_model(assistant_model)

        vmin = score_mat.min().min()
        vmax = score_mat.max().max()

        display(score_mat.style
            .format('{:.1f}', na_rep='—')
            .background_gradient(cmap='RdYlGn', vmin=vmin, vmax=vmax)
            .set_caption(
                f"Block 3 — Final judge score of row prompt on examples shared with col prompt "
                f"| {pretty_model} | split={config.get('split', '?')} | direct_prompt={direct_prompt} | seed={config['seed']} "
                f"(read column-wise for fair CQ comparison)"))

        display(n_mat.style
            .format('{:.0f}')
            .background_gradient(cmap='Blues')
            .set_caption(f"Block 3 — n shared clarified examples per cell | {pretty_model} | split={config.get('split', '?')} | direct_prompt={direct_prompt} | seed={config['seed']}"))


# ── Routing / selection gain decomposition ────────────────────────────────────

def compute_gain_decomposition(
    metrics, config, metric_type,
    ref_type="1", direct_prompt=None, settings=None,
):
    """Compute per-setting gain decomposition. Returns a DataFrame indexed by setting key.

    Columns: n_c, n_d, n_a, n_c_pos, n_c_neg, n_a_success, abstain_success,
             abstain_gain, within_clarify, within_direct, within_branch, Δtotal, dp

    Gains (all in pp, i.e. multiplied by 100) decompose the total BAG vs default gap:
      Δtotal       = abstain_gain + within_clarify + within_direct
                   = acc(BAG on nc+nd) - acc(default on all N)
      abstain_gain = acc(default on nc+nd) - acc(default on all N)
                     Selection effect: positive when abstained questions are below-average difficulty.
      within_clarify/direct = weighted gain within each branch (weight = branch fraction of nc+nd).
    """
    _j   = metric_type == "judge"
    _any = ref_type == "any"
    key_inter   = ("full_interactive_judge_anyref" if _any else "full_interactive_judge") if _j else "full_interactive"
    key_base    = ("full_baseline_judge_anyref"    if _any else "full_baseline_judge")    if _j else "full_baseline"
    key_abstain = "direct_judge_abstain_anyref" if _any else "direct_judge_abstain"
    key_direct  = "direct_judge_anyref" if _any else "direct_judge"

    if settings is not None:
        settings_list = settings
    else:
        _dp = direct_prompt or config['direct_prompts'][0]
        settings_list = [(mdl, _dp) for mdl in config['assistant_models']]

    rows = {}
    for assistant_model, dp in settings_list:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                for final_prompt in config.get('final_prompts', ['prompt']):
                    if not _valid_fp(reasoner_prompt, final_prompt):
                        continue
                    m = metrics[assistant_model][dp][belief_sampling][reasoner_prompt][final_prompt]

                    is_clarify = [s == 'clarification_question' for s in m['reasoner_strategy']]
                    is_direct  = [s == 'direct_answer'          for s in m['reasoner_strategy']]
                    is_abstain = [s == 'abstain'                for s in m['reasoner_strategy']]

                    # compute pairs first so all means share exactly the same nc+nd subset
                    pairs_nc = [(fi, bi) for fi, bi, c in zip(m[key_inter], m[key_base], is_clarify)
                                if c and fi is not None and bi is not None]
                    pairs_nd = [(fi, bi) for fi, bi, d in zip(m[key_inter], m[key_base], is_direct)
                                if d and fi is not None and bi is not None]

                    # clarify→abstain: routed to clarify but final step produced no answer (abstain or error)
                    # full_baseline is None for these, so use direct_judge directly for the selection effect
                    base_ca = [m[key_direct][i] for i, c in enumerate(is_clarify)
                               if c and m[key_base][i] is None and m[key_direct][i] is not None]

                    # all N: direct_judge scores for every example (fixes the bug where n_ca was excluded)
                    base_N_all = [v for v in m[key_direct] if v is not None]

                    nc_nd = len(pairs_nc) + len(pairs_nd)
                    if nc_nd == 0 or not base_N_all:
                        continue

                    mean_inter_nc_nd     = np.mean([fi for fi, bi in pairs_nc + pairs_nd])
                    mean_base_nc_nd      = np.mean([bi for fi, bi in pairs_nc + pairs_nd])
                    mean_base_nc_nd_nca  = np.mean([bi for fi, bi in pairs_nc + pairs_nd] + base_ca)
                    mean_base_N          = np.mean(base_N_all)

                    # gain within nc+nd branches: acc(BAG) - acc(default), same subset
                    within_branch = mean_inter_nc_nd - mean_base_nc_nd
                    # reasoner abstain selection: baseline shifts because n_a examples are excluded from nc+nd+nca
                    abstain_gain         = mean_base_nc_nd_nca - mean_base_N
                    # clarify→abstain selection: additional shift from excluding n_ca from nc+nd
                    clarify_abstain_gain = mean_base_nc_nd - mean_base_nc_nd_nca
                    # Δtotal = acc(BAG on nc+nd) - acc(default on all N)
                    total_gain = within_branch + abstain_gain + clarify_abstain_gain

                    n_c  = sum(is_clarify)
                    n_d  = sum(is_direct)
                    n_a  = sum(is_abstain)
                    n_ca = len(base_ca)

                    # weight by branch fraction so within_clarify + within_direct = within_branch exactly
                    within_clarify = (len(pairs_nc) / nc_nd) * np.mean([fi - bi for fi, bi in pairs_nc]) if pairs_nc else 0.0
                    within_direct  = (len(pairs_nd) / nc_nd) * np.mean([fi - bi for fi, bi in pairs_nd]) if pairs_nd else 0.0

                    assert abs(abstain_gain + clarify_abstain_gain + within_clarify + within_direct - total_gain) < 1e-9, (
                        f"Gain decomposition does not sum to Δtotal: "
                        f"{abstain_gain:.6f} + {clarify_abstain_gain:.6f} + {within_clarify:.6f} + {within_direct:.6f} "
                        f"!= {total_gain:.6f}"
                    )

                    n_c_pos = sum(1 for fi, bi in pairs_nc if fi - bi >= 1.0)
                    n_c_neg = sum(1 for fi, bi in pairs_nc if bi - fi >= 1.0)

                    abstain_scores  = [v for v in m[key_abstain] if v is not None]
                    n_a_success     = sum(1 for s in abstain_scores if s == 0)
                    abstain_success = n_a_success / len(abstain_scores) if abstain_scores else 0.0

                    fp_part = "+" if final_prompt != 'prompt' else ""
                    rows[f"{assistant_model}_{reasoner_prompt}{fp_part}"] = {
                        "n_c": n_c, "n_d": n_d, "n_a": n_a, "n_ca": n_ca,
                        "n_c_pos": n_c_pos, "n_c_neg": n_c_neg,
                        "n_a_success": n_a_success,
                        "abstain_success":      round(abstain_success      * 100, 1),
                        "abstain_gain":         round(abstain_gain         * 100, 1),
                        "clarify_abstain_gain": round(clarify_abstain_gain * 100, 1),
                        "within_clarify":       round(within_clarify       * 100, 1),
                        "within_direct":        round(within_direct        * 100, 1),
                        "within_branch":        round(within_branch        * 100, 1),
                        "Δtotal":               round(total_gain           * 100, 1),
                        "dp":                   dp,
                    }

    return pd.DataFrame(rows).T


def plot_routing_decomposition(metrics, config, metric_type, ref_type="1", direct_prompt=None, settings=None, show_labels=True, variant="all"):
    """Bar chart decomposing total BAG gain (vs default) into abstain gain and within-branch gains.

    Pass settings=[(model, direct_prompt), ...] to compare specific (model, dp) pairs.
    show_labels toggles the fraction/% annotations inside the bars.
    variant: "all" | "+" | "prompt" — filter to plus, non-plus, or all final_prompt variants.
    """
    df_routing = compute_gain_decomposition(
        metrics, config, metric_type,
        ref_type=ref_type, direct_prompt=direct_prompt, settings=settings,
    )

    if variant == "+":
        df_routing = df_routing[df_routing.index.str.contains(r'\+', regex=True)]
    elif variant == "prompt":
        df_routing = df_routing[~df_routing.index.str.contains(r'\+', regex=True)]

    if settings is not None:
        settings_list = settings
    else:
        _dp = direct_prompt or config['direct_prompts'][0]
        settings_list = [(mdl, _dp) for mdl in config['assistant_models']]

    def _split_label(label):
        for mdl in config['assistant_models']:
            if label.startswith(mdl + '_'):
                return mdl, label[len(mdl)+1:]
        return label, ''

    row_labels = df_routing.index.tolist()
    parsed     = [_split_label(s) for s in row_labels]
    mdls     = [p[0] for p in parsed]
    variants = [p[1] for p in parsed]

    model_order    = list(dict.fromkeys(mdls))
    n_models       = len(model_order)
    rows_per_model = {mdl: mdls.count(mdl) for mdl in model_order}

    colors = {
        'abstain':          '#FF9800',
        'clarify_abstain':  '#009688',
        'within_clarify':   '#4CAF50',
        'within_direct':    '#2196F3',
    }

    ncols = 1
    nrows = n_models
    row_heights = [max(1.8, rows_per_model[m] * 0.45) for m in model_order]
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6, sum(row_heights)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    legend_handles = {}  # label -> handle, only populated when any bar in that category is non-zero
    unique_dps = df_routing['dp'].unique()
    show_dp = not (len(unique_dps) == 1 and unique_dps[0] == 'vanilla')

    for ax, model in zip(axes_flat, model_order):
        mask     = [mdl == model for mdl in mdls]
        idx      = [i for i, m in enumerate(mask) if m]
        sub_vars = [_prettify_reasoner_prompt(variants[i]) for i in idx]
        sub_df   = df_routing.iloc[idx]
        y        = np.arange(len(idx))

        abstain_gain         = sub_df['abstain_gain'].values
        clarify_abstain_gain = sub_df['clarify_abstain_gain'].values
        within_clarify       = sub_df['within_clarify'].values
        within_direct        = sub_df['within_direct'].values

        pos_abstain_gain   = np.clip(abstain_gain,         0, None)
        neg_abstain_gain   = np.clip(abstain_gain,         None, 0)
        pos_ca_gain        = np.clip(clarify_abstain_gain, 0, None)
        neg_ca_gain        = np.clip(clarify_abstain_gain, None, 0)
        pos_within_clarify = np.clip(within_clarify,       0, None)
        neg_within_clarify = np.clip(within_clarify,       None, 0)
        pos_within_direct  = np.clip(within_direct,        0, None)
        neg_within_direct  = np.clip(within_direct,        None, 0)

        def _track(handle, label, has_data):
            if has_data and label not in legend_handles:
                legend_handles[label] = handle

        b = ax.barh(y, pos_abstain_gain, color=colors['abstain'], alpha=0.85)
        _track(b, 'abstain', np.any(pos_abstain_gain > 0))
        b = ax.barh(y, pos_ca_gain, left=pos_abstain_gain, color=colors['clarify_abstain'], alpha=0.85, hatch='//')
        _track(b, 'clarify→abstain', np.any(pos_ca_gain > 0))
        b = ax.barh(y, pos_within_clarify, left=pos_abstain_gain + pos_ca_gain, color=colors['within_clarify'], alpha=0.85)
        _track(b, 'clarify', np.any(pos_within_clarify > 0))
        b = ax.barh(y, pos_within_direct, left=pos_abstain_gain + pos_ca_gain + pos_within_clarify,
                    color=colors['within_direct'], alpha=0.85)
        _track(b, 'direct', np.any(pos_within_direct > 0))

        ax.barh(y, neg_abstain_gain, color=colors['abstain'], alpha=0.85)
        ax.barh(y, neg_ca_gain, left=neg_abstain_gain, color=colors['clarify_abstain'], alpha=0.85, hatch='//')
        ax.barh(y, neg_within_clarify, left=neg_abstain_gain + neg_ca_gain, color=colors['within_clarify'], alpha=0.85)
        ax.barh(y, neg_within_direct, left=neg_abstain_gain + neg_ca_gain + neg_within_clarify,
                    color=colors['within_direct'], alpha=0.85)

        sc = ax.scatter(sub_df['Δtotal'].values, y, color='black', zorder=5, marker='|', s=200)
        _track(sc, 'Δtotal', True)

        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_yticks(y)
        ax.set_yticklabels(sub_vars, fontsize=11)
        dp_label = sub_df['dp'].iloc[0]
        title = f"{_prettify_model(model)} | {dp_label}" if show_dp else _prettify_model(model)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_facecolor(ax.get_facecolor())

        if show_labels:
            for i, row in enumerate(sub_df.itertuples()):
                nc  = int(row.n_c)
                na  = int(row.n_a)
                nca = int(row.n_ca)
                net = int(row.n_c_pos - row.n_c_neg)

                if na > 0 and pos_abstain_gain[i] > 0:
                    x_na = pos_abstain_gain[i] / 2
                    n_as = int(row.n_a_success)
                    ax.text(x_na, i, f"{n_as}/{na}\n({row.abstain_success:.0f}%)",
                            va='center', ha='center', fontsize=9,
                            color='white', fontweight='bold', clip_on=True)

                if nca > 0 and pos_ca_gain[i] > 0:
                    x_ca = pos_abstain_gain[i] + pos_ca_gain[i] / 2
                    ax.text(x_ca, i, f"n={nca}",
                            va='center', ha='center', fontsize=9,
                            color='white', fontweight='bold', clip_on=True)

                if nc > 0 and pos_within_clarify[i] > 0:
                    x_cr = pos_abstain_gain[i] + pos_ca_gain[i] + pos_within_clarify[i] / 2
                    ax.text(x_cr, i, f"{net}/{nc}\n({net/nc:.0%})",
                            va='center', ha='center', fontsize=9,
                            color='white', fontweight='bold', clip_on=True)

    axes_flat[n_models - 1].set_xlabel(f'Gap = BAG - baseline acc (standard {ref_type}-intent)')
    axes_flat[n_models - 1].legend(
        list(legend_handles.values()), list(legend_handles.keys()),
        loc='lower right', fontsize=11,
    )

    if settings is not None:
        _settings_label = "  |  ".join(f"{m}/{d}" for m, d in settings_list)
    else:
        _settings_label = f"direct_prompt={settings_list[0][1]}"
    plt.tight_layout()
    plt.show()

# ── Direct evaluation ─────────────────────────────────────────────────────────

def print_parse_errors(outputs, config):
    """Print parse error counts and mean response/reasoning lengths by model/prompt/step."""
    parse_errors = []
    lengths = []
    for assistant_model in config['assistant_models']:
        for direct_prompt in config['direct_prompts']:
            for belief_sampling in config['belief_samplings']:
                for step in ['direct', 'disambiguated']:
                    items = outputs[assistant_model][direct_prompt][belief_sampling].get(step, [])
                    for item in items:
                        for gen_type, gens in item['generations'].items():
                            for gen in (gens if isinstance(gens, list) else [gens]):
                                if isinstance(gen, dict):
                                    if gen.get('raw_response') is None:
                                        parse_errors.append({'model': assistant_model, 'prompt': direct_prompt,
                                                             'sampling': belief_sampling, 'step': step, 'gen_type': gen_type})
                                    else:
                                        reasoning = gen.get('reasoning')
                                        lengths.append({'model': assistant_model, 'prompt': direct_prompt,
                                                        'sampling': belief_sampling, 'step': step, 'gen_type': gen_type,
                                                        'response_words': len(gen['raw_response'].split()),
                                                        'reasoning_words': len(reasoning.split()) if reasoning else None})

    total = sum(
        len(outputs[m][p][s].get(step, []))
        for m in config['assistant_models']
        for p in config['direct_prompts']
        for s in config['belief_samplings']
        for step in ['direct', 'disambiguated']
    )
    print(f"Parse errors: {len(parse_errors)} / {total} ({100*len(parse_errors)/total:.1f}%)")
    if parse_errors:
        print(pd.DataFrame(parse_errors).groupby(['model', 'prompt', 'sampling', 'step']).size().rename('errors').to_string())
    print("\nResponse length (words):")
    print(pd.DataFrame(lengths).groupby(['model', 'prompt', 'sampling', 'step'])[['response_words', 'reasoning_words']].mean().round(1).to_string())


def compute_direct_metrics(outputs, config, do_print='none'):
    """Compute judge metrics for direct and disambiguated steps; returns metrics[model][prompt][sampling]."""
    import evaluation_utils
    metrics = collections.defaultdict(lambda: collections.defaultdict(dict))
    for assistant_model in config['assistant_models']:
        for direct_prompt in config['direct_prompts']:
            for belief_sampling in config['belief_samplings']:
                o = outputs[assistant_model][direct_prompt][belief_sampling]
                if not o.get("direct"):
                    continue
                disambiguated  = o.get("disambiguated")  or [None] * len(o["direct"])
                direct_judge   = o.get("direct_judge")   or [None] * len(o["direct"])
                disambig_judge = o.get("disambig_judge") or [None] * len(o["direct"])
                belief_judge   = o.get("belief_judge")   or [None] * len(o["direct"])
                pairs = [
                    (d, s, dj, dgj, bj)
                    for d, s, dj, dgj, bj in zip(o["direct"], disambiguated, direct_judge, disambig_judge, belief_judge)
                    if sample_is_valid(d)
                ]
                n_skipped = len(o["direct"]) - len(pairs)
                if n_skipped:
                    print(f"Skipped {n_skipped} parse errors for {assistant_model}/{direct_prompt}/{belief_sampling}")
                if pairs:
                    direct_valid, disambig_valid, dj_valid, dgj_valid, bj_valid = zip(*pairs)
                else:
                    direct_valid, disambig_valid, dj_valid, dgj_valid, bj_valid = [], [], [], [], []
                output = {
                    "direct":        list(direct_valid),
                    "disambiguated": list(disambig_valid),
                    "reasoner": [{"generations": {"strategy": None}}] * len(direct_valid),
                    "user":     [{"generations": None}]               * len(direct_valid),
                    "final":    [{"generations": None}]               * len(direct_valid),
                    "direct_judge":   list(dj_valid),
                    "disambig_judge": list(dgj_valid),
                    "belief_judge":   list(bj_valid),
                }
                metrics[assistant_model][direct_prompt][belief_sampling] = evaluation_utils.compute_metrics(
                    output, do_print=do_print, belief=True, skip_rouge=True)
    return metrics


def _valid_fp(reasoner_prompt, final_prompt):
    """SAG (prompt) pairs with prompt/prompt1; BAG (belief*) pairs with prompt/belief."""
    is_bag = reasoner_prompt != 'prompt'
    if final_prompt == 'belief'  and not is_bag: return False
    if final_prompt == 'prompt1' and     is_bag: return False
    return True


def _resolve_fp(reasoner_prompt, final_prompt, config):
    """Resolve '+' sentinel to the correct non-default final_prompt for this reasoner."""
    if final_prompt != '+':
        return final_prompt
    for fp in config.get('final_prompts', ['prompt']):
        if fp != 'prompt' and _valid_fp(reasoner_prompt, fp):
            return fp
    return 'prompt'


def _prettify_reasoner_prompt(name, short=False):
    """Map internal reasoner-prompt keys to display names: belief→BAG1, belief6→BAG2, belief7→BAG3, prompt→SAG.
    belief_avg→BAG (averaged over BAG variants).
    short=True: "SAG"→"S", "BAG1"→"B1", "BAG2"→"B2", "BAG3"→"B3", etc.

    Canonical source of the paper-facing name mapping. Also mirrored in
    visualiser/pipeline.html (REASONER_LABEL / DIRECT_LABEL) — update both
    together if the mapping changes.

    Note: belief1..belief5 and belief8 were exploratory reasoner prompts and
    are not reported in the paper; the paper uses only belief (BAG1),
    belief6 (BAG2), belief7 (BAG3), and prompt (SAG). Internal filenames and
    prompts.py retain the original names.
    """
    _BELIEF_TO_BAG = {'belief': 'BAG1', 'belief6': 'BAG2', 'belief7': 'BAG3', 'belief_avg': 'BAG'}
    suffix = '+' if name.endswith('+') else ''
    base = name[:-1] if suffix else name
    if base in _BELIEF_TO_BAG:
        name = _BELIEF_TO_BAG[base] + suffix
    else:
        name = re.sub(r'^belief', 'BAG', name)
    name = re.sub(r'^prompt', 'SAG', name)
    if short:
        name = re.sub(r'^SAG$', 'S', name)
        name = re.sub(r'^BAG(\d*)(\+?)$', lambda m: 'B' + m.group(1) + m.group(2), name)
    return name


def _prettify_direct_prompt(name):
    """Map internal direct-prompt keys to display names: vanilla → free."""
    return 'free' if name == 'vanilla' else name


def _prettify_model(name, latex=False, short=False):
    """Strip org prefix and -instruct suffix; capitalize and fix size suffix.
    short=True: first 3 alpha chars of family + last dash-segment, e.g. "Gem-f", "OLM-13B".
    """
    name = name.split('/')[-1]
    name = re.sub(r'[-_]instruct$', '', name, flags=re.IGNORECASE)
    name = (name[0].upper() + name[1:]) if name else name
    name = re.sub(r'(?i)olmo', 'OLMo', name)                              # preserve official casing
    name = re.sub(r'(\d+)b\b', lambda m: m.group(1) + 'B', name, flags=re.IGNORECASE)
    name = re.sub(r'(?i)\bflash\b', 'Flash', name)                        # proper case; abbreviated below for short
    if short:
        parts  = name.split('-')
        prefix = re.sub(r'\d+', '', parts[0])[:3]
        suffix = parts[-1] if len(parts) > 1 else ''
        suffix = suffix.replace('Flash', 'f')
        name   = prefix + ('-' + suffix if suffix else '')
    if latex:
        name = name.replace('_', r'\_')
    return name


def _build_direct_df(metrics, config, metric_type, direct_prompt, sampling):
    """Build the direct-eval DataFrame (2-tuple col keys, words scaled); shared by display and latex."""
    j = metric_type == "judge"
    col_specs = [
        ("All", "Standard (1 intent)",   "direct_judge"                        if j else "direct_answer_randomref"),
        ("All", "Disambig (1 intent)",   "full_disambig_upperbound_judge"       if j else "full_disambig_upperbound"),
        # ("All", "Belief (1 intent)",     "belief_judge"                         if j else "direct_belief_randomref"),
        ("All", "Standard (any intent)", "direct_judge_anyref"                  if j else "direct_answer_anyref"),
        ("All", "Disambig (any intent)", "full_disambig_upperbound_judge_anyref" if j else "full_disambig_upperbound_anyref"),
        # ("All", "Belief (any intent)",   "belief_judge_anyref"                  if j else "direct_belief_anyref"),
        ("All", "#words",                "direct_answer_length"),
    ]
    prompts = config['direct_prompts'] if direct_prompt is None else [direct_prompt]
    df = build_table(prompts, col_specs, metrics, config['assistant_models'], sampling, drop_prompt=direct_prompt is not None)
    df[("All", "#words")] /= 100
    return df


def _build_direct_counts(metrics, config, direct_prompt, sampling):
    """Build count-of-non-None DataFrame matching _build_direct_df columns."""
    col_keys = [
        "Standard (1 intent)", "Disambig (1 intent)", # "Belief (1 intent)",
        "Standard (any intent)", "Disambig (any intent)", # "Belief (any intent)", "#words",
    ]
    metric_keys = [
        "direct_judge", "full_disambig_upperbound_judge", #"belief_judge",
        "direct_judge_anyref", "full_disambig_upperbound_judge_anyref", #"belief_judge_anyref",
        "direct_answer_length",
    ]
    prompts = config['direct_prompts'] if direct_prompt is None else [direct_prompt]
    table = collections.defaultdict(dict)
    for model in config['assistant_models']:
        for prompt in prompts:
            try:
                m = metrics[model][prompt][sampling]
            except KeyError:
                continue
            row_key = model if direct_prompt is not None else (model, prompt)
            for col, key in zip(col_keys, metric_keys):
                table[row_key][col] = sum(1 for v in m.get(key, []) if v is not None)
    df = pd.DataFrame(table).T
    if direct_prompt is None:
        df.index = pd.MultiIndex.from_tuples(
            [(_prettify_model(m), _prettify_direct_prompt(p)) for m, p in df.index]
        )
    else:
        df.index = pd.Index([_prettify_model(m) for m in df.index])
    return df


def display_direct_results(metrics, config, sampling='unbiased', metric_type='judge', direct_prompt='concise', gradient_by='model'):
    """Display results table (all questions only).

    direct_prompt=None  → (model, prompt) MultiIndex rows, all prompts side-by-side.
    direct_prompt=<str> → model-only rows for that single prompt.
    Columns: intent (1 intent / any intent) × method (Standard / Disambig / Belief) + #words.
    """
    from IPython.display import display
    df = _build_direct_df(metrics, config, metric_type, direct_prompt, sampling)

    _col2 = {
        ("All", "Standard (1 intent)"):   ("1 intent",   "Standard"),
        ("All", "Disambig (1 intent)"):   ("1 intent",   "Disambig"),
        # ("All", "Belief (1 intent)"):     ("1 intent",   "Belief"),
        ("All", "Standard (any intent)"): ("any intent", "Standard"),
        ("All", "Disambig (any intent)"): ("any intent", "Disambig"),
        # ("All", "Belief (any intent)"):   ("any intent", "Belief"),
        ("All", "#words"):                ("",           "#words"),
    }
    df.columns = pd.MultiIndex.from_tuples([_col2[c] for c in df.columns])

    if direct_prompt is not None:
        df.index = pd.Index([_prettify_model(m) for m in df.index])
    else:
        df.index = pd.MultiIndex.from_tuples(
            [(_prettify_model(m), _prettify_direct_prompt(p)) for m, p in df.index], names=["model", "prompt"]
        )

    words_cols = [("", "#words")]
    caption = f"Seed {config['seed']} | split={config.get('split', '?')} | {metric_type}" + (f" | {direct_prompt}" if direct_prompt else "")
    display(style_direct_table(df, caption=caption, no_gradient_cols=words_cols, gradient_by=gradient_by))

    df_counts = _build_direct_counts(metrics, config, direct_prompt, sampling)
    df_counts.columns = pd.MultiIndex.from_tuples([("n non-None", c) for c in df_counts.columns])
    display(df_counts.style
        .set_caption("Non-None count per cell")
        .set_table_styles([
            {"selector": "th.col_heading", "props": [("text-align", "center"), ("white-space", "normal"),
                                                      ("word-break", "break-word"), ("max-width", "60px")]},
        ])
    )


def style_direct_table(df, caption="", no_gradient_cols=(), gradient_by="model"):
    """Style a grouped metrics DataFrame with group-divider borders and colour gradient.

    gradient_by:
      "model"  (default) — each model group (all its prompt rows) shares one scale spanning
               across columns; reveals col-to-col trends within a model.
      "global" — single scale across all cells in the table (original behaviour).
    """
    cols = list(df.columns)
    nlevels = df.columns.nlevels
    gradient_cols = [c for c in cols if c not in no_gradient_cols]
    seen_top, seen_sub, dividers = set(), set(), {}
    for col in cols:
        if col[0] not in seen_top:
            seen_top.add(col[0])
            if nlevels == 3 and col[1]:
                seen_sub.add((col[0], col[1]))
            dividers[col] = [{"selector": "", "props": [("border-left", "3px solid #888")]}]
        elif nlevels == 3 and col[1] and (col[0], col[1]) not in seen_sub:
            seen_sub.add((col[0], col[1]))
            dividers[col] = [{"selector": "", "props": [("border-left", "1.5px solid #bbb")]}]
    styler = (df.style
        .format("{:.1f}")
        .set_caption(caption)
        .set_table_styles(dividers, overwrite=False, axis=0)
        .set_table_styles([{"selector": "th.col_heading", "props": [("text-align", "center")]}], overwrite=False)
    )
    if gradient_by == "global":
        vmin, vmax = df[gradient_cols].min().min(), df[gradient_cols].max().max()
        styler = styler.background_gradient(cmap="RdYlGn", subset=gradient_cols, vmin=vmin, vmax=vmax)
    elif gradient_by == "row":
        styler = styler.background_gradient(cmap="RdYlGn", subset=gradient_cols, axis=1)
    elif isinstance(df.index, pd.MultiIndex):
        for model in df.index.get_level_values(0).unique():
            model_data = df.loc[model, gradient_cols]
            vmin, vmax = model_data.min().min(), model_data.max().max()
            styler = styler.background_gradient(
                cmap="RdYlGn",
                subset=(pd.IndexSlice[model, :], gradient_cols),
                vmin=vmin, vmax=vmax,
            )
    else:
        styler = styler.background_gradient(cmap="RdYlGn", subset=gradient_cols, axis=1)
    return styler


def build_table(prompts, col_specs, metrics, models, sampling, drop_prompt=False):
    """Build a metrics DataFrame indexed by (model, prompt) with colour-coded group columns."""
    table = collections.defaultdict(dict)
    for model in models:
        for prompt in prompts:
            try:
                m = metrics[model][prompt][sampling]
            except KeyError:
                continue
            masks = {
                "Ambig":     m["is_ambig"],
                "Non-ambig": [not v for v in m["is_ambig"]],
                "All":       [True] * len(m["is_ambig"]),
            }
            for group, label, key in col_specs:
                table[(model, prompt)][(group, label)] = masked_mean(m[key], masks[group])
    df = pd.DataFrame(table).T * 100
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.names = ["model", "prompt"]
    if drop_prompt:
        df.index = df.index.droplevel("prompt")
    return df


# ── Decoding method comparison ────────────────────────────────────────────────

def display_decoding_comparison(metrics, config, ref_type="1", subset="all", metric_type="judge"):
    """Compare answer quality across decoding methods for the direct step.

    Columns: answer | belief_state_sample | entire_belief_state

    - answer              model-recommended sampling (direct_judge)
    - belief_state_sample expected quality of a single unbiased draw (mean verdict per sample)
    - entire_belief_state best-of-K across all belief samples (belief_judge)

    Args:
        ref_type: "1" uses a single random reference; "any" uses best-of-N references.
        subset:   "all", "ambig", or "non-ambig".
        metric_type: "judge" (default).
    """
    from IPython.display import display

    j    = metric_type == "judge"
    _any = ref_type == "any"

    col_specs = [
        ("answer",              "direct_judge_anyref"        if _any else "direct_judge"),
        ("belief_state_sample", "belief_judge_sample_anyref" if _any else "belief_judge_sample"),
        ("entire_belief_state", "belief_judge_anyref"        if _any else "belief_judge"),
    ]

    subset_label = {"all": "All", "ambig": "Ambig", "non-ambig": "Non-ambig"}.get(subset, "All")
    sampling = config['belief_samplings'][0]

    table = collections.defaultdict(dict)

    for model in config['assistant_models']:
        for prompt in config['direct_prompts']:
            try:
                m = metrics[model][prompt][sampling]
            except KeyError:
                continue

            is_ambig = m.get("is_ambig", [])
            mask = {
                "all":       [True] * len(is_ambig),
                "ambig":     is_ambig,
                "non-ambig": [not v for v in is_ambig],
            }.get(subset, [True] * len(is_ambig))

            for col_label, key in col_specs:
                table[(model, prompt)][col_label] = masked_mean(m.get(key, []), mask)

    df = pd.DataFrame(table).T * 100
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "prompt"])

    vmin, vmax = df.min().min(), df.max().max()
    display(df.style
        .background_gradient(cmap="RdYlGn", vmin=vmin, vmax=vmax)
        .format("{:.1f}")
        .set_caption(
            f"Decoding comparison | {metric_type} | split={config.get('split', '?')} | ref={ref_type} | subset={subset_label} | seed={config['seed']}"
        )
        .set_table_styles(
            [{"selector": "th.col_heading", "props": [("text-align", "center")]}],
        )
    )


# ── Trace viewer ──────────────────────────────────────────────────────────────

def _build_trace(i, d_item, r_item, user_items, final_items, m):
    r_gens      = r_item.get('generations') or {}
    r_ctx       = r_item.get('context', {})
    ctx         = d_item.get('context', {})
    u_item      = user_items[i]  if user_items  and i < len(user_items)  else None
    fn_item     = final_items[i] if final_items and i < len(final_items) else None
    belief_state = r_ctx.get('belief_state') or r_ctx.get('beliefs') or r_ctx.get('belief_samples') or ''
    u_gens      = (u_item or {}).get('generations') or {}
    def _safe(key): return m.get(key, [None])[i] if i < len(m.get(key, [None])) else None
    return dict(
        idx=i,
        strategy=r_gens.get('strategy') or 'None',
        n_disambigs=m['num_disambigs'][i] if i < len(m.get('num_disambigs', [])) else 0,
        question=ctx.get('question', ''),
        references=ctx.get('references', []),
        disambigs=ctx.get('disambiguations', []),
        belief_state=belief_state,
        reasoning=r_gens.get('reasoning') or '',
        cq_or_ans=r_gens.get('response') or '',
        user_response=u_gens.get('response') or '',
        direct_text=_extract_direct_gen(d_item),
        final_text=_extract_final_gen(fn_item) if fn_item is not None else '',
        direct_j=_safe('direct_judge'),
        direct_any_j=_safe('direct_judge_anyref'),
        belief_j=_safe('belief_judge'),
        final_j=_safe('final_judge'),
        final_any_j=_safe('final_judge_anyref'),
        disambig_j=_safe('disambig_judge'),
        reason_direct_j=_safe('reason_direct_judge'),
    )


def _matches_condition(cond, i, m, ref_type='1'):
    _any  = ref_type == 'any'
    dj    = m['direct_judge_anyref' if _any else 'direct_judge'][i]     if i < len(m.get('direct_judge_anyref' if _any else 'direct_judge', [])) else None
    fj    = m['final_judge_anyref' if _any else 'final_judge'][i]        if i < len(m.get('final_judge_anyref' if _any else 'final_judge', [])) else None
    dj_d  = m['disambig_judge_anyref' if _any else 'disambig_judge'][i] if i < len(m.get('disambig_judge_anyref' if _any else 'disambig_judge', [])) else None
    rdj   = m['reason_direct_judge_anyref' if _any else 'reason_direct_judge'][i] if i < len(m.get('reason_direct_judge_anyref' if _any else 'reason_direct_judge', [])) else None
    strat = (m.get('reasoner_strategy') or [])[i] if i < len(m.get('reasoner_strategy', [])) else None
    is_cq = strat == 'clarification_question'
    is_rd = strat == 'direct_answer'
    if cond in ('recovery',    'clarify_pos'):      return is_cq and dj == 0 and fj == 1
    if cond in ('regression',  'clarify_neg'):      return is_cq and dj == 1 and fj == 0
    if cond == 'disambig_pos':                      return dj_d is not None and dj == 0 and dj_d == 1
    if cond == 'disambig_neg':                      return dj_d is not None and dj == 1 and dj_d == 0
    if cond == 'reasoner_direct_pos':               return is_rd and rdj is not None and dj == 0 and rdj == 1
    if cond == 'reasoner_direct_neg':               return is_rd and rdj is not None and dj == 1 and rdj == 0
    return False


def _base_filter(i, r_item, m, filter_strategy, filter_ambiguous_only):
    r_gens      = r_item.get('generations') or {}
    strategy    = r_gens.get('strategy') or 'None'
    n_disambigs = m['num_disambigs'][i] if i < len(m.get('num_disambigs', [])) else 0
    if filter_strategy != 'all' and strategy != filter_strategy:
        return False
    if filter_ambiguous_only and n_disambigs == 0:
        return False
    return True


def run_trace_viewer(outputs, metrics, trace_config, named_filters=None, width=90):
    """Print pipeline traces for selected examples.

    trace_config keys:
        model, reasoner_prompt, direct_prompt, belief_sampling, final_prompt,
        n_traces, filter_strategy, filter_ambiguous_only
    named_filters: dict where each value is "recovery", "regression", or a list of int ids.
    """
    named_filters = named_filters or {}
    model         = trace_config['model']
    rp            = trace_config['reasoner_prompt']
    dp            = trace_config['direct_prompt']
    bs            = trace_config['belief_sampling']
    fp            = trace_config.get('final_prompt', 'prompt')
    n_traces      = trace_config.get('n_traces', 10)
    filter_strategy      = trace_config.get('filter_strategy', 'all')
    filter_ambiguous_only = trace_config.get('filter_ambiguous_only', False)
    show_belief          = trace_config.get('show_belief', True)
    ref_type             = trace_config.get('ref_type', '1')

    o = outputs[model][dp][bs]
    m = metrics[model][dp][bs][rp][fp]

    direct_items   = o['direct']
    reasoner_items = o['reasoner'][rp]
    user_items     = o.get('user', {}).get(rp) if isinstance(o.get('user'), dict) else None
    final_items    = (o.get('final') or {}).get(rp, {}).get(fp)

    W = width
    print(f"Setting : {model} | {rp} | {dp} | {bs}")

    _active  = {k: v for k, v in named_filters.items() if v}
    use_named = bool(_active)

    if use_named:
        _list_ids  = {i for v in _active.values() if isinstance(v, list) for i in v}
        _cond_keys = {k for k, v in _active.items() if isinstance(v, str)}

        trace_by_idx = {}
        cond_traces  = {k: [] for k in _cond_keys}

        for i, (d_item, r_item) in enumerate(zip(direct_items, reasoner_items)):
            need_trace = i in _list_ids
            for k in _cond_keys:
                if _matches_condition(_active[k], i, m, ref_type=ref_type):
                    if i not in trace_by_idx:
                        trace_by_idx[i] = _build_trace(i, d_item, r_item, user_items, final_items, m)
                    cond_traces[k].append(trace_by_idx[i])
            if need_trace and i not in trace_by_idx:
                trace_by_idx[i] = _build_trace(i, d_item, r_item, user_items, final_items, m)

        cases_named = {}
        for label, val in _active.items():
            if isinstance(val, str):
                cases_named[label] = cond_traces[label]
            else:
                cases_named[label] = [trace_by_idx[i] for i in val if i in trace_by_idx]

        cases_named = {k: v[:n_traces] for k, v in cases_named.items()}
        total = sum(len(v) for v in cases_named.values())
        print(f"Filters : {', '.join(f'{k}={len(v)}' for k, v in cases_named.items())}")
        print(f"Showing : {total} trace(s)\n")
        for label, traces in cases_named.items():
            if not traces:
                continue
            print(f"\n{'#' * W}")
            print(f"## {label.upper()}  ({len(traces)} example(s))")
            print(f"{'#' * W}\n")
            for c in traces:
                print_trace(c, label=label, width=W, show_belief=show_belief)
    else:
        cases = []
        for i, (d_item, r_item) in enumerate(zip(direct_items, reasoner_items)):
            if not _base_filter(i, r_item, m, filter_strategy, filter_ambiguous_only):
                continue
            cases.append(_build_trace(i, d_item, r_item, user_items, final_items, m))
            if len(cases) >= n_traces:
                break

        print(f"Filters : strategy={filter_strategy}  ambiguous_only={filter_ambiguous_only}")
        print(f"Showing : {len(cases)} trace(s)\n")
        for c in cases:
            print_trace(c, width=W)


def print_trace(c, label=None, width=90, show_belief=True):
    """Print a single pipeline trace dict produced by _build_trace."""
    print('=' * width)
    if label:
        print(f">>> {label.upper()} <<<")
    j_str = (f"direct={c['direct_j']}  direct_any={c['direct_any_j']}  "
             f"belief={c['belief_j']}  final={c['final_j']}  final_any={c.get('final_any_j')}  "
             f"disambig={c.get('disambig_j')}  reason_direct={c.get('reason_direct_j')}")
    print(f"[idx={c['idx']}]  strategy={c['strategy']}  n_disambigs={c['n_disambigs']}")
    print(f"JUDGES : {j_str}")
    print()
    print(f"QUESTION:\n  {c['question']}")
    print()
    if c['disambigs']:
        print("DISAMBIGUATIONS:")
        for d in c['disambigs']:
            print(f"  - {d}")
        print()
    print("REFERENCES:")
    for r in c['references']:
        print(f"  - {r}")
    print()
    if show_belief and c['belief_state']:
        bs = c['belief_state']
        if isinstance(bs, list):
            print(f"BELIEF STATE ({len(bs)} samples):")
            for j, gen in enumerate(bs, 1):
                print(f"  [{j}] {gen}")
        else:
            print(f"BELIEF STATE:\n  {bs}")
        print()
    if c['reasoning']:
        print(f"REASONER REASONING:\n{c['reasoning']}")
        print()
    lbl = 'CLARIFICATION QUESTION' if c['strategy'] == 'clarification_question' else 'REASONER RESPONSE'
    print(f"{lbl}:\n{c['cq_or_ans']}")
    print()
    if c['user_response']:
        print(f"USER RESPONSE:\n{c['user_response']}")
        print()
    if c.get('direct_text'):
        print(f"DIRECT ANSWER:\n{c['direct_text']}")
        print()
    if c['final_text']:
        print(f"FINAL ANSWER:\n{c['final_text']}")
        print()


def print_disambig_wrong_belief_right(outputs, metrics, config, model=None, direct_prompt="vanilla", belief_sampling="unbiased", n=10, width=80):
    """Print cases where disambig oracle is wrong (1-intent) but belief state covers the reference."""
    model = model or config['assistant_models'][0]
    o = outputs[model][direct_prompt][belief_sampling]

    direct_items   = o['direct']
    disambig_items = o.get('disambiguated') or [None] * len(direct_items)
    direct_judge   = o.get('direct_judge')  or [None] * len(direct_items)
    disambig_judge = o.get('disambig_judge') or [None] * len(direct_items)
    belief_judge   = o.get('belief_judge')   or [None] * len(direct_items)

    shown = 0
    for i, (d, dg, dj, dgj, bj) in enumerate(zip(direct_items, disambig_items, direct_judge, disambig_judge, belief_judge)):
        if dgj is None or bj is None:
            continue
        d_v = ((dgj.get('generations') or {}).get('disambig') or {}).get('verdict')
        b_v = ((bj.get('generations')  or {}).get('belief')   or {}).get('verdict')
        if d_v != 0 or b_v != 1:
            continue

        ctx    = d.get('context', {})
        dg_ctx = (dg or {}).get('context') or {}

        std_ans_raw = ((d.get('generations') or {}).get('answer') or [None])[0]
        std_text    = (std_ans_raw.get('raw_response') if isinstance(std_ans_raw, dict) else std_ans_raw) or '(none)'

        dg_ans_raw = ((dg or {}).get('generations', {}).get('answer') or [None])[0]
        dg_text    = (dg_ans_raw.get('raw_response') if isinstance(dg_ans_raw, dict) else dg_ans_raw) or '(none)'

        belief_gens = d['generations'].get('belief_state', [])
        b_samples   = ((bj.get('generations') or {}).get('belief') or {}).get('samples', [])
        b_ref       = (bj.get('context') or {}).get('belief_ref')

        std_j  = (dj.get('generations') or {}).get('direct') or {} if dj else {}
        std_v  = std_j.get('verdict')
        std_mark = {1: '✓', 0: '✗'}.get(std_v, '?')

        print('=' * width)
        print(f"[{i}]  Q : {ctx.get('question')}")
        print(f"     REF: {b_ref}    (all: {ctx.get('references')})")
        print()
        print(f"  STANDARD ANS: {std_text[:200]}")
        print(f"  JUDGE [{std_mark}]  : {std_j.get('reasoning')}")
        print()
        print(f"  DISAMBIG Q  : {dg_ctx.get('disambiguated_question')}")
        print(f"  DISAMBIG ANS: {dg_text[:200]}")
        dgj_entry = (dgj.get('generations') or {}).get('disambig') or {}
        print(f"  JUDGE  [✗]  : {dgj_entry.get('reasoning')}")
        print()
        print(f"  BELIEF SAMPLES:")
        for j, (gen, s) in enumerate(zip(belief_gens, b_samples), 1):
            raw  = (gen.get('raw_response') if isinstance(gen, dict) else gen) or ''
            mark = '✓' if s.get('verdict') == 1 else '✗'
            print(f"    [{j}] {mark}  {raw[:120]}")
        print()

        shown += 1
        if shown >= n:
            break

    print(f"— {shown} / {n} cases shown (model={model}, prompt={direct_prompt}) —")
