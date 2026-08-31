import re

import matplotlib.pyplot as plt
import pandas as pd

from nb_utils import (
    _build_direct_df,
    _build_pipeline_df,
    _prettify_direct_prompt,
    _prettify_model,
    _prettify_reasoner_prompt,
    _resolve_fp,
    _routing_profile_rows,
)


def pipeline_results_to_latex(metrics, config, metric_type='judge', ref_type='1', gradient_by='global', two_col=True, show_abstain=False, direct_prompt=None, final_variant='all', include_words_col=False):
    """Print (and return) a LaTeX table matching build_and_display_metrics_tables.

    Three-level header:
      Direct Generation (1 intent | any intent) | Strategy-Aug. Generation (1 intent)

    gradient_by:
      "global" (default) — single colour scale across all cells.
      "model"            — each model group gets its own scale.
      "row"              — each (model, brevity) row gets its own scale.
    direct_prompt:
      None (default) — all brevity rows shown (2 label cols: model + brevity).
      str            — only rows matching that brevity; 1 label col (model only).
    two_col:
      True  (default) — table* spanning both columns.
      False           — single-column table with footnotesize and tighter tabcolsep.
    final_variant:
      "all" — all pipeline columns; "+" — only + columns; "prompt" — only non-+ columns.
    show_abstain:
      True — each cell shows the score on top and the abstain count (relative to the
             first pipeline column for that row) as a tiny subscript, using \\shortstack.

    Requires in the LaTeX preamble:
        \\usepackage{booktabs}
        \\usepackage{multirow}
        \\usepackage[table]{xcolor}
    """
    df, df_counts = _build_pipeline_df(metrics, config, metric_type, ref_type)

    if direct_prompt is not None:
        pretty = _prettify_direct_prompt(direct_prompt)
        df        = df.loc[df.index.get_level_values(1) == pretty]
        df_counts = df_counts.loc[df_counts.index.get_level_values(1) == pretty]

    _DIRECT_COLS = {"Standard", "Disambig", "Belief", "Standard (any)", "Disambig (any)"}
    if include_words_col:
        _DIRECT_COLS = _DIRECT_COLS | {"#words"}
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

    j = metric_type == "judge"
    key_default = "direct_judge" if j else "direct_answer_randomref"
    m0 = metrics[config['assistant_models'][0]][config['direct_prompts'][0]][config['belief_samplings'][0]][config['reasoner_prompts'][0]][config.get('final_prompts', ['prompt'])[0]]
    n_examples = sum(1 for v in m0[key_default] if v is not None)

    col_names     = list(df.columns)
    d1_cols       = [c for c in col_names if c in {"Standard", "Disambig", "Belief", "#words"}]
    dany_cols     = [c for c in col_names if c in {"Standard (any)", "Disambig (any)"}]
    pipeline_cols = [c for c in col_names if c not in _DIRECT_COLS]
    grad_cols     = [c for c in col_names if c != '#words']
    n_d1, n_da, n_p = len(d1_cols), len(dany_cols), len(pipeline_cols)
    n_direct = n_d1 + n_da

    n_label = 1 if (direct_prompt is not None or df.index.get_level_values(1).nunique() == 1) else 2

    col_spec = 'l' * n_label + '|' + 'c' * n_d1
    if n_da:
        col_spec += '|' + 'c' * n_da
    col_spec += '|' + 'c' * n_p

    off = n_label + 1
    d1_range   = (off,          off + n_d1 - 1)
    da_range   = (off + n_d1,   off + n_d1 + n_da - 1)
    p_range    = (off + n_direct, off + n_direct + n_p - 1)
    dir_range  = (off,           off + n_direct - 1)

    model_span = f'\\multicolumn{{{n_label}}}{{{"l|"}}}{{}}'

    # Row 1: Direct Generation | Strat.-Aug. Generation
    row1_parts = [model_span]
    if n_direct:
        fmt = 'c|' if n_p else 'c'
        row1_parts.append(f'\\multicolumn{{{n_direct}}}{{{fmt}}}{{\\textbf{{Direct Generation}}}}')
    if n_p:
        row1_parts.append(f'\\multicolumn{{{n_p}}}{{c}}{{\\textbf{{Strat.-Aug. Generation}}}}')
    row1 = ' & '.join(row1_parts) + r' \\'

    cmidrule1 = []
    if n_direct:
        cmidrule1.append(f'\\cmidrule(lr){{{dir_range[0]}-{dir_range[1]}}}')
    if n_p:
        cmidrule1.append(f'\\cmidrule(lr){{{p_range[0]}-{p_range[1]}}}')

    # Row 2: 1 intent | any intent | 1 intent
    row2_parts = [''] * n_label
    if n_d1:
        fmt = 'c|' if n_da or n_p else 'c'
        row2_parts.append(f'\\multicolumn{{{n_d1}}}{{{fmt}}}{{1\\,intent}}')
    if n_da:
        fmt = 'c|' if n_p else 'c'
        row2_parts.append(f'\\multicolumn{{{n_da}}}{{{fmt}}}{{any\\,intent}}')
    if n_p:
        row2_parts.append(f'\\multicolumn{{{n_p}}}{{c}}{{1\\,intent}}')
    row2 = ' & '.join(row2_parts) + r' \\'

    cmidrule2 = []
    if n_d1:
        cmidrule2.append(f'\\cmidrule(lr){{{d1_range[0]}-{d1_range[1]}}}')
    if n_da:
        cmidrule2.append(f'\\cmidrule(lr){{{da_range[0]}-{da_range[1]}}}')
    if n_p:
        cmidrule2.append(f'\\cmidrule(lr){{{p_range[0]}-{p_range[1]}}}')

    def _leaf(c):
        if c in ("Standard", "Standard (any)"): return "Standard"
        if c in ("Disambig",  "Disambig (any)"): return "Disambig"
        if c == "Belief": return "Belief"
        return c.replace(f' {ref_type}', '').strip()

    row3 = ' & '.join([''] * n_label + [_leaf(c) for c in col_names]) + r' \\'

    cmap = plt.cm.RdYlGn
    total_cols = n_label + len(col_names)

    if gradient_by == "global":
        _global_vmin = df[grad_cols].min().min()
        _global_vmax = df[grad_cols].max().max()

    first_pipeline = pipeline_cols[0] if pipeline_cols else col_names[0]

    def _cell(value, vmin, vmax, n_abstain=None, no_color=False):
        if pd.isna(value):
            return '--'
        if no_color:
            return f'{value:.1f}'
        val_str = f'{value:.1f}'
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        r, g, b, _ = cmap(max(0.0, min(1.0, t)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text = 'white' if lum < 0.4 else 'black'
        bg = f'\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}'
        if n_abstain is not None:
            content = f'\\shortstack[c]{{{val_str}\\\\[-2pt]{{\\tiny {n_abstain}}}}}'
            return f'{bg}\\textcolor{{{text}}}{{{content}}}'
        return f'{bg}\\textcolor{{{text}}}{{{val_str}}}'

    models_order = list(dict.fromkeys(idx[0] for idx in df.index))
    data_rows = []
    for m_idx, model in enumerate(models_order):
        model_df = df.loc[model]
        n = len(model_df)
        pretty = _prettify_model(model, latex=True)
        if gradient_by == "model":
            vmin, vmax = model_df[grad_cols].min().min(), model_df[grad_cols].max().max()
        elif gradient_by != "row":
            vmin, vmax = _global_vmin, _global_vmax
        for row_idx, (prompt, row) in enumerate(model_df.iterrows()):
            if gradient_by == "row":
                grad_vals = [row[c] for c in grad_cols if not pd.isna(row[c])]
                vmin, vmax = (min(grad_vals), max(grad_vals)) if grad_vals else (0, 1)
            if show_abstain:
                n_baseline = df_counts.loc[(model, prompt), first_pipeline]
                abstains = {c: max(0, int(n_baseline - df_counts.loc[(model, prompt), c])) for c in col_names}
            else:
                abstains = {c: None for c in col_names}
            vals = [_cell(row[c], vmin, vmax, abstains[c], no_color=(c == '#words')) for c in col_names]
            if n_label == 2:
                label = f'\\multirow{{{n}}}{{*}}{{{pretty}}} & {prompt}' if row_idx == 0 else f' & {prompt}'
            else:
                label = pretty
            data_rows.append(label + ' & ' + ' & '.join(vals) + r' \\')
            if row_idx < n - 1:
                data_rows.append(f'\\cline{{2-{total_cols}}}')
        if m_idx < len(models_order) - 1:
            data_rows.append(r'\arrayrulecolor{black!35}\specialrule{1pt}{2pt}{2pt}\arrayrulecolor{black}')

    seed  = config.get('seed', '?')
    split = config.get('split', '?')
    norm_desc  = 'per-model' if gradient_by == 'model' else 'per-row' if gradient_by == 'row' else 'globally'
    abstain_note = ' Small numbers show abstain count relative to the first SAG column.' if show_abstain else ''
    caption = (
        f'QA performance on {n_examples} AmbigQA-{split} questions. '
        f'Seed~{seed}, ref~{ref_type}, brevity~{direct_prompt}. '
        f'Colors {norm_desc} normalized.{abstain_note}'
    )
    env      = 'table*' if two_col else 'table'
    size_cmd = r'\small' if two_col else r'\footnotesize'
    tabcolsep = '' if two_col else r'\setlength{\tabcolsep}{4pt}' + '\n'
    lines = [
        f'\\begin{{{env}}}[ht]',
        f'\\centering{size_cmd}',
        tabcolsep + f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        row1,
        ' '.join(cmidrule1),
        row2,
        ' '.join(cmidrule2),
        row3,
        r'\midrule',
        *data_rows,
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:bag_results}',
        f'\\end{{{env}}}',
    ]
    latex = '\n'.join(lines)
    print(latex)
    return latex


def routing_profile_to_latex(metrics, config, metric_type='judge', ref_type='1', gradient_by='global', block1_cols=('routing', 'quality', 'ambig'), direct_prompt=None, settings=None, avg_bag_variants=False):
    """Print (and return) a LaTeX table matching Block 1 from plot_routing_profile.

    gradient_by:
      "global" (default) — colour scale across all rows within each column group.
      "model"            — each model group gets its own scale.
      "row"              — each row gets its own scale per column group.
    avg_bag_variants: if True, average BAG1/2/3 variants into a single BAG row per model

    Requires in the LaTeX preamble:
        \\usepackage{booktabs}
        \\usepackage{multirow}
        \\usepackage[table]{xcolor}
    """
    if direct_prompt is None:
        direct_prompt = config['direct_prompts'][0]
    if settings is not None:
        settings_list = settings
    else:
        settings_list = [(mdl, direct_prompt) for mdl in config['assistant_models']]

    rows = _routing_profile_rows(metrics, config, metric_type, ref_type, settings_list, block1_cols, settings_passed=settings is not None, avg_bag_variants=avg_bag_variants)

    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(
        [(_prettify_model(m), _prettify_reasoner_prompt(rp)) for m, rp in df.index],
        names=["model", "reasoner"],
    )

    col_groups = []
    if 'routing' in block1_cols:
        col_groups.append((['clarify %', 'direct %', 'abstain %'], plt.cm.Blues))
    if 'quality' in block1_cols:
        col_groups.append((['acc(C)', 'acc(D)', 'acc(A)'], plt.cm.RdYlGn))
    if 'ambig' in block1_cols:
        col_groups.append((['amb(C)', 'amb(D)', 'amb(A)'], plt.cm.Blues))

    col_names = [c for cols, _ in col_groups for c in cols]

    group_global, group_model = {}, {}
    if gradient_by == "global":
        for cols, _ in col_groups:
            sub = df[cols]
            group_global[tuple(cols)] = (sub.min().min(), sub.max().max())
    elif gradient_by == "model":
        for model in df.index.get_level_values(0).unique():
            group_model[model] = {}
            for cols, _ in col_groups:
                sub = df.loc[model][cols]
                group_model[model][tuple(cols)] = (sub.min().min(), sub.max().max())

    def _cell(value, cmap, vmin, vmax):
        if pd.isna(value):
            return '--'
        val_str = f'{value:.1f}'
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        r, g, b, _ = cmap(max(0.0, min(1.0, t)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text = 'white' if lum < 0.4 else 'black'
        bg = f'\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}'
        return f'{bg}\\textcolor{{{text}}}{{{val_str}}}'

    col_spec   = 'll|' + 'c' * len(col_names)
    header     = ' & '.join(['', ''] + col_names) + r' \\'
    total_cols = 2 + len(col_names)

    def _stack_model(name):
        # "Olmo2-13B" → ("Ol2","13B"), "Qwen3-8B" → ("Qw3","8B"), "Gemini-2.5-f" → ("Gem","2.5f")
        m = re.match(r'([A-Za-z]+)(\d+)?(.*)', name)
        if not m:
            return name, ''
        family, ver, rest = m.groups()
        ver  = ver or ''
        rest = rest.lstrip('-').replace('-', '')   # "13B", "2.5f", "8B"
        line1 = family[:(2 if ver else 3)] + ver
        return line1, rest

    models_order = list(dict.fromkeys(idx[0] for idx in df.index))
    data_rows = []
    for m_idx, model in enumerate(models_order):
        model_df = df.loc[model]
        n        = len(model_df)
        l1, l2   = _stack_model(_prettify_model(model, latex=True))
        stacked  = f'\\shortstack[l]{{{l1}\\\\{l2}}}' if l2 else l1
        for row_idx, (prompt, row) in enumerate(model_df.iterrows()):
            short_prompt = _prettify_reasoner_prompt(prompt, short=True)
            cells = []
            for cols, cmap in col_groups:
                if gradient_by == "global":
                    vmin, vmax = group_global[tuple(cols)]
                elif gradient_by == "model":
                    vmin, vmax = group_model[model][tuple(cols)]
                else:
                    row_vals = [row[c] for c in cols if not pd.isna(row[c])]
                    vmin = min(row_vals) if row_vals else 0.0
                    vmax = max(row_vals) if row_vals else 1.0
                for c in cols:
                    cells.append(_cell(row[c], cmap, vmin, vmax))
            if row_idx == 0:
                label = f'\\multirow{{{n}}}{{*}}{{{stacked}}} & {short_prompt}'
            else:
                label = f' & {short_prompt}'
            data_rows.append(label + ' & ' + ' & '.join(cells) + r' \\')
            if row_idx < n - 1:
                data_rows.append(f'\\cline{{2-{total_cols}}}')
        if m_idx < len(models_order) - 1:
            data_rows.append(r'\arrayrulecolor{black!35}\specialrule{1pt}{2pt}{2pt}\arrayrulecolor{black}')

    seed      = config.get('seed', '?')
    split     = config.get('split', 'train')
    norm_desc = 'per-model' if gradient_by == 'model' else ('per-row' if gradient_by == 'row' else 'globally')
    caption   = (
        r'Action routing: accuracy of \textsc{Standard (1 intent)} on the subset of questions '
        r'routed to each action by the uncertainty reasoner '
        r'(e.g., acc(C) is the baseline accuracy on questions routed to clarification). '
        r'The three blue rightmost columns show the \% of ambiguous questions per subset. '
        f'Direct prompt: {_prettify_direct_prompt(direct_prompt)}. Seed~{seed}, ref~{ref_type}, {split} split. '
        f'Colors {norm_desc} normalized.'
    )
    lines = [
        r'\begin{table}[ht]',
        r'\centering\scriptsize',
        r'\setlength{\tabcolsep}{4pt}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header,
        r'\midrule',
        *data_rows,
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:routing_profile}',
        r'\end{table}',
    ]
    latex = '\n'.join(lines)
    print(latex)
    return latex


def clarify_effectiveness_to_latex(metrics, config, metric_type='judge', ref_type='1', gradient_by='global', direct_prompt=None, final_prompt=None, settings=None):
    """Print (and return) a LaTeX table matching Block 2 from plot_clarify_effectiveness.

    gradient_by:
      "global" (default) — colour scale across all rows per column.
      "model"            — each model group gets its own scale.

    Requires in the LaTeX preamble:
        \\usepackage{booktabs}
        \\usepackage{multirow}
        \\usepackage[table]{xcolor}
    """
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
    _key_final = "final_judge"                                        if _j else "final"
    _key_cbase = ("direct_judge_anyref" if _any else "direct_judge") if _j else "direct_answer_clarifyref"

    clarify_effect_rows = {}
    for assistant_model, dp in settings_list:
        for belief_sampling in config['belief_samplings']:
            for reasoner_prompt in config['reasoner_prompts']:
                for fp in [_resolve_fp(reasoner_prompt, final_prompt, config)]:
                    m = metrics[assistant_model][dp][belief_sampling][reasoner_prompt][fp]
                    n_total = len(m['final'])

                    is_clarify = [m['reasoner_strategy'][i] == 'clarification_question' for i in range(n_total)]
                    dp_suffix  = f"_{dp}" if settings is not None else ""
                    fp_suffix  = "+" if fp != 'prompt' else ""
                    row_key    = (assistant_model, f"{reasoner_prompt}{fp_suffix}{dp_suffix}")

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

    df = pd.DataFrame(clarify_effect_rows).T
    df.index = pd.MultiIndex.from_tuples(
        [(_prettify_model(m), _prettify_reasoner_prompt(rp)) for m, rp in df.index],
        names=["model", "reasoner"],
    )

    show_abstain_cols = final_prompt != 'prompt'
    # (col, cmap_or_None, fixed_vmin, fixed_vmax) — None cmap = no gradient
    col_specs = [
        ('n_clarify_judged',  None,               None, None),
        ('n_wrong_base',      None,               None, None),
        ('n_right_base',      None,               None, None),
        ('recovery rate %',   plt.cm.RdYlGn,      None, None),
        ('regression rate %', plt.cm.RdYlGn_r,    None, None),
        ('net flip /N %',     plt.cm.RdYlGn,      -15,  15),
    ]
    if show_abstain_cols:
        col_specs += [
            ('abstain|wrong %', plt.cm.RdYlGn_r, None, None),
            ('abstain|right %', plt.cm.RdYlGn_r, None, None),
        ]

    col_names = [c for c, *_ in col_specs]

    def _col_bounds(col, fixed_vmin, fixed_vmax):
        if fixed_vmin is not None:
            return fixed_vmin, fixed_vmax
        return df[col].min(), df[col].max()

    global_bounds = {c: _col_bounds(c, vn, vx) for c, _, vn, vx in col_specs}
    model_bounds  = {}
    if gradient_by == "model":
        for model in df.index.get_level_values(0).unique():
            model_bounds[model] = {}
            for c, _, vn, vx in col_specs:
                if vn is not None:
                    model_bounds[model][c] = (vn, vx)
                else:
                    model_bounds[model][c] = (df.loc[model][c].min(), df.loc[model][c].max())

    int_cols = {'n_clarify_judged', 'n_wrong_base', 'n_right_base'}

    def _cell(value, cmap, vmin, vmax, is_int=False):
        if pd.isna(value):
            return '--'
        val_str = f'{int(value)}' if is_int else f'{value:.1f}'
        if cmap is None:
            return val_str
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        r, g, b, _ = cmap(max(0.0, min(1.0, t)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text = 'white' if lum < 0.4 else 'black'
        bg = f'\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}'
        return f'{bg}\\textcolor{{{text}}}{{{val_str}}}'

    col_spec   = 'll|' + 'c' * len(col_names)
    header     = ' & '.join(['', ''] + col_names) + r' \\'
    total_cols = 2 + len(col_names)

    models_order = list(dict.fromkeys(idx[0] for idx in df.index))
    data_rows = []
    for m_idx, model in enumerate(models_order):
        model_df = df.loc[model]
        n        = len(model_df)
        pretty   = _prettify_model(model, latex=True)
        for row_idx, (prompt, row) in enumerate(model_df.iterrows()):
            cells = []
            for c, cmap, vn, vx in col_specs:
                if gradient_by == "model" and cmap is not None:
                    vmin, vmax = model_bounds[model][c]
                else:
                    vmin, vmax = global_bounds[c]
                cells.append(_cell(row[c], cmap, vmin, vmax, is_int=c in int_cols))
            if row_idx == 0:
                label = f'\\multirow{{{n}}}{{*}}{{{pretty}}} & {prompt}'
            else:
                label = f' & {prompt}'
            data_rows.append(label + ' & ' + ' & '.join(cells) + r' \\')
            if row_idx < n - 1:
                data_rows.append(f'\\cline{{2-{total_cols}}}')
        if m_idx < len(models_order) - 1:
            data_rows.append(r'\arrayrulecolor{black!35}\specialrule{1pt}{2pt}{2pt}\arrayrulecolor{black}')

    seed      = config.get('seed', '?')
    split     = config.get('split', '?')
    norm_desc = 'per-model' if gradient_by == 'model' else 'globally'
    caption   = (
        f'Clarification effectiveness | {metric_type} | seed~{seed}, {split} split. '
        f'Conditioned on clarify; fixed denominators. Colors {norm_desc} normalized per column.'
    )
    lines = [
        r'\begin{table}[ht]',
        r'\centering\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header,
        r'\midrule',
        *data_rows,
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:clarify_effectiveness}',
        r'\end{table}',
    ]
    latex = '\n'.join(lines)
    print(latex)
    return latex


def direct_results_to_latex(metrics, config, sampling='unbiased', metric_type='judge', direct_prompt='concise'):
    """Print (and return) a LaTeX table matching display_direct_results (all questions).

    Two-level header: intent (1 intent / any intent) × method (Standard / Disambig / Belief) + #W.
    Colours cells with RdYlGn per-row gradient (matching the notebook style).
    Caption embeds all hidden variables so the table is self-contained in Overleaf.
    Requires in the LaTeX preamble:
        \\usepackage{booktabs}
        \\usepackage{multirow}
        \\usepackage[table]{xcolor}
    """
    df = _build_direct_df(metrics, config, metric_type, direct_prompt, sampling)

    _col_labels = {
        ("All", "Standard (1 intent)"):   "Standard",
        ("All", "Disambig (1 intent)"):   "Disambig",
        ("All", "Belief (1 intent)"):     "Belief",
        ("All", "Standard (any intent)"): "Standard",
        ("All", "Disambig (any intent)"): "Disambig",
        ("All", "Belief (any intent)"):   "Belief",
        ("All", "#words"):               r"\#W",
    }
    no_color_cols = {("All", "#words")}
    gradient_cols = [c for c in df.columns if c not in no_color_cols]
    cmap = plt.cm.RdYlGn

    def _cell(value, col, vmin, vmax):
        if pd.isna(value):
            return '--'
        val_str = f'{value:.1f}'
        if col in no_color_cols:
            return val_str
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        r, g, b, _ = cmap(max(0.0, min(1.0, t)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text = 'white' if lum < 0.4 else 'black'
        return f'\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}\\textcolor{{{text}}}{{{val_str}}}'

    multirow = direct_prompt is None
    n_label_cols = 2 if multirow else 1
    n_data_cols = len(df.columns)
    total_cols = n_label_cols + n_data_cols

    # intent spans: 1 intent (3 cols), any intent (3 cols), #W (1 col)
    intent_spans = [("1\\,intent", 3), ("any\\,intent", 3), ("", 1)]
    col_spec = 'l' * n_label_cols + '|ccc|ccc|c'

    if multirow:
        model_header = f'\\multicolumn{{2}}{{l|}}{{\\textbf{{Model}}}}'
    else:
        model_header = r'\textbf{Model}'

    # ── Header row 1: intent spans ──
    row1_parts = [model_header]
    for i, (label, span) in enumerate(intent_spans):
        fmt = 'c|' if i < len(intent_spans) - 1 else 'c'
        cell = f'\\multicolumn{{{span}}}{{{fmt}}}{{{label}}}' if label else f'\\multicolumn{{{span}}}{{{fmt}}}{{}}'
        row1_parts.append(cell)
    row1 = ' & '.join(row1_parts) + r' \\'

    col_idx = n_label_cols + 1
    cmidrules1 = []
    for _, span in intent_spans:
        cmidrules1.append(f'\\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}')
        col_idx += span

    # ── Header row 2: individual method labels ──
    row2 = ' & '.join([''] * n_label_cols + [_col_labels.get(c, c[1]) for c in df.columns]) + r' \\'

    # Data rows — per-row gradient normalization
    data_rows = []
    if multirow:
        models_order = list(dict.fromkeys(idx[0] for idx in df.index))
        for m_idx, model in enumerate(models_order):
            model_df = df.loc[model]
            n = len(model_df)
            pretty = _prettify_model(model, latex=True)
            for row_idx, (prompt, row) in enumerate(model_df.iterrows()):
                grad_vals = [row[c] for c in gradient_cols if not pd.isna(row[c])]
                vmin, vmax = (min(grad_vals), max(grad_vals)) if grad_vals else (0, 1)
                vals = [_cell(row[c], c, vmin, vmax) for c in df.columns]
                dp_label = _prettify_direct_prompt(prompt)
                if row_idx == 0:
                    label = f'\\multirow{{{n}}}{{*}}{{{pretty}}} & {dp_label}'
                else:
                    label = f' & {dp_label}'
                data_rows.append(label + ' & ' + ' & '.join(vals) + r' \\')
                if row_idx < n - 1:
                    data_rows.append(f'\\cline{{2-{total_cols}}}')
            if m_idx < len(models_order) - 1:
                data_rows.append(r'\midrule')
    else:
        for model, row in df.iterrows():
            grad_vals = [row[c] for c in gradient_cols if not pd.isna(row[c])]
            vmin, vmax = (min(grad_vals), max(grad_vals)) if grad_vals else (0, 1)
            vals = [_cell(row[c], c, vmin, vmax) for c in df.columns]
            data_rows.append(_prettify_model(model, latex=True) + ' & ' + ' & '.join(vals) + r' \\')

    seed = config.get('seed', '?')
    _lookup_prompt = (config['direct_prompts'][0] if direct_prompt is None else direct_prompt)
    n_examples = next(
        (len(metrics[m][_lookup_prompt][sampling]["is_ambig"])
         for m in config['assistant_models']
         if _lookup_prompt in metrics.get(m, {}) and sampling in metrics[m][_lookup_prompt]),
        '?'
    )
    prompt_str = 'all prompts' if direct_prompt is None else f'\\texttt{{{_prettify_direct_prompt(direct_prompt)}}}'
    split = config.get('split', '?')
    caption = (
        f'Direct generations for {n_examples}~ambigqa-{split} questions (all questions). '
        f'Seed~{seed}, {prompt_str}, {split} split. '
        f'Colors row-normalized (compares metrics within, not across models).'
    )

    lines = [
        r'\begin{table*}[ht]',
        r'\centering\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        row1,
        ' '.join(cmidrules1),
        row2,
        r'\midrule',
        *data_rows,
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:direct_results}',
        r'\end{table*}',
    ]
    result = '\n'.join(lines)
    print(result)
    return result
