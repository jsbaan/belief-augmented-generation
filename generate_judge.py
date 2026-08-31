#!/usr/bin/env python3
"""LLM judge evaluation for the interactive QA pipeline.

Evaluates a single pipeline branch using an LLM judge as a semantically-aware
alternative to ROUGE-L recall.

Key difference from ROUGE-L: the judge checks whether the model *asserts* the reference
as its main claim, rather than whether the reference string appears anywhere in the output.
This prevents long or hedging answers from getting undeserved credit.

Designed as a drop-in pipeline step with the same conventions as other generate_* modules:
  - Takes a pre-loaded item list + branch name + Config
  - Returns a list of output dicts (same shape as other steps)
  - Called from pipeline.py which handles file I/O via save_output
"""

import logging
from typing import List, Dict, Optional

from config import Config
from evaluation_utils import get_item_ref
from parse_utils import parse_llm_judge_eval_response
from prompts import create_llm_judge_eval_prompt, create_llm_judge_eval_prompt_anyref

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maps branch name to the key used in the output generations dict
_VERDICT_KEY = {
    "direct":  "direct",
    "disambig": "disambig",
    "clarify": "reason_direct",  # reasoner RESPONSE when strategy == direct_answer
    "final":   "final",
    "belief":  "belief",         # set coverage across K belief-state samples
}

# Anyref verdict keys — only for branches with multiple possible references.
# "disambig" anyref requires all_refs_lookup (AmbigQA refs not stored in disambig output).
_ANYREF_VERDICT_KEY = {
    "direct":   "direct_anyref",
    "clarify":  "reason_direct_anyref",
    "disambig": "disambig_anyref",
    "belief":   "belief_anyref",
    "final":    "final_anyref",
}


def _get_raw(gen_item) -> str:
    """Extract raw generation text, handling both parsed dicts and plain strings."""
    if isinstance(gen_item, dict):
        return gen_item.get('raw_response') or ''
    return gen_item or ''


def _aggregate_samples(sample_results: List[Optional[Dict]]) -> Dict:
    """Aggregate per-sample judge dicts into a single coverage verdict.

    Verdict semantics (mirrors ROUGE-L set coverage):
      1  — at least one sample asserts the reference
      0  — no sample asserts the reference (all returned 0)
     -1  — no usable verdict (all samples returned parse errors or were skipped)
    """
    verdicts_list = [r['verdict'] for r in sample_results if r is not None]
    if not verdicts_list:
        agg = -1
    elif any(v == 1 for v in verdicts_list):
        agg = 1
    elif all(v == -1 for v in verdicts_list):
        agg = -1
    else:
        agg = 0
    return {'verdict': agg, 'samples': sample_results}


def generate_llm_judge_eval(
    items: List[Dict],
    branch: str,
    config: Config,
    anyref: bool = False,
    all_refs_lookup: Optional[Dict[str, List]] = None,
) -> List[Dict]:
    """Run LLM judge over one pipeline branch.

    Evaluated branches:
      direct  — direct answer vs a randomly selected reference
      disambig — disambiguated answer vs its specific stored reference
      clarify — reasoner RESPONSE when strategy==direct_answer vs a randomly selected reference
      final   — final answer after clarification vs its specific stored reference
      belief  — set coverage: judge each of K belief-state samples; verdict=1 if any asserts the ref

    All judge calls are batched into a single generate_fn invocation for efficiency.
    Reference selection for direct/clarify/belief uses config.data_seed to match
    evaluation_utils.compute_metrics.

    Args:
        items:           output list from the corresponding pipeline step
        branch:          one of "direct", "disambig", "clarify", "final", "belief"
        config:          Config with judge model as generate_fn; data_seed for ref selection
        anyref:          if True, also run a second batch evaluating against ALL reference groups.
                         For "disambig" and "final" anyref, all_refs_lookup must be provided.
        all_refs_lookup: {item_id: all_refs} mapping used for disambig anyref (sourced from
                         the raw AmbigQA dataset). Only required when branch=="disambig" and
                         anyref==True; ignored otherwise.

    Returns:
        List of dicts, one per example:
          id:                str
          generation_config: dict
          generations:       {verdict_key: {verdict, reasoning, raw_response},
                              verdict_key_anyref: {...}}
                             For "belief": verdict dicts also contain a "samples" list with
                             per-sample results.
                             verdict: 1=correct, 0=incorrect, -1=unparseable, None=skipped
          context:           dict with question and the ref used
    """
    if branch not in _VERDICT_KEY:
        raise ValueError(f"Unknown branch {branch!r}. Must be one of {list(_VERDICT_KEY)}")

    verdict_key = _VERDICT_KEY[branch]
    anyref_verdict_key = _ANYREF_VERDICT_KEY.get(branch)  # None for "final"

    if anyref and branch == "disambig" and all_refs_lookup is None:
        logger.warning("anyref=True for 'disambig' branch but all_refs_lookup not provided — skipping anyref")
        anyref = False


    # ── Pass 1: collect all prompts, tagging each with item index ─────────────
    # For belief: tags are (item_idx, sample_idx) tuples; for all others: plain item_idx ints.
    call_tags: List = []
    call_prompts: List[List[Dict]] = []
    anyref_call_tags: List = []
    anyref_call_prompts: List[List[Dict]] = []
    all_contexts: List[Dict] = []

    for i, item in enumerate(items):
        ctx_dict = item.get('context') or {}
        question = ctx_dict.get('question') or ctx_dict.get('disambiguated_question')
        ctx: Dict = {'question': question}
        ctx['disambiguations'] = ctx_dict.get('disambiguations') or []
        gen_candidate = None
        ref = None

        if branch == "direct":
            all_refs = item['context']['references']
            ref = get_item_ref(item['id'], all_refs)
            # 'samples' is the legacy key used by early think-model runs; current runs use 'answer'
            gen_candidate = _get_raw((item['generations'].get('answer') or item['generations'].get('samples') or [{}])[0])
            ctx['direct_ref'] = ref
            if anyref and gen_candidate and len(all_refs) > 1:
                anyref_call_tags.append(i)
                anyref_call_prompts.append(create_llm_judge_eval_prompt_anyref(question, gen_candidate, all_refs))
                ctx['direct_all_refs'] = all_refs

        elif branch == "disambig":
            answer_list = (item.get('generations') or {}).get('answer')
            ref = ctx_dict.get('reference')
            if answer_list and ref:
                gen_candidate = _get_raw(answer_list[0])
                question = ctx_dict.get('disambiguated_question', question)
                ctx['question'] = question
                ctx['disambig_ref'] = ref
                if anyref and gen_candidate and all_refs_lookup is not None:
                    item_all_refs = all_refs_lookup.get(item['id'])
                    if item_all_refs:
                        anyref_call_tags.append(i)
                        anyref_call_prompts.append(create_llm_judge_eval_prompt_anyref(question, gen_candidate, item_all_refs))
                        ctx['disambig_all_refs'] = item_all_refs

        elif branch == "clarify":
            gens = item.get('generations') or {}
            ctx['strategy'] = gens.get('strategy')
            all_refs = item['context']['references']
            ref = get_item_ref(item['id'], all_refs)
            if gens.get('strategy') == 'direct_answer':
                gen_candidate = gens.get('response') or ''
                ctx['reason_direct_ref'] = ref
                if anyref and gen_candidate and len(all_refs) > 1:
                    anyref_call_tags.append(i)
                    anyref_call_prompts.append(create_llm_judge_eval_prompt_anyref(question, gen_candidate, all_refs))
                    ctx['reason_direct_all_refs'] = all_refs

        elif branch == "final":
            final_gens = item.get('generations')
            final_ref = (item.get('context') or {}).get('reference')
            if final_gens and final_ref:
                ref = final_ref[0] if isinstance(final_ref[0], list) else final_ref
                ctx['final_ref'] = ref
                if isinstance(final_gens, list):
                    gen_candidate = _get_raw(final_gens[0])
                elif isinstance(final_gens, dict):
                    ctx['strategy'] = final_gens.get('strategy')
                    if final_gens.get('strategy') == 'direct_answer':
                        gen_candidate = final_gens.get('response') or ''
                if anyref and gen_candidate and all_refs_lookup is not None:
                    item_all_refs = all_refs_lookup.get(item['id'])
                    if item_all_refs:
                        anyref_call_tags.append(i)
                        anyref_call_prompts.append(create_llm_judge_eval_prompt_anyref(question, gen_candidate, item_all_refs))
                        ctx['final_all_refs'] = item_all_refs

        elif branch == "belief":
            all_refs = item['context']['references']
            ref = get_item_ref(item['id'], all_refs)
            belief_gens = item['generations'].get('belief_state') or []
            ctx['belief_ref'] = ref
            is_ambig = len(all_refs) > 1
            if anyref and is_ambig:
                ctx['belief_all_refs'] = all_refs
            for j, gen_item in enumerate(belief_gens):
                candidate = _get_raw(gen_item)
                if candidate:
                    call_tags.append((i, j))
                    call_prompts.append(create_llm_judge_eval_prompt(question, candidate, ref))
                    if anyref and is_ambig:
                        anyref_call_tags.append((i, j))
                        anyref_call_prompts.append(create_llm_judge_eval_prompt_anyref(question, candidate, all_refs))

        ctx['generation_candidate'] = gen_candidate
        all_contexts.append(ctx)

        # Non-belief branches: add single prompt per item
        if branch != "belief" and gen_candidate and ref:
            call_tags.append(i)
            call_prompts.append(create_llm_judge_eval_prompt(question, gen_candidate, ref))

    # ── Pass 2: batched generate_fn calls ─────────────────────────────────────
    logger.info(f"Running {len(call_prompts)} LLM judge calls for branch '{branch}' across {len(items)} examples...")
    raw_responses = config.generate_fn(call_prompts, config)[0] if call_prompts else []

    anyref_raw_responses = []
    if anyref and anyref_call_prompts:
        logger.info(f"Running {len(anyref_call_prompts)} anyref LLM judge calls for branch '{branch}'...")
        anyref_raw_responses = config.generate_fn(anyref_call_prompts, config)[0]

    # ── Pass 3: parse and assemble ────────────────────────────────────────────
    verdicts = [None] * len(items)
    anyref_verdicts = [None] * len(items)

    if branch == "belief":
        per_item_samples: List[List] = [[] for _ in range(len(items))]
        for (item_idx, _), response_list in zip(call_tags, raw_responses):
            raw = response_list[0] if response_list else ''
            per_item_samples[item_idx].append(parse_llm_judge_eval_response(raw))
        for i, samples in enumerate(per_item_samples):
            if samples:
                verdicts[i] = _aggregate_samples(samples)

        if anyref_raw_responses:
            per_item_anyref: List[List] = [[] for _ in range(len(items))]
            for (item_idx, _), response_list in zip(anyref_call_tags, anyref_raw_responses):
                raw = response_list[0] if response_list else ''
                per_item_anyref[item_idx].append(parse_llm_judge_eval_response(raw))
            for i, samples in enumerate(per_item_anyref):
                if samples:
                    anyref_verdicts[i] = _aggregate_samples(samples)
    else:
        for item_idx, response_list in zip(call_tags, raw_responses):
            raw = response_list[0] if response_list else ''
            verdicts[item_idx] = parse_llm_judge_eval_response(raw)

        for item_idx, response_list in zip(anyref_call_tags, anyref_raw_responses):
            raw = response_list[0] if response_list else ''
            anyref_verdicts[item_idx] = parse_llm_judge_eval_response(raw)

    # For non-ambig items anyref == randomref (single ref), so copy the randomref verdict
    # rather than leaving anyref as None and silently dropping items from aggregate metrics.
    if anyref and anyref_verdict_key:
        for i in range(len(items)):
            if anyref_verdicts[i] is None and verdicts[i] is not None:
                anyref_verdicts[i] = verdicts[i]

    results = []
    for i in range(len(items)):
        generations = {verdict_key: verdicts[i]}
        if anyref and anyref_verdict_key:
            generations[anyref_verdict_key] = anyref_verdicts[i]
        results.append({
            'id': items[i]['id'],
            'generation_config': config.to_dict(),
            'generations': generations,
            'context': all_contexts[i],
        })
    return results
