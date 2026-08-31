#!/usr/bin/env python3
"""Faithfulness (belief-state diversity) evaluation for the BAG pipeline.

For each item, sends ONE LLM call with all K belief-state samples. The LLM
clusters the samples by main factual claim and counts how many distinct claims
it finds. This is a direct measure of belief-state diversity:

    n_distinct_claims == 1  →  model is confident  (→ should answer directly)
    n_distinct_claims  > 1  →  model is uncertain   (→ should clarify or abstain)

Designed as a drop-in pipeline step with the same conventions as generate_judge.py:
  - Takes a pre-loaded item list + Config
  - Returns a list of output dicts
  - Called from pipeline.py which handles file I/O via save_output
"""

import logging
from typing import List, Dict

from config import Config
from parse_utils import parse_claim_variation_response, parse_contextualisation_response
from prompts import create_claim_variation_prompt, create_contextualisation_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_belief_samples(item: Dict) -> List[str]:
    """Extract text of each belief-state sample from a direct-answer item.

    Uses 'response' when non-empty (same logic as generate_clarify.py), falling back
    to 'raw_response'. This gives cleaner answers without <think>...</think> preamble.
    """
    belief_state = (item.get('generations') or {}).get('belief_state') or []
    samples = []
    for gen in belief_state:
        if isinstance(gen, dict):
            text = gen.get('response') or gen.get('raw_response') or ''
        else:
            text = gen or ''
        if text:
            samples.append(text)
    return samples


def _get_final_belief_samples(item: Dict) -> List[str]:
    """Extract belief-state strings from a final-answer (final_prompt=belief) item.

    generate_final_answer.py stores these at the top level as item['belief_state'] (list of
    raw strings), unlike direct-answer items where samples are dicts under generations.
    """
    return [s for s in (item.get('belief_state') or []) if s]


def generate_claim_variation_eval(
    items: List[Dict],
    config: Config,
    sample_extractor=None,
) -> List[Dict]:
    """Run claim variation (belief-state claim diversity) evaluation.

    One LLM call per item: the LLM sees all K belief-state samples and clusters
    them by main factual claim. Output key is 'claim_variation'.

    Args:
        items:  direct-answer pipeline output (each item has 'generations.belief_state')
        config: Config with judge/API model as generate_fn

    Returns:
        List of dicts, one per example:
          id:                str
          generation_config: dict
          generations:       {'claim_variation': {n_distinct_claims, claims, raw_response}}
          context:           {question, n_belief_samples}
    """
    # ── Pass 1: collect prompts ───────────────────────────────────────────────
    call_tags: List[int] = []
    call_prompts: List[List[Dict]] = []
    all_contexts: List[Dict] = []

    for i, item in enumerate(items):
        ctx_dict = item.get('context') or {}
        question = ctx_dict.get('question') or ''
        samples = (sample_extractor or _get_belief_samples)(item)
        all_contexts.append({'question': question, 'n_belief_samples': len(samples)})

        if samples and question:
            call_tags.append(i)
            call_prompts.append(create_claim_variation_prompt(question, samples))

    # ── Pass 2: single batched generate_fn call ────────────────────────────────
    logger.info(
        f"Running {len(call_prompts)} claim_variation LLM calls across {len(items)} examples..."
    )
    raw_responses = config.generate_fn(call_prompts, config)[0] if call_prompts else []

    # ── Pass 3: parse and assemble ────────────────────────────────────────────
    parsed: List[Dict] = [None] * len(items)
    for item_idx, response_list in zip(call_tags, raw_responses):
        raw = response_list[0] if response_list else ''
        parsed[item_idx] = parse_claim_variation_response(raw)

    results = []
    for i in range(len(items)):
        results.append({
            'id': items[i]['id'],
            'generation_config': config.to_dict(),
            'generations': {'claim_variation': parsed[i]},
            'context': all_contexts[i],
        })
    return results


def generate_interpretation_variation_eval(
    items: List[Dict],
    config: Config,
) -> List[Dict]:
    """Run contextualisation (belief-state scope presence) evaluation.

    One LLM call per item: the LLM classifies each of the K belief-state samples
    as contextualised (contains an explicit scope marker not in the question),
    uncontextualised, clarifying, or refusing. Output key is 'interpretation_variation'.

    Args:
        items:  direct-answer pipeline output (each item has 'generations.belief_state')
        config: Config with judge/API model as generate_fn

    Returns:
        List of dicts, one per example:
          id:                str
          generation_config: dict
          generations:       {'interpretation_variation': {n_contextualised, n_uncontextualised,
                                                            n_clarifying, n_refusing,
                                                            classifications, raw_response}}
          context:           {question, n_belief_samples}
    """
    # ── Pass 1: collect prompts ───────────────────────────────────────────────
    call_tags: List[int] = []
    call_prompts: List[List[Dict]] = []
    all_contexts: List[Dict] = []

    for i, item in enumerate(items):
        ctx_dict = item.get('context') or {}
        question = ctx_dict.get('question') or ''
        samples = _get_belief_samples(item)
        all_contexts.append({'question': question, 'n_belief_samples': len(samples)})

        if samples and question:
            call_tags.append(i)
            call_prompts.append(create_contextualisation_prompt(question, samples))

    # ── Pass 2: single batched generate_fn call ────────────────────────────────
    logger.info(
        f"Running {len(call_prompts)} interpretation_variation LLM calls across {len(items)} examples..."
    )
    raw_responses = config.generate_fn(call_prompts, config)[0] if call_prompts else []

    # ── Pass 3: parse and assemble ────────────────────────────────────────────
    parsed: List[Dict] = [None] * len(items)
    for item_idx, response_list in zip(call_tags, raw_responses):
        raw = response_list[0] if response_list else ''
        parsed[item_idx] = parse_contextualisation_response(raw)

    results = []
    for i in range(len(items)):
        results.append({
            'id': items[i]['id'],
            'generation_config': config.to_dict(),
            'generations': {'interpretation_variation': parsed[i]},
            'context': all_contexts[i],
        })
    return results
