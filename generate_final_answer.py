#!/usr/bin/env python3
from dataclasses import replace

from config import Config
from parse_utils import parse_clarification_response
from prompts import (
    create_final_answer_prompt,
    create_final_answer_prompt1,
    create_final_belief_reasoner_prompt,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_FINAL_PROMPT_FNS = {
    "prompt":  create_final_answer_prompt,
    "prompt1": create_final_answer_prompt1,
}


def generate_final_answer(user_answers, config: Config):
    """Generate final answers after clarification.

    Dispatches based on config.final_prompt:
      "prompt"  — forced answer (original behavior, no abstain option)
      "prompt1" — prompt-only baseline: instructs model to answer or abstain
      "belief"  — BAG: sample K belief states from conversation, then reason over consensus
    """
    prompt_inputs = {}
    context_map = {}
    for i, user_answer_item in enumerate(user_answers):
        if user_answer_item['generations'] and user_answer_item['generations']['response']:
            question = user_answer_item['context']['question']
            clarification_q = user_answer_item['context']['clarification_q']
            user_answer = user_answer_item['generations']['response']
            disambiguation = user_answer_item['context']['disambiguation']
            reference = user_answer_item['context']['reference']
            prompt_inputs[i] = {"question": question, "clarification_q": clarification_q, "user_answer": user_answer}
            context_map[i] = {"question": question, "clarification_q": clarification_q, "user_answer": user_answer, "reference": reference, "disambiguation": disambiguation}

    if not prompt_inputs:
        logger.info("No clarification questions + user answers to respond to. Skipping")
        return [
            {
                "id": user_answers[i]["id"],
                "generation_config": config.to_dict(),
                "generations": None,
                "prompt": None,
                "processed_prompt": None,
                "context": None,
            } for i in range(len(user_answers))
        ]

    # ── Branch: BAG (belief) ─────────────────────────────────────────────────
    if "belief" in config.final_prompt:
        # Phase 1: sample K belief states using the full conversation as prompt.
        # Always use "sentence" brevity — we only need the factual claim, and short
        # samples keep the phase-2 reasoner input well within budget.
        sample_prompts = [create_final_answer_prompt(pi["question"], pi["clarification_q"], pi["user_answer"], "sentence")
                          for pi in prompt_inputs.values()]

        sample_config = replace(config, max_new_tokens=100)
        belief_generations, _, resolved_params = config.generate_fn(sample_prompts, sample_config)
        # Truncate each sample to 50 words as a hard fallback so the reasoner instructions
        # never get pushed off the end of the context window.
        _MAX_SAMPLE_WORDS = 50
        belief_map = {
            idx: [" ".join(s.split()[:_MAX_SAMPLE_WORDS]) if isinstance(s, str) else s
                  for s in belief_generations[j]]
            for j, idx in enumerate(prompt_inputs.keys())
        }

        # Phase 2: reason over consensus.
        # max_input_tokens raised to 2048: K=10 samples * ~50 words + base prompt fits comfortably.
        reasoner_prompts = [create_final_belief_reasoner_prompt(
                                prompt_inputs[i]["question"], prompt_inputs[i]["clarification_q"],
                                prompt_inputs[i]["user_answer"], config.direct_prompt, belief_map[i])
                            for i in prompt_inputs.keys()]

        reasoner_config = replace(config, sampling="recommended", n_samples=1, temperature=None, max_input_tokens=2048)
        reasoner_generations, reasoner_processed_prompts, _ = config.generate_fn(reasoner_prompts, reasoner_config)

        generations_map = {idx: parse_clarification_response(reasoner_generations[j][0])
                           for j, idx in enumerate(prompt_inputs.keys())}
        reasoner_prompt_map = {idx: reasoner_prompts[j] for j, idx in enumerate(prompt_inputs.keys())}
        processed_prompt_map = {idx: reasoner_processed_prompts[j] for j, idx in enumerate(prompt_inputs.keys())}

        return [
            {
                "id": user_answers[i]["id"],
                "generation_config": config.to_dict(resolved_params=resolved_params),
                "generations": generations_map.get(i),
                "belief_state": belief_map.get(i),
                "prompt": reasoner_prompt_map.get(i),
                "processed_prompt": processed_prompt_map.get(i),
                "context": context_map.get(i),
            } for i in range(len(user_answers))
        ]

    # ── Branch: prompt / prompt1 ─────────────────────────────────────────────
    prompt_fn = _FINAL_PROMPT_FNS[config.final_prompt]
    prompts = [prompt_fn(pi["question"], pi["clarification_q"], pi["user_answer"], config.direct_prompt)
               for pi in prompt_inputs.values()]

    generations, processed_prompts, resolved_params = config.generate_fn(prompts, config)

    # prompt1 returns structured STRATEGY/RESPONSE output that needs parsing; plain prompt returns raw text
    if config.final_prompt == "prompt":
        generations_map = {idx: generations[j] for j, idx in enumerate(prompt_inputs.keys())}
    else:
        generations_map = {idx: parse_clarification_response(generations[j][0])
                           for j, idx in enumerate(prompt_inputs.keys())}

    prompt_map = {idx: prompts[j] for j, idx in enumerate(prompt_inputs.keys())}
    processed_prompt_map = {idx: processed_prompts[j] for j, idx in enumerate(prompt_inputs.keys())}

    return [
        {
            "id": user_answers[i]["id"],
            "generation_config": config.to_dict(resolved_params=resolved_params),
            "generations": generations_map.get(i),
            "prompt": prompt_map.get(i),
            "processed_prompt": processed_prompt_map.get(i),
            "context": context_map.get(i),
        } for i in range(len(user_answers))
    ]
