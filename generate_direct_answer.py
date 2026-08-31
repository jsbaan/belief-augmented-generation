#!/usr/bin/env python3
from dataclasses import replace
from typing import List, Dict

from config import Config
from evaluation_utils import get_item_ref
from parse_utils import parse_direct_response
from prompts import create_direct_answer_prompt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_direct_answer(qa_dataset: List[Dict], config: Config):
    """Process dataset and generate belief state samples."""

    # Prepare prompts
    prompts = []
    contexts = []
    for i, qa_item in enumerate(qa_dataset):
        question = qa_item['question']
        references = qa_item["references"] # List of references, which are lists of surface form strings
        disambiguations = qa_item['disambiguations']
        prompts.append(create_direct_answer_prompt(question, direct_prompt=config.direct_prompt))
        contexts.append({"question": question, "references": references, "disambiguations": disambiguations})

    # Generate using injected function
    # 1. Belief state: unbiased sampling, K samples
    belief_generations, processed_prompts, resolved_params_belief = config.generate_fn(prompts, config)
    # 2. Answer: recommended sampling, 1 sample
    answer_config = replace(config, sampling="recommended", n_samples=1, temperature=None)
    answer_generations, _, _ = config.generate_fn(prompts, answer_config)
    # 3. Greedy: optional, for evaluation comparison only
    if config.no_greedy:
        greedy_generations = [[] for _ in belief_generations]
    else:
        greedy_config = replace(config, temperature=0.0, n_samples=1)
        greedy_generations, _, _ = config.generate_fn(prompts, greedy_config)

    # Construct outputs
    outputs = [
        {
            "id": qa_dataset[i]["id"],
            "generation_config": config.to_dict(resolved_params=resolved_params_belief),
            "generations": {
                "belief_state": [parse_direct_response(gen) for gen in belief_generations[i]],
                "answer": [parse_direct_response(gen) for gen in answer_generations[i]],
                "greedy": [parse_direct_response(gen) for gen in greedy_generations[i]],
            },
            "prompt": prompts[i],
            "processed_prompt": processed_prompts[i],
            "context": contexts[i],
        } for i in range(len(qa_dataset))
    ]

    return outputs

def generate_direct_answer_disambiguated(qa_dataset: List[Dict], config: Config):
    """Process dataset and generate belief state samples."""

    # Prepare prompts
    prompt_map = {}
    context_map = {}
    for i, qa_item in enumerate(qa_dataset):
        if qa_item['disambiguations']:
            disambiguation = get_item_ref(qa_item['id'], qa_item['disambiguations'])
            disambiguated_question = disambiguation["question"]
            reference = disambiguation["answer"] # List of surface form strings
            prompt_map[i] = create_direct_answer_prompt(disambiguated_question, direct_prompt=config.direct_prompt)
            context_map[i] = {"disambiguated_question": disambiguated_question, "reference": reference}

    if not prompt_map:
        logger.info("No disambiguation annotations. Skipping")
        answer_generation_map = {}
        greedy_generation_map = {}
        processed_prompt_map = {}
        resolved_params_answer = {}
    else:
        answer_generations, processed_prompts, resolved_params_answer = config.generate_fn(list(prompt_map.values()), config)

        if config.no_greedy:
            greedy_generations = [[] for _ in answer_generations]
        else:
            greedy_config = replace(config, temperature=0.0, n_samples=1)
            greedy_generations, _, _ = config.generate_fn(list(prompt_map.values()), greedy_config)

        # Mapping back
        answer_generation_map = {idx: answer_generations[i] for i, idx in enumerate(prompt_map.keys())}
        greedy_generation_map = {idx: greedy_generations[i] for i, idx in enumerate(prompt_map.keys())}
        processed_prompt_map = {idx: processed_prompts[i] for i, idx in enumerate(prompt_map.keys())}


    # Construct outputs
    outputs = [
        {
            "id": qa_dataset[i]["id"],
            "generation_config": config.to_dict(resolved_params=resolved_params_answer),
            "generations": {
                "answer": answer_generation_map.get(i),
                "greedy": greedy_generation_map.get(i),
            },
            "prompt": prompt_map.get(i),
            "processed_prompt": processed_prompt_map.get(i),
            "context": context_map.get(i),
        } for i in range(len(qa_dataset))
    ]

    return outputs
