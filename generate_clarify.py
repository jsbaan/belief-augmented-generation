#!/usr/bin/env python3
from typing import List, Dict

from config import Config
from parse_utils import parse_clarification_response
from prompts import (
    create_clarification_prompt,
    create_clarification_prompt1,
    create_clarification_prompt2,
    create_clarification_prompt3,
    create_clarification_prompt4,
    create_clarification_prompt5,
    create_clarification_prompt6,
    create_clarification_prompt7,
    create_clarification_prompt8,
    create_clarification_prompt9,
    create_clarification_prompt_no_belief,
)

_BELIEF_PROMPT_FNS = {
    "belief":  create_clarification_prompt,
    "belief1": create_clarification_prompt1,
    "belief2": create_clarification_prompt2,
    "belief3": create_clarification_prompt3,
    "belief4": create_clarification_prompt4,
    "belief5": create_clarification_prompt5,
    "belief6": create_clarification_prompt6,
    "belief7": create_clarification_prompt7,
    "belief8": create_clarification_prompt8,
    "belief9": create_clarification_prompt9,
}


def generate_clarification(belief_states: List[Dict], config: Config):
    prompts = []
    contexts = []
    for belief_state_item in belief_states:
        ctx = belief_state_item.get('context', belief_state_item)
        question = ctx['question']
        if "belief" in config.reasoner_prompt:
            belief_state = [gen['response'] if gen['response'] else gen['raw_response'] for gen in belief_state_item['generations']['belief_state']]
        else:
            belief_state = None
        references = ctx['references']  # List of lists of surface form strings
        disambiguations = ctx['disambiguations']

        if config.reasoner_prompt == "prompt":
            prompts.append(create_clarification_prompt_no_belief(question))
        else:
            prompts.append(_BELIEF_PROMPT_FNS[config.reasoner_prompt](question, belief_state))

        contexts.append({"question": question, "belief_state": belief_state, "references": references, "disambiguations": disambiguations})

    raw_responses, processed_prompts, resolved_params = config.generate_fn(prompts, config)

    # Parse raw responses
    parsed_responses = [parse_clarification_response(raw_response[0]) for raw_response in raw_responses]

    outputs = [
        {
            "id": belief_states[i]["id"],
            "generation_config": config.to_dict(resolved_params=resolved_params),
            "generations": parsed_responses[i],
            "prompt": prompts[i],
            "processed_prompt": processed_prompts[i],
            "context": contexts[i],
        } for i in range(len(belief_states))
    ]

    return outputs
