from config import Config
from evaluation_utils import get_item_ref
from parse_utils import parse_user_answer
from prompts import create_user_answer_prompt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_user_answer(clarification_qs, config: Config):
    prompt_map = {}
    context_map = {}

    for i, clarify_item in enumerate(clarification_qs):
        # Only generate a user answer when the reasoner followed the clarification strategy
        if clarify_item['generations']['strategy'] and "clarification" in clarify_item['generations']['strategy'] and clarify_item['generations']['response']:
            question = clarify_item['context']['question']
            clarification_q = clarify_item['generations']['response']
            disambiguations = clarify_item['context']["disambiguations"]

            # Note: empty disambiguations doesn't guarantee there is zero underspec/ambiguity: just not annotated
            disambiguation = get_item_ref(clarify_item['id'], disambiguations) if disambiguations else None
            disambig_question = disambiguation["question"] if disambiguations else None
            reference = [disambiguation["answer"]] if disambiguations else clarify_item['context']["references"]

            # Using first surface form reference as user answer: perhaps we can do this in a smarter way?
            prompt_map[i] = create_user_answer_prompt(question, clarification_q, disambig_question, reference[0][0])
            context_map[i] = {"question": question, "clarification_q": clarification_q, "reference": reference, "disambiguation": disambiguation}

    if not prompt_map:
        logger.info("No clarification questions to simulate user answers for. Skipping")
        generation_map = {}
        processed_prompt_map = {}
        resolved_params = {}
    else:
        raw_generations, processed_prompts, resolved_params = config.generate_fn(list(prompt_map.values()), config)
        parsed_generations = [parse_user_answer(raw_generation[0]) for raw_generation in raw_generations]
        generation_map = {idx: parsed_generations[i] for i, idx in enumerate(prompt_map.keys())}
        processed_prompt_map = {idx: processed_prompts[i] for i, idx in enumerate(prompt_map.keys())}

    outputs = [
        {
            "id": clarification_qs[i]["id"],
            "generation_config": config.to_dict(resolved_params=resolved_params),
            "generations": generation_map.get(i),
            "prompt": prompt_map.get(i),
            "processed_prompt": processed_prompt_map.get(i),
            "context": context_map.get(i),
        } for i in range(len(clarification_qs))
    ]

    return outputs
