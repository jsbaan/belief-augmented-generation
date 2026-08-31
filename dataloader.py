import functools
from collections import defaultdict
from typing import List, Dict, Any, Optional
import json
import os
import random

from transformers import pipeline

from config import build_output_fname

import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(
        dataset_path: str,
        max_examples: int = None,
        seed: int = None,
        keep_indices: List[int] = None,
) -> List[Dict[str, Any]]:

    """Load clarification dataset from JSON or JSONL file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON file: {e}")
                raise
    else:
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                    continue


    if seed:
        random.seed(seed)
        random.shuffle(data)
        logger.info(f"Shuffled dataset with seed {seed}")
    if max_examples and len(data) > max_examples:
        data = data[: max_examples]
        # logger.info(f"Limited dataset to {max_examples} examples")
    if keep_indices:
        data = [item for i, item in enumerate(data) if i in keep_indices]
        # logger.info(f"Filtering on {len(keep_indices)} indices")
    # logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data

def load_ambigqa(
        dataset_path: str,
        max_examples: int = None,
        filter_conflicts: bool = False,
        seed: int = 42
) -> List[Dict]:
    dataset = load_jsonl(dataset_path, seed=seed)

    # Sometimes there are annotations with conflicting types (multipleQAs and singeAnswer). This occurs for 10/10036
    # questions in train and 170/2002 in dev. Either filter them out, or the multipleQA annotation is leading.
    if filter_conflicts:
        dataset = [item for item in dataset if len(set([annot['type'] for annot in item['annotations']])) == 1]

    if max_examples and len(dataset) > max_examples:
        dataset = dataset[:max_examples]

    for item in dataset:
        disambiguations = []
        references = []
        for i, annotations in enumerate(item['annotations']):
            if annotations['type'] == 'multipleQAs':
                for qa_pairs in annotations['qaPairs']:
                    disambiguations.append(qa_pairs | {'annotation_index': i})
                    references.append(qa_pairs['answer']) # Appending a list with 1 or more surface form strings
            elif annotations['type'] == 'singleAnswer':
                references.append(annotations['answer']) # Appending a list with 1 or more surface form strings
        item['disambiguations'] = disambiguations
        item['references'] = references # List of lists of strings
        del item['annotations']

    return dataset

def load_pipeline_output(
        assistant_models: List[str],
        user_model: str,
        direct_prompts: List[str],
        reasoner_prompts: List[str],
        belief_samplings: List[str],
        steps: Optional[List[str]],
        seed: int,
        filter_evergreen: bool,
        split: str = "train",
        judge_model: Optional[str] = None,
        final_prompts: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        path: Optional[str] = None,
):
    if path is None:
        path = "../data/generations" if split == "train" else f"../data/generations/{split}"
    if 'all' in steps or steps is None:
        steps = ['direct', 'disambiguated', 'reasoner', 'user', 'final']
    outputs = defaultdict(lambda: defaultdict(dict))

    if filter_evergreen:
        # get questions from any of the pipeline outputs
        tmp_fname = build_output_fname("direct", assistant_models[0], direct_prompts[0], belief_samplings[0], seed)
        tmp_output = load_jsonl(f'{path}/{tmp_fname}.jsonl')
        questions = [d['context']['question'] for d in tmp_output]
        evergreen_indices = classify_evergreen(tuple(questions))
    else:
        evergreen_indices = None

    missing_files = set()

    def _load(fpath):
        try:
            return load_jsonl(fpath, keep_indices=evergreen_indices, max_examples=max_examples)
        except FileNotFoundError:
            missing_files.add(fpath)
            return None

    for assistant_model in assistant_models:
        for belief_sampling in belief_samplings:
            for direct_prompt in direct_prompts:
                output = defaultdict(dict)
                for pipeline_step in ["direct", "disambiguated"]:
                    base_args = {
                        "assistant_model": assistant_model,
                        "direct_prompt": direct_prompt,
                        "belief_sampling": belief_sampling,
                        "seed": seed
                    }
                    fname = build_output_fname(pipeline_step, **base_args)
                    if pipeline_step in steps:
                        data = _load(f'{path}/{fname}.jsonl')
                        if data is not None:
                            output[pipeline_step] = data

                if judge_model:
                    judge_base_args = {**base_args, "judge_model": judge_model}
                    if "direct" in steps:
                        fname = build_output_fname("judge", judge_branch="direct", **judge_base_args)
                        data = _load(f'{path}/{fname}.jsonl')
                        if data is not None:
                            output["direct_judge"] = data
                        fname = build_output_fname("judge", judge_branch="belief", **judge_base_args)
                        data = _load(f'{path}/{fname}.jsonl')
                        if data is not None:
                            output["belief_judge"] = data
                    if "disambiguated" in steps:
                        fname = build_output_fname("judge", judge_branch="disambig", **judge_base_args)
                        data = _load(f'{path}/{fname}.jsonl')
                        if data is not None:
                            output["disambig_judge"] = data

                for reasoner_prompt in reasoner_prompts:
                    reasoner_args = {**base_args, "reasoner_prompt": reasoner_prompt}
                    all_args = {**reasoner_args, "user_model": user_model}
                    # "prompt" is neutral; "prompt1" is SAG-only; "belief" is BAG-only — no cross-mixing
                    is_belief_reasoner = "belief" in reasoner_prompt
                    applicable_final_prompts = [
                        fp for fp in (final_prompts or ["prompt"])
                        if fp == "prompt" or ("belief" in fp) == is_belief_reasoner
                    ]
                    if "reasoner" in steps:
                        clarify_fname = build_output_fname("clarify", **reasoner_args)
                        data = _load(f'{path}/{clarify_fname}.jsonl')
                        if data is not None:
                            output["reasoner"][reasoner_prompt] = data
                    if "user" in steps:
                        user_fname = build_output_fname("user", **all_args)
                        data = _load(f'{path}/{user_fname}.jsonl')
                        if data is not None:
                            output["user"][reasoner_prompt] = data
                    if "final" in steps:
                        for final_prompt in applicable_final_prompts:
                            final_fname = build_output_fname("final", **all_args, final_prompt=final_prompt)
                            data = _load(f'{path}/{final_fname}.jsonl')
                            if data is not None:
                                output["final"].setdefault(reasoner_prompt, {})[final_prompt] = data
                    if judge_model:
                        judge_rp_args = {**judge_base_args, "reasoner_prompt": reasoner_prompt}
                        judge_all_args = {**judge_rp_args, "user_model": user_model}
                        if "reasoner" in steps:
                            fname = build_output_fname("judge", judge_branch="clarify", **judge_rp_args)
                            data = _load(f'{path}/{fname}.jsonl')
                            if data is not None:
                                output["clarify_judge"][reasoner_prompt] = data
                        if "final" in steps:
                            for final_prompt in applicable_final_prompts:
                                fname = build_output_fname("judge", judge_branch="final", **judge_all_args, final_prompt=final_prompt)
                                data = _load(f'{path}/{fname}.jsonl')
                                if data is not None:
                                    output["final_judge"].setdefault(reasoner_prompt, {})[final_prompt] = data
                outputs[assistant_model][direct_prompt][belief_sampling] = output

    if missing_files:
        lines = "\n".join(f"  MISSING: {f}" for f in sorted(missing_files))
        logger.warning(f"{len(missing_files)} file(s) not found and skipped:\n{lines}")

    return {k: dict(v) for k, v in outputs.items()}

# To avoid loading the running the model every time we load a single pipeline output, we will cache the results
@functools.lru_cache(maxsize=8)
def classify_evergreen(questions, model_name: str = "s-nlp/E5-EverGreen-Multilingual-Large"):
    pipe = pipeline("text-classification", model_name)
    evergreens = pipe(list(questions))
    evergreen_indices = [i for i, item in enumerate(evergreens) if item['label'] == 'Evergreen']
    return evergreen_indices

