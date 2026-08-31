import collections
import logging
import re
import string
import random
from typing import List, Dict, Optional, Any, Callable

from functools import lru_cache

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def extract_gen(item) -> str:
    return item['raw_response'] if isinstance(item, dict) else item


def normalize(s: str) -> Optional[str]:
    """Lower text and remove punctuation, articles and extra whitespace. From Zhang et al."""
    if s is None:
        return None

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_item_ref(item_id, all_refs):
    """Deterministically select a reference group for an item using per-item seeding.

    Uses an isolated RNG per item_id so the result is independent of iteration order,
    batch composition, and run seed — any pipeline step gets the same ref for the same item.
    """
    return random.Random(str(item_id)).choice(all_refs)


def evaluate_exact_match(
    samples: List[str], greedy: str, reference_answers: List[str], do_normalize: bool = True
) -> Dict:
    """Evaluate how many reference answer are in model generations."""

    if do_normalize:
        greedy = normalize(greedy)
        samples = [normalize(sample) for sample in samples]
        reference_answers = [normalize(ref) for ref in reference_answers if ref is not None]
        # Filter out empty references after normalization
        reference_answers = [ref for ref in reference_answers if ref.strip()]

    greedy_reference_count = 0
    samples_reference_count = [0 for _ in range(len(samples))]
    unique_reference_answers = list(set(reference_answers))
    set_reference_count = [0 for _ in range(len(unique_reference_answers))]

    for r, ref_answer in enumerate(unique_reference_answers):
        if ref_answer and ref_answer in greedy:
            greedy_reference_count += 1

        for i, sample in enumerate(samples):
            if ref_answer and ref_answer in sample:
                samples_reference_count[i] += 1
                set_reference_count[r] = 1
    greedy_reference_coverage = greedy_reference_count / len(reference_answers)
    samples_reference_coverage = [count / len(reference_answers) for count in samples_reference_count]

    scores = {
        "greedy": {
            "any": 1 if greedy_reference_count > 0 else 0,  # is any of the references in the greedy answer?
            "coverage": greedy_reference_coverage,  # what fraction of all references in the greedy answer?
        },
        "samples": {
            "any": [
                1 if count > 0 else 0 for count in samples_reference_count
            ],  # any of the references in each sample?
            "coverage": samples_reference_coverage,  # what fraction of all references in each sample?
            "set_coverage": sum(set_reference_count)
            / len(unique_reference_answers),  # how many of the unique references are covered by the entire sample set?
        },
    }

    return scores


def evaluate_text_length(samples: List[str], greedy: str) -> Dict:
    """Evaluate text lengths in words for samples and greedy generation."""

    # Calculate lengths using word count (split by spaces)
    greedy_length = len(greedy.split(" ")) if greedy else 0
    sample_lengths = [len(sample.split(" ")) if sample else 0 for sample in samples]

    # Calculate mean and std of sample lengths
    mean_sample_length = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0

    # Calculate standard deviation
    if len(sample_lengths) > 1:
        variance = sum((x - mean_sample_length) ** 2 for x in sample_lengths) / len(sample_lengths)
        std_sample_length = variance**0.5
    else:
        std_sample_length = 0.0

    scores = {
        "greedy": {
            "length": greedy_length,
        },
        "samples": {
            "lengths": sample_lengths,
            "mean": mean_sample_length,
            "std": std_sample_length,
        },
    }

    return scores


@lru_cache(maxsize=1)
def get_rouge_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

@lru_cache(maxsize=None)
def _evaluate_rouge_cached(references: tuple, generation: str, rouge_type: str):
    if rouge_type == 'recall':
        extract = lambda x: x['rougeL'].recall
    elif rouge_type == 'precision':
        extract = lambda x: x['rougeL'].precision
    elif rouge_type == 'fmeasure':
        extract = lambda x: x['rougeL'].fmeasure
    else:
        raise ValueError(f"Invalid metric: {rouge_type}")

    scorer = get_rouge_scorer()
    scores = []
    for reference_type in references:
        highest_score = extract(scorer.score("no", "match"))
        for surface_form in reference_type:
            new_score = extract(scorer.score(normalize(surface_form), normalize(generation)))
            if new_score > highest_score:
                highest_score = new_score
        scores.append(highest_score)
    return scores


def evaluate_rouge(references: List[List[str]], generation: str, rouge_type: str):
    if not isinstance(references, list) or not all(isinstance(ref, list) for ref in references):
        raise TypeError("References must be a list of lists.")
    if not isinstance(generation, str):
        raise TypeError("Generation must be a string.")
    return list(_evaluate_rouge_cached(tuple(tuple(r) for r in references), generation, rouge_type))


def information_leak(user_output: List[Dict], final_output: List[Dict], thresh: int = 1, do_print: bool = False):
    """
    output: dict with keys 'direct', 'disambiguated', 'reasoner', 'user', 'final' for a single pipeline configuration

    I could probably move this logic to the metric function below

    """
    metrics = {
        "rougel": collections.defaultdict(list),
        "count": collections.defaultdict(int)
    }

    for user, final in zip(user_output, final_output):
        if user['generations'] and user['generations']['response'] and final and final.get('context'):
            metrics["rougel"]["clarification_qs"].append(max(evaluate_rouge(user['context']['reference'], user['context']['clarification_q'], 'recall')))
            metrics["rougel"]["user"].append(max(evaluate_rouge(user['context']['reference'], user['generations']['response'], 'recall')))
            gens = final['generations']
            final_text = gens[0] if isinstance(gens, list) else (gens.get('response') or '' if isinstance(gens, dict) and gens.get('strategy') == 'direct_answer' else '')
            metrics["rougel"]["final"].append(max(evaluate_rouge(final['context']['reference'], final_text, 'recall')))

            metrics['count']['clarification_qs'] += 1 if metrics['rougel']['clarification_qs'][-1] >= thresh else 0
            metrics['count']['user'] += 1 if metrics['rougel']['user'][-1] >= thresh else 0
            metrics['count']['final'] += 1 if metrics['rougel']['final'][-1] >= thresh else 0

            # Number of times the reference is in the user generation but not in the clarification question (=leak)
            metrics['count']['user_only'] += 1 if metrics['rougel']['user'][-1] >= thresh and metrics['rougel']['clarification_qs'][-1] == 0 else 0
            # Number of times the reference is in the final generation but not in the user generation (no leak)
            metrics['count']['final_only'] += 1 if metrics['rougel']['final'][-1] >= thresh and metrics['rougel']['user'][-1] == 0 else 0

            metrics['count']['total'] += 1

            # Print problematic examples
            if do_print and (metrics['rougel']['user'][-1] >= thresh and metrics['rougel']['clarification_qs'][-1] == 0):
                print("Leak example:")
                print(f"> Question: {user['context']['question']}")
                print(f"> Clarification: {user['context']['clarification_q']}")
                print(f"> User answer: {user['generations']['response']}")
                print(f"> Final answer: {final['generations']}")
                print(f"> Reference: {user['context']['reference']}")
                print()

    # print("\nReference occurs in:")
    # from pprint import pprint
    # pprint(dict(metrics['count']))
    # print("\nAverage rougeL:")
    # pprint({k: np.mean(v) for k,v in metrics['rougel'].items()})
    # print()

    # print(f"{metrics['count']['user_only']} out of {metrics['count']['final']} ({metrics['count']['user_only'] / metrics['count']['final']*100:.1f}%) correct final generations are leaked by the user simulator.")
    return metrics

def _judge_verdict(judge_item, verdict_key) -> Optional[int]:
    """Extract a judge verdict (0 or 1), returning None for missing or unparseable (-1) results."""
    if judge_item is None:
        return None
    v = (judge_item.get('generations') or {}).get(verdict_key)
    if not v:
        return None
    verdict = v.get('verdict')
    return verdict if verdict in (0, 1) else None  # -1 (parse error) and None both → None


def compute_metrics(
        output,
        rougel_type: str= 'recall',
        agg: Callable = max,
        do_print=False,
        belief: bool = False,
        skip_rouge: bool = False,
):
    """
    output: dict with keys 'direct', 'disambiguated', 'reasoner', 'user', 'final' for a single pipeline configuration.
    Optionally also: 'direct_judge', 'disambig_judge', 'clarify_judge', 'final_judge' — lists of judge eval items
    parallel to the corresponding pipeline step lists. Judge verdicts of -1 (parse error) are treated as None.
    """

    warning_count = 0
    metrics = collections.defaultdict(list)

    def rouge(refs, gen):
        if skip_rouge:
            return None
        return agg(evaluate_rouge(refs, gen, rougel_type))

    n = len(output['direct'])
    direct_judge_items   = output.get('direct_judge')  or [None] * n
    disambig_judge_items = output.get('disambig_judge') or [None] * n
    clarify_judge_items  = output.get('clarify_judge') or [None] * n
    final_judge_items    = output.get('final_judge')   or [None] * n
    belief_judge_items   = output.get('belief_judge')  or [None] * n

    # Greedy is config-level (no_greedy flag): either present for all items or none.
    # Check once so the per-item loop only handles data-conditional absences.
    has_greedy = bool((output['direct'][0]['generations'].get('greedy') or []))
    for direct, disambig, reasoner, user, final, direct_j, disambig_j, clarify_j, final_j, belief_j in zip(
        output['direct'], output['disambiguated'], output['reasoner'], output['user'], output['final'],
        direct_judge_items, disambig_judge_items, clarify_judge_items, final_judge_items, belief_judge_items,
    ):
        ########## Evaluate DIRECT ##########
        question = direct['context']['question']

        # Extract direct generations
        gens = direct['generations']
        direct_answer_sample = (gens.get('answer') or gens.get('samples') or [{}])[0]  # older think-model runs used 'samples' instead of 'answer'
        direct_answer_gen = extract_gen(direct_answer_sample)
        direct_belief_gens = [extract_gen(g) for g in gens.get('belief_state', [])]  # older think-model runs lack belief_state
        direct_greedy_gen = extract_gen(direct['generations']['greedy'][0]) if has_greedy else None

        all_refs = direct['context']['references']
        random_ref = (disambig['context']['reference'] if disambig['context'] else all_refs[0]) if not skip_rouge else None

        metrics['num_disambigs'].append(len(direct['context']['disambiguations']))
        metrics["direct_answer_anyref"].append(rouge(all_refs, direct_answer_gen))
        metrics["direct_answer_randomref"].append(rouge([random_ref], direct_answer_gen))
        reasoning_words = len((direct_answer_sample.get('reasoning') or '').split()) if isinstance(direct_answer_sample, dict) else 0
        metrics["direct_answer_length"].append(len((direct_answer_gen or '').split()) + reasoning_words)
        if belief:
            metrics["direct_belief_anyref"].append(max(rouge(all_refs, g) for g in direct_belief_gens) if not skip_rouge else None)
            metrics["direct_belief_randomref"].append(max(rouge([random_ref], g) for g in direct_belief_gens) if not skip_rouge else None)
            metrics["direct_belief_length"].append(sum(len(g.split()) for g in direct_belief_gens) / len(direct_belief_gens) if direct_belief_gens else None)
        metrics["direct_greedy_anyref"].append(rouge(all_refs, direct_greedy_gen) if has_greedy else None)
        metrics["direct_greedy_randomref"].append(rouge([random_ref], direct_greedy_gen) if has_greedy else None)
        metrics["direct_greedy_length"].append(len(direct_greedy_gen.split()) if has_greedy else None)

        direct_judge_v    = _judge_verdict(direct_j, 'direct')
        direct_anyref_v   = _judge_verdict(direct_j, 'direct_anyref')
        belief_judge_v    = _judge_verdict(belief_j, 'belief')
        belief_anyref_v   = _judge_verdict(belief_j, 'belief_anyref')
        metrics["direct_judge"].append(direct_judge_v)
        # Non-ambig examples have 1 ref so anyref is not run; fall back to single-ref (semantically identical).
        metrics["direct_judge_anyref"].append(direct_anyref_v if direct_anyref_v is not None else direct_judge_v)
        metrics["belief_judge"].append(belief_judge_v)
        metrics["belief_judge_anyref"].append(belief_anyref_v if belief_anyref_v is not None else belief_judge_v)

        # Mean verdict across individual belief samples = expected quality of a single unbiased draw.
        belief_gens = (belief_j or {}).get('generations') or {}
        _bverdicts     = [s['verdict'] for s in (belief_gens.get('belief')      or {}).get('samples', []) if s.get('verdict') in (0, 1)]
        _bverdicts_any = [s['verdict'] for s in (belief_gens.get('belief_anyref') or {}).get('samples', []) if s.get('verdict') in (0, 1)]
        metrics["belief_judge_sample"].append(sum(_bverdicts) / len(_bverdicts) if _bverdicts else None)
        metrics["belief_judge_sample_anyref"].append(sum(_bverdicts_any) / len(_bverdicts_any) if _bverdicts_any else None)

        # Optionally print information about the direct generation
        if do_print in ['direct', 'all']:
            print("---")
            print(f"\n>Mode: Direct")
            print(f"Question: {question}")
            print(f"Generation (answer): {direct_answer_gen}")
            print(f"References: {all_refs} | Score: {metrics['direct_answer_anyref'][-1]}")
            print(f"Random ref: {random_ref} | Score: {metrics['direct_answer_randomref'][-1]}")

        ########## Evaluate DISAMBIGUATED ##########
        disambig_answer = disambig['generations']['answer'] if disambig else None  # disambig step may not have been run
        metrics["is_ambig"].append(bool(disambig_answer))
        if not disambig_answer:
            metrics["disambig_answer"].append(None)
            metrics["disambig_answer_any"].append(None)
            metrics["direct_answer_disambigref"].append(None)
            metrics["direct_notdisambig"].append(rouge(all_refs, direct_answer_gen))
            metrics["disambig_judge"].append(None)
            metrics["disambig_judge_anyref"].append(None)

            # Check that for non-ambig questions, there is only a single ref
            # if metrics['direct_greedy_anyref'][-1] != metrics['direct_greedy_randomref'][-1]:
            #     print("Warning: anyref != randomref for non-ambig, 2 distinct refs as surface forms:", direct['context']['question'], direct['context']['references'])
            #     warning_count += 1
        else:
            disambig_answer_gen = extract_gen(disambig_answer[0])
            disambig_ref = disambig['context']['reference']

            metrics["disambig_answer"].append(rouge([disambig_ref], disambig_answer_gen))
            metrics["disambig_answer_any"].append(rouge(all_refs, disambig_answer_gen))
            metrics["direct_answer_disambigref"].append(rouge([disambig_ref], direct_answer_gen))
            metrics["direct_notdisambig"].append(None)
            metrics["disambig_judge"].append(_judge_verdict(disambig_j, 'disambig'))
            metrics["disambig_judge_anyref"].append(_judge_verdict(disambig_j, 'disambig_anyref'))

            # Optionally print info
            if do_print in ['disambig', 'all']:
                print(f"\n>Mode: Disambig")
                print(f"Disambiguated question: {disambig['context']['disambiguated_question']}")
                print(f"Generation (answer): {disambig_answer_gen}")
                print(f"Disambig ref: {disambig_ref} | score: {metrics['disambig_answer'][-1]}")

        ########## Evaluate REASONER.direct ##########
        metrics["reasoner_strategy"].append(reasoner['generations']['strategy'])
        reason_response = reasoner['generations']['response'] if reasoner['generations']['strategy'] == "direct_answer" else None
        metrics["reason_direct_anyref"].append(rouge(all_refs, reason_response) if reason_response is not None else None)
        metrics["reason_direct_randomref"].append(rouge([random_ref], reason_response) if reason_response is not None else None)
        metrics["direct_answer_reasoner.direct_randomref"].append(rouge([random_ref], direct_answer_gen) if reason_response is not None else None)
        metrics["reason_direct_judge"].append(_judge_verdict(clarify_j, 'reason_direct'))
        metrics["reason_direct_judge_anyref"].append(_judge_verdict(clarify_j, 'reason_direct_anyref'))
        if do_print in ['reasoner.direct', 'all'] and reason_response is not None:
            print(f"\n>Mode: Reason-Direct")
            print(f"Generation: {reason_response}")
            print(f"References: {all_refs} | Score: {metrics['reason_direct_anyref'][-1]}")

        ########## Evaluate REASONER.abstain ##########
        metrics["direct_answer_abstain_randomref"].append(
            rouge([random_ref], direct_answer_gen) if reasoner['generations']['strategy'] == "abstain" else None
        )
        _is_abstain = reasoner['generations']['strategy'] == "abstain"
        metrics["direct_judge_abstain"].append(direct_judge_v if _is_abstain else None)
        metrics["direct_judge_abstain_anyref"].append(
            (direct_anyref_v if direct_anyref_v is not None else direct_judge_v) if _is_abstain else None
        )

        ########## Evaluate USER ##########
        if user['generations'] and user['generations']['response']:
            user_ref = user['context']['reference'] # the reference used by the user simulator to answer the CQ
            user_gen = user['generations']['response']
            metrics["user"].append(rouge(user_ref, user_gen))

        else:
            metrics["user"].append(None)


        ########## Evaluate FINAL ##########
        gens = final['generations']
        if isinstance(gens, list):
            final_gen = gens[0] if gens else None        # prompt variant: raw text list
        elif isinstance(gens, dict):
            strategy = gens.get('strategy')
            final_gen = gens.get('response') if strategy == 'direct_answer' else None  # prompt1/belief: abstain → None
        else:
            final_gen = None
        if final_gen:
            user_ref = final['context']['reference'] # the reference used by the user simulator to answer the CQ
            clarification_q = final['context']['clarification_q']
            user_gen = final['context']['user_answer']

            metrics["final"].append(rouge(user_ref, final_gen))
            metrics["direct_answer_clarifyref"].append(rouge(user_ref, direct_answer_gen))
            metrics["clarify_clarifyref"].append(rouge(user_ref, clarification_q))
            metrics["final_judge"].append(_judge_verdict(final_j, 'final'))
            metrics["final_judge_anyref"].append(_judge_verdict(final_j, 'final_anyref'))

            if do_print in ['final', 'all']:
                print(f"\n>Mode: Final")
                print(f"- Question: {question}")
                print(f"- Direct: {direct_answer_gen}")
                print(f"- Clarification: {clarification_q}")
                print(f"- User answer: {user_gen}")
                print(f"- Generation: {final_gen}")
                print(f"- References: {user_ref} | Score: {metrics['final'][-1]}")
        else:
            metrics["final"].append(None)
            metrics["direct_answer_clarifyref"].append(None)
            metrics["clarify_clarifyref"].append(None)
            metrics["final_judge"].append(None)
            metrics["final_judge_anyref"].append(None)

    # Compute final interactive performance and compare to baseline
    for i in range(len(metrics['reason_direct_randomref'])): # this length is equal for all modes
        # Include reasoner.direct in the interactive pipeline and the direct generations in the baseline
        if metrics['reason_direct_randomref'][i] is not None:
            metrics['full_interactive'].append(metrics['reason_direct_randomref'][i])
            metrics['full_baseline'].append(metrics['direct_answer_randomref'][i])
        # Exclude reasoner.abstain from the interactive pipeline but include direct generations in the baseline.
        elif metrics["direct_answer_abstain_randomref"][i] is not None:
            metrics['full_interactive'].append(None)
            metrics['full_baseline'].append(metrics["direct_answer_abstain_randomref"][i])
        # Include the final generation in the interactive pipeline, and direct generations against the same ref in the baseline
        elif metrics['final'][i] is not None:
            metrics['full_interactive'].append(metrics['final'][i])
            metrics['full_baseline'].append(metrics['direct_answer_clarifyref'][i])
        # Exclude parsing or format errors from both
        else:
            metrics['full_interactive'].append(None)
            metrics['full_baseline'].append(None)

        # Judge-based full interactive/baseline (same routing logic as ROUGE; direct_judge used as baseline throughout)
        if metrics['reason_direct_judge'][i] is not None:
            metrics['full_interactive_judge'].append(metrics['reason_direct_judge'][i])
            metrics['full_baseline_judge'].append(metrics['direct_judge'][i])
        elif metrics['reasoner_strategy'][i] == 'abstain':
            metrics['full_interactive_judge'].append(None)
            metrics['full_baseline_judge'].append(metrics['direct_judge'][i])
        elif metrics['final_judge'][i] is not None:
            metrics['full_interactive_judge'].append(metrics['final_judge'][i])
            metrics['full_baseline_judge'].append(metrics['direct_judge'][i])
        else:
            metrics['full_interactive_judge'].append(None)
            metrics['full_baseline_judge'].append(None)

        # Anyref judge-based full interactive/baseline (same routing; direct_judge_anyref as baseline)
        if metrics['reason_direct_judge_anyref'][i] is not None:
            metrics['full_interactive_judge_anyref'].append(metrics['reason_direct_judge_anyref'][i])
            metrics['full_baseline_judge_anyref'].append(metrics['direct_judge_anyref'][i])
        elif metrics['reasoner_strategy'][i] == 'abstain':
            metrics['full_interactive_judge_anyref'].append(None)
            metrics['full_baseline_judge_anyref'].append(metrics['direct_judge_anyref'][i])
        elif metrics['final_judge_anyref'][i] is not None:
            metrics['full_interactive_judge_anyref'].append(metrics['final_judge_anyref'][i])
            metrics['full_baseline_judge_anyref'].append(metrics['direct_judge_anyref'][i])
        else:
            metrics['full_interactive_judge_anyref'].append(None)
            metrics['full_baseline_judge_anyref'].append(None)

        # For questions with disambiguations, add the recommended answer for a single disambiguation to the upperbound
        if metrics['disambig_answer'][i] is not None:
            metrics['full_disambig_upperbound'].append(metrics['disambig_answer'][i])
        # For questions without disambiguations, add the direct generation against the only reference to the upperbound
        else:
            metrics['full_disambig_upperbound'].append(metrics['direct_answer_anyref'][i])

        # Judge equivalent: disambig verdict for ambiguous, direct verdict for non-ambiguous
        if metrics['disambig_judge'][i] is not None:
            metrics['full_disambig_upperbound_judge'].append(metrics['disambig_judge'][i])
        else:
            metrics['full_disambig_upperbound_judge'].append(metrics['direct_judge'][i])

        # Anyref judge equivalent: disambig_anyref for ambiguous, direct_anyref for non-ambiguous
        if metrics['disambig_judge_anyref'][i] is not None:
            metrics['full_disambig_upperbound_judge_anyref'].append(metrics['disambig_judge_anyref'][i])
        else:
            metrics['full_disambig_upperbound_judge_anyref'].append(metrics['direct_judge_anyref'][i])

        # Anyref rouge equivalent: disambig_any for ambiguous, direct_anyref for non-ambiguous
        if metrics['disambig_answer_any'][i] is not None:
            metrics['full_disambig_upperbound_anyref'].append(metrics['disambig_answer_any'][i])
        else:
            metrics['full_disambig_upperbound_anyref'].append(metrics['direct_answer_anyref'][i])

    # Sanity check: "full coverage" metrics should have no Nones (they cover ambig + non-ambig).
    # Any Nones after filtering = parse errors in the underlying judge/rouge metric.
    # Skip rouge-based key when skip_rouge=True (all rouge values are intentionally None).
    n_total = len(output['direct'])
    has_judge = bool(output.get('direct_judge') or output.get('disambig_judge'))
    rouge_keys = [] if skip_rouge else ['full_disambig_upperbound', 'full_disambig_upperbound_anyref']
    judge_keys = ['full_disambig_upperbound_judge', 'full_disambig_upperbound_judge_anyref'] if has_judge else []
    for key in rouge_keys + judge_keys:
        n_valid = sum(1 for v in metrics[key] if v is not None)
        if n_valid != n_total:
            print(f"WARNING: {key} has {n_valid}/{n_total} non-None entries — {n_total - n_valid} likely parse errors")

    if warning_count / len(output['direct']) > 0.05:
        print(f"Warning: {warning_count / len(output['direct']) * 100:.1f}% of non-ambig questions have distinct references.")
    return metrics


def find_flips(before_scores, after_scores, threshold=0.0):
    """Return indices where score crossed threshold (wrong->right or right->wrong)."""
    pos, neg = [], []
    for i, (b, a) in enumerate(zip(before_scores, after_scores)):
        if b is None or a is None:
            continue
        if a - b >= threshold:
            pos.append((i, b, a))
        elif b - a >= threshold:
            neg.append((i, b, a))
    return {"pos": pos, "neg": neg}