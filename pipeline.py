"""BAG pipeline entry point — runs the 9-step generation + evaluation pipeline.

Given an ambiguous question from AmbigNQ, the pipeline produces:

  1. direct                     — a direct generation `answer` using the LLM's recommended sampling settings, and a `belief_state`: K sampled answers
  2. disambiguated              — an `answer` per AmbigNQ-annotated disambiguated question, using recommended sampling settings
  3. reasoner (a.k.a. "clarify")— a `strategy` (answer/clarify/abstain) + `response`; SAG conditions on the question, BAG on step 1's `belief_state`
  4. user                       — a simulated user `response` to the clarification question, conditioned on a random AmbigNQ disambiguation
  5. final                      — the final `answer` after the user's reply; optionally with SAG+ or BAG+ including another round of K samples
  6. judge                      — reference-based LLM judge `verdict` (pass/fail) + `reasoning` for a pipeline branch (`--judge_branch <branch>`)
  7. claim_variation            — cluster step 1's K `belief_state` samples by main factual claim → `n_distinct_claims`, `claims`
  8. interpretation_variation   — classify each `belief_state` sample as contextualised / uncontextualised / clarifying / refusing
  9. claim_variation_final      — step 7 re-run on step 5's post-clarification `belief_state` (only with `--final_prompt belief`)

Each step reads the output(s) of earlier steps from `./data/generations{/<split>}/`
and writes its own `<step>_<model>_<...>.jsonl` back to the same directory
(filename layout defined in config.build_output_fname). Existing outputs are
skipped unless --force is passed.

Typical call:

    python pipeline.py \\
        --assistant_model qwen3-8b --user_model gemini-2.5-flash \\
        --judge_model gemini-2.5-flash --reasoner_prompt belief7 \\
        --direct_prompt vanilla --final_prompt prompt \\
        --belief_sampling unbiased --seed 50 --split dev \\
        --steps all

To call a single step (e.g. just re-run the judge on the final branch):

    python pipeline.py ... --steps judge --judge_branch final
"""

import argparse
import datetime
import logging
import os

from config import Config, build_output_fname
from generation_utils import run_generation_mode, return_generation_fn
from generate_direct_answer import generate_direct_answer, generate_direct_answer_disambiguated
from generate_clarify import generate_clarification
from generate_user_answer import generate_user_answer
from generate_final_answer import generate_final_answer
from generate_judge import generate_llm_judge_eval
from generate_faithfulness import generate_claim_variation_eval, generate_interpretation_variation_eval, _get_final_belief_samples
from dataloader import load_jsonl, load_ambigqa
from generation_utils import save_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEPS = ["direct", "disambiguated", "reasoner", "user", "final", "judge", "claim_variation", "interpretation_variation", "claim_variation_final", "all"]

def run_pipeline(
        assistant_model_name: str,
        user_model_name: str,
        reasoner_prompt: str,
        direct_prompt: str,
        max_examples: int,
        n_samples: int,
        belief_sampling: str,
        seed: int,
        test: bool,
        api_processing: str,
        steps: list[str] = None,
        data_dir: str = "./data",
        force: bool = False,
        no_greedy: bool = None,
        judge_model_name: str = None,
        judge_branch: str = None,
        anyref: bool = False,
        final_prompt: str = None,
        split: str = "train",
        filter_conflicts: bool = False,
):
    """Run the full generation pipeline."""
    if steps is None or "all" in steps:
        steps = STEPS

    # SAG+ uses prompt1 final with a prompt-based reasoner; BAG+ uses belief final with a belief* reasoner.
    # Mixing these (e.g. belief* reasoner + prompt1 final, or prompt reasoner + belief final) is invalid.
    if "final" in steps and final_prompt is not None:
        is_belief_reasoner = reasoner_prompt is not None and "belief" in reasoner_prompt
        if final_prompt == "belief" and not is_belief_reasoner:
            raise ValueError(f"final_prompt='belief' (BAG+) requires a belief* reasoner_prompt, got: {reasoner_prompt!r}")
        if final_prompt == "prompt1" and is_belief_reasoner:
            raise ValueError(f"final_prompt='prompt1' (SAG+) is incompatible with belief* reasoner_prompt, got: {reasoner_prompt!r}")

    split_subdir = "" if split == "train" else f"/{split}"
    output_dir  = f"{data_dir}/generations{split_subdir}{'/test' if test else ''}"
    input_dir   = f"{data_dir}/generations{split_subdir}"  # judge always reads from real outputs
    ambigqa_path = f"{data_dir}/input/ambignq_light/{split}_light.json"
    max_examples = max_examples if not test else 10
    print(datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S"))

    LOCAL_STEPS = {"direct", "disambiguated", "reasoner", "final"}
    needs_local = bool(set(steps) & LOCAL_STEPS)
    if needs_local:
        local_generate_fn = return_generation_fn(assistant_model_name)
    else:
        local_generate_fn = None
    if user_model_name:
        api_generate_fn = return_generation_fn(user_model_name)
    else:
        api_generate_fn = None
    # generate_fn selector: assistant steps use the local model unless it happens to match the API user model
    assistant_generate_fn = api_generate_fn if assistant_model_name == user_model_name else local_generate_fn

    if "direct" in steps:
        logger.info("Step 1/9: Generating direct answers...")
        _fname = build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed)
        _out = f"{output_dir}/{_fname}.jsonl"
        if not force and os.path.exists(_out):
            logger.info(f"Output already exists, skipping step: {_out}")
        else:
            direct_answer_config = Config(
                mode="direct",
                model_name=assistant_model_name,
                assistant_model=assistant_model_name,
                generate_fn=assistant_generate_fn,
                max_examples=max_examples,
                filter_conflicts=filter_conflicts,
                data_seed=seed,
                input_path=ambigqa_path,
                output_dir=output_dir,
                direct_prompt=direct_prompt,
                n_samples=n_samples,
                sampling=belief_sampling,
                belief_sampling=belief_sampling,
                no_greedy=no_greedy,
                max_input_tokens=128,
                max_new_tokens=4096 if 'think' in assistant_model_name else 300,
                api_processing=api_processing,
            )
            print(direct_answer_config)
            run_generation_mode(
                generate_direct_answer,
                direct_answer_config
            )

    if "disambiguated" in steps:
        logger.info("Step 2/9: Generating disambiguated answers...")
        _fname = build_output_fname("disambiguated", assistant_model_name, direct_prompt, belief_sampling, seed)
        _out = f"{output_dir}/{_fname}.jsonl"
        if not force and os.path.exists(_out):
            logger.info(f"Output already exists, skipping step: {_out}")
        else:
            disambig_config = Config(
                mode="disambiguated",
                model_name=assistant_model_name,
                assistant_model=assistant_model_name,
                generate_fn=assistant_generate_fn,
                max_examples=max_examples,
                filter_conflicts=filter_conflicts,
                data_seed=seed,
                input_path=ambigqa_path,
                output_dir=output_dir,
                direct_prompt=direct_prompt,
                n_samples=1,
                sampling="recommended",
                belief_sampling=belief_sampling,
                no_greedy=no_greedy,
                max_input_tokens=128,
                max_new_tokens=4096 if 'think' in assistant_model_name else 300,
                api_processing=api_processing,
            )
            print(disambig_config)
            run_generation_mode(
                generate_direct_answer_disambiguated,
                disambig_config
            )

    if "reasoner" in steps:
        # Naming note: the CLI step "reasoner", the Config mode "clarify", the file
        # generate_clarify.py, and the function generate_clarification all refer to
        # the same step — the strategy decision (answer / clarify / abstain). The
        # "clarify" name is baked into output filenames via build_output_fname.
        logger.info("Step 3/9: Reasoning over strategy...")
        direct_fname = build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed)
        if "belief" in reasoner_prompt:
            max_input_tokens = 3400
        else:
            max_input_tokens = 512

        _fname = build_output_fname("clarify", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt)
        _out = f"{output_dir}/{_fname}.jsonl"
        if not force and os.path.exists(_out):
            logger.info(f"Output already exists, skipping step: {_out}")
        else:
            clarification_config = Config(
                mode="clarify",
                reasoner_prompt=reasoner_prompt,
                model_name=assistant_model_name,
                assistant_model=assistant_model_name,
                generate_fn=assistant_generate_fn,
                input_path=f'{output_dir}/{direct_fname}.jsonl',
                direct_prompt=direct_prompt,
                data_seed=seed,
                output_dir=output_dir,
                sampling="recommended",
                belief_sampling=belief_sampling,
                max_new_tokens=1200 if reasoner_prompt in ("belief7", "belief8") else 512,
                max_input_tokens=max_input_tokens,
                api_processing=api_processing,
            )
            print(clarification_config)
            if reasoner_prompt == 'prompt':
                ambigqa_items = load_ambigqa(
                    ambigqa_path,
                    max_examples=max_examples,
                    filter_conflicts=filter_conflicts,
                    seed=seed,
                )
                run_generation_mode(generate_clarification, clarification_config, input_data=ambigqa_items)
            else:
                run_generation_mode(generate_clarification, clarification_config)

    if "user" in steps:
        logger.info("Step 4/9: Generating user answers...")
        reasoner_fname = build_output_fname("clarify", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt)
        _fname = build_output_fname("user", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt, user_model=user_model_name)
        _out = f"{output_dir}/{_fname}.jsonl"
        if not force and os.path.exists(_out):
            logger.info(f"Output already exists, skipping step: {_out}")
        else:
            user_answer_config = Config(
                mode="user",
                reasoner_prompt=reasoner_prompt,
                model_name=user_model_name,
                assistant_model=assistant_model_name,
                user_model=user_model_name,
                generate_fn=api_generate_fn,
                input_path=f'{output_dir}/{reasoner_fname}.jsonl',
                direct_prompt=direct_prompt,
                data_seed=seed,
                output_dir=output_dir,
                belief_sampling=belief_sampling,
                temperature=0.0,
                max_new_tokens=400,
                api_processing=api_processing,
            )
            print(user_answer_config)
            run_generation_mode(
                generate_user_answer,
                user_answer_config
            )

    if "final" in steps:
        logger.info(f"Step 5/9: Generating final answers (final_prompt={final_prompt})...")
        user_fname = build_output_fname("user", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt, user_model=user_model_name)
        # In test mode, read real user outputs (same as judge) but write to test dir
        final_input_dir = input_dir if test else output_dir
        user_path = f'{final_input_dir}/{user_fname}.jsonl'
        _fname = build_output_fname("final", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt, user_model=user_model_name, final_prompt=final_prompt)
        _out = f"{output_dir}/{_fname}.jsonl"
        if not force and os.path.exists(_out):
            logger.info(f"Output already exists, skipping step: {_out}")
        else:
            if final_prompt == "belief":
                final_n_samples = n_samples
                final_sampling = belief_sampling if belief_sampling else "recommended"
            else:
                final_n_samples = 1
                final_sampling = "recommended"

            final_answer_config = Config(
                mode="final",
                reasoner_prompt=reasoner_prompt,
                final_prompt=final_prompt,
                model_name=assistant_model_name,
                assistant_model=assistant_model_name,
                user_model=user_model_name,
                generate_fn=assistant_generate_fn,
                input_path=user_path,
                direct_prompt=direct_prompt,
                data_seed=seed,
                output_dir=output_dir,
                max_examples=max_examples,
                n_samples=final_n_samples,
                sampling=final_sampling,
                belief_sampling=belief_sampling,
                max_new_tokens=512,
                max_input_tokens=1024,
            )
            print(final_answer_config)
            run_generation_mode(
                generate_final_answer,
                final_answer_config
            )

    if "judge" in steps:
        logger.info(f"Step 6/9: Running LLM judge evaluation (branch={judge_branch})...")
        if not judge_model_name:
            raise ValueError("--judge_model is required for the judge step")
        if not judge_branch:
            raise ValueError("--judge_branch is required for the judge step")
        if judge_branch == "final" and not user_model_name:
            raise ValueError("Judge 'final' branch requires --user_model")
        if judge_branch == "reasoner" and not reasoner_prompt:
            raise ValueError("Judge 'reasoner' branch requires --reasoner_prompt")

        jb = "clarify" if judge_branch == "reasoner" else judge_branch

        src_fnames = {
            "direct":   lambda: build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed),
            "disambig": lambda: build_output_fname("disambiguated", assistant_model_name, direct_prompt, belief_sampling, seed),
            "belief":   lambda: build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed),
            "reasoner": lambda: build_output_fname("clarify", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt),
            "final":    lambda: build_output_fname("final", assistant_model_name, direct_prompt, belief_sampling, seed, reasoner_prompt=reasoner_prompt, user_model=user_model_name, final_prompt=final_prompt),
        }
        src_path = f"{input_dir}/{src_fnames[judge_branch]()}.jsonl"
        out_fname = build_output_fname("judge", assistant_model_name, direct_prompt, belief_sampling, seed,
                                       reasoner_prompt=reasoner_prompt if jb in ("clarify", "final") else None,
                                       judge_model=judge_model_name, judge_branch=jb,
                                       user_model=user_model_name if jb == "final" else None,
                                       final_prompt=final_prompt if jb == "final" else None)
        out_path = f"{output_dir}/{out_fname}.jsonl"

        judge_generate_fn = return_generation_fn(judge_model_name)

        if not force and os.path.exists(out_path):
            logger.info(f"Output already exists, skipping: {out_path}")
        elif not os.path.exists(src_path):
            logger.info(f"Judge: source file not found for branch '{judge_branch}': {src_path}")
        else:
            judge_config = Config(
                mode="judge",
                model_name=judge_model_name,
                assistant_model=assistant_model_name,
                user_model=judge_model_name,
                generate_fn=judge_generate_fn,
                reasoner_prompt=reasoner_prompt,
                direct_prompt=direct_prompt,
                belief_sampling=belief_sampling,
                data_seed=seed,
                input_path=src_path,
                output_dir=output_dir,
                output_fname=out_fname,
                temperature=0.0,
                api_processing=api_processing,
                max_new_tokens=150,
                max_input_tokens=1024,
            )
            items = load_jsonl(src_path)[:max_examples]
            print(judge_config)

            all_refs_lookup = None
            if anyref and jb in ("disambig", "final"):
                ambigqa = load_ambigqa(ambigqa_path, max_examples=max_examples, filter_conflicts=filter_conflicts, seed=seed)
                all_refs_lookup = {item['id']: item['references'] for item in ambigqa}

            outputs = generate_llm_judge_eval(items, jb, judge_config, anyref=anyref, all_refs_lookup=all_refs_lookup)
            save_output(outputs, judge_config)

    if "claim_variation" in steps:
        logger.info("Step 7/9: Running claim variation (belief-state claim diversity) evaluation...")
        if not judge_model_name:
            raise ValueError("--judge_model is required for the claim_variation step")

        src_fname = build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed)
        src_path = f"{input_dir}/{src_fname}.jsonl"
        out_fname = build_output_fname(
            "claim_variation", assistant_model_name, direct_prompt, belief_sampling, seed,
            judge_model=judge_model_name,
        )
        out_path = f"{output_dir}/{out_fname}.jsonl"

        if not force and os.path.exists(out_path):
            logger.info(f"Output already exists, skipping: {out_path}")
        elif not os.path.exists(src_path):
            logger.info(f"Claim variation: source file not found: {src_path}")
        else:
            cv_generate_fn = return_generation_fn(judge_model_name)
            cv_config = Config(
                mode="claim_variation",
                model_name=judge_model_name,
                assistant_model=assistant_model_name,
                user_model=judge_model_name,
                generate_fn=cv_generate_fn,
                direct_prompt=direct_prompt,
                belief_sampling=belief_sampling,
                data_seed=seed,
                input_path=src_path,
                output_dir=output_dir,
                output_fname=out_fname,
                temperature=0.0,
                api_processing=api_processing,
                max_new_tokens=600,
                max_input_tokens=4096,
            )
            items = load_jsonl(src_path)[:max_examples]
            print(cv_config)
            outputs = generate_claim_variation_eval(items, cv_config)
            save_output(outputs, cv_config)

    if "interpretation_variation" in steps:
        logger.info("Step 8/9: Running interpretation variation (belief-state scope diversity) evaluation...")
        if not judge_model_name:
            raise ValueError("--judge_model is required for the interpretation_variation step")

        src_fname = build_output_fname("direct", assistant_model_name, direct_prompt, belief_sampling, seed)
        src_path = f"{input_dir}/{src_fname}.jsonl"
        out_fname = build_output_fname(
            "interpretation_variation", assistant_model_name, direct_prompt, belief_sampling, seed,
            judge_model=judge_model_name,
        )
        out_path = f"{output_dir}/{out_fname}.jsonl"

        if not force and os.path.exists(out_path):
            logger.info(f"Output already exists, skipping: {out_path}")
        elif not os.path.exists(src_path):
            logger.info(f"Interpretation variation: source file not found: {src_path}")
        else:
            iv_generate_fn = return_generation_fn(judge_model_name)
            iv_config = Config(
                mode="interpretation_variation",
                model_name=judge_model_name,
                assistant_model=assistant_model_name,
                user_model=judge_model_name,
                generate_fn=iv_generate_fn,
                direct_prompt=direct_prompt,
                belief_sampling=belief_sampling,
                data_seed=seed,
                input_path=src_path,
                output_dir=output_dir,
                output_fname=out_fname,
                temperature=0.0,
                api_processing=api_processing,
                max_new_tokens=600,
                max_input_tokens=4096,
            )
            items = load_jsonl(src_path)[:max_examples]
            print(iv_config)
            outputs = generate_interpretation_variation_eval(items, iv_config)
            save_output(outputs, iv_config)

    if "claim_variation_final" in steps:
        logger.info("Step 9/9: Running claim variation final (post-clarification belief diversity)...")
        if not judge_model_name:
            raise ValueError("--judge_model is required for the claim_variation_final step")
        if not user_model_name:
            raise ValueError("--user_model is required for the claim_variation_final step")
        if not reasoner_prompt or "belief" not in reasoner_prompt:
            logger.info("claim_variation_final requires a belief reasoner_prompt; skipping")
        else:
            final_fname = build_output_fname(
                "final", assistant_model_name, direct_prompt, belief_sampling, seed,
                reasoner_prompt=reasoner_prompt, user_model=user_model_name, final_prompt="belief",
            )
            src_path = f"{input_dir}/{final_fname}.jsonl"
            out_fname = build_output_fname(
                "claim_variation_final", assistant_model_name, direct_prompt, belief_sampling, seed,
                judge_model=judge_model_name, reasoner_prompt=reasoner_prompt, user_model=user_model_name,
            )
            out_path = f"{output_dir}/{out_fname}.jsonl"

            if not force and os.path.exists(out_path):
                logger.info(f"Output already exists, skipping: {out_path}")
            elif not os.path.exists(src_path):
                logger.info(f"Claim variation final: source file not found: {src_path}")
            else:
                cvf_generate_fn = return_generation_fn(judge_model_name)
                cvf_config = Config(
                    mode="claim_variation_final",
                    model_name=judge_model_name,
                    assistant_model=assistant_model_name,
                    user_model=user_model_name,
                    generate_fn=cvf_generate_fn,
                    reasoner_prompt=reasoner_prompt,
                    direct_prompt=direct_prompt,
                    belief_sampling=belief_sampling,
                    data_seed=seed,
                    input_path=src_path,
                    output_dir=output_dir,
                    output_fname=out_fname,
                    temperature=0.0,
                    api_processing=api_processing,
                    max_new_tokens=600,
                    max_input_tokens=4096,
                )
                items = load_jsonl(src_path)[:max_examples]
                print(cvf_config)
                outputs = generate_claim_variation_eval(items, cvf_config,
                                                        sample_extractor=_get_final_belief_samples)
                save_output(outputs, cvf_config)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_examples", type=int, default=1000)
    argparser.add_argument("--n_samples", type=int, default=10)
    argparser.add_argument("--assistant_model", type=str)
    argparser.add_argument("--user_model", type=str)
    argparser.add_argument("--judge_model", type=str, help="API model for LLM judge evaluation (e.g. gemini-2.5-flash)")
    argparser.add_argument("--api_processing", type=str, choices=["sync", "async"], default="sync")
    argparser.add_argument("--belief_sampling", type=str, choices=['unbiased', 'recommended'])
    argparser.add_argument("--reasoner_prompt", choices=["belief", "belief1", "belief2", "belief3", "belief4", "belief5", "belief6", "belief7", "belief8", "belief9", "prompt"])
    argparser.add_argument("--direct_prompt", choices=["vanilla", "concise", "cot", "sentence"])
    argparser.add_argument("--seed", type=int)
    argparser.add_argument('--test', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--no_greedy', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--steps", nargs="+",
        choices=STEPS,
        default=["all"])
    argparser.add_argument('--force', action='store_true', default=False,
        help='Re-run steps even if output files already exist')
    argparser.add_argument("--final_prompt", choices=["prompt", "prompt1", "belief"], required=False, default=None,
        help="Final answer prompt variant. prompt=forced answer, prompt1=prompt-only abstain baseline, belief=BAG with consensus reasoner")
    argparser.add_argument("--judge_branch",
        choices=["direct", "disambig", "belief", "reasoner", "final"],
        default=None,
        help="Which branch to evaluate with the LLM judge. E.g. --judge_branch final")
    argparser.add_argument("--anyref", action="store_true", default=False,
        help="Also evaluate against all reference groups (anyref), in addition to a single random ref")
    argparser.add_argument("--split", choices=["train", "dev"], default="train",
        help="Dataset split to run on (train_light.json or dev_light.json)")
    argparser.add_argument("--filter_conflicts", action="store_true", default=False,
        help="Filter out examples where annotations have conflicting types")

    args = argparser.parse_args()

    # Run the full pipeline
    run_pipeline(
        assistant_model_name=args.assistant_model,
        user_model_name=args.user_model,
        reasoner_prompt=args.reasoner_prompt,
        direct_prompt=args.direct_prompt,
        max_examples=args.max_examples,
        n_samples=args.n_samples,
        belief_sampling=args.belief_sampling,
        seed=args.seed,
        test=args.test,
        steps=args.steps,
        force=args.force,
        api_processing=args.api_processing,
        no_greedy=args.no_greedy,
        judge_model_name=args.judge_model,
        judge_branch=args.judge_branch,
        anyref=args.anyref,
        final_prompt=args.final_prompt,
        split=args.split,
        filter_conflicts=args.filter_conflicts,
    )
