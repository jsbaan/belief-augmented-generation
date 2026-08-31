"""Configuration for generation experiments."""

from dataclasses import dataclass, asdict
from typing import Optional, Callable


# Valid generation modes
VALID_MODES = ["direct", "clarify", "user", "final", "disambiguated", "judge", "claim_variation", "interpretation_variation", "claim_variation_final"]
VALID_REASONER_PROMPTS = ["prompt", "belief", "belief1", "belief2", "belief3", "belief4", "belief5", "belief6", "belief7", "belief8", "belief9"]
VALID_FINAL_PROMPTS = ["prompt", "prompt1", "belief"]


@dataclass
class Config:
    """Configuration for a generation mode."""

    # Model settings
    model_name: str
    mode: str

    # Input/output
    input_path: str
    output_dir: str
    output_fname: Optional[str] = None
    max_examples: Optional[int] = None
    filter_conflicts: bool = False
    data_seed: int = 42

    # Prompt settings
    direct_prompt: Optional[str] = None
    reasoner_prompt: Optional[str] = None
    final_prompt: Optional[str] = None

    # Generation parameters
    no_greedy: Optional[bool] = None
    sampling: Optional[str] = None        # controls actual sampling params (temperature, top_p, etc.)
    belief_sampling: Optional[str] = None # identifies which belief state version; used for filenames in downstream steps
    temperature: Optional[float] = None # takes hf/model default
    top_p: Optional[float] = None # takes hf/model default
    top_k: Optional[int] = None # takes hf/model default
    min_p: Optional[float] = None # takes hf/model default

    n_samples: int = 1
    max_new_tokens: Optional[int] = None

    # Local models only
    use_8bit: Optional[bool] = True # for now this is actually always set to True in `return_generation_fn`
    batch_size: Optional[int] = None
    max_input_tokens: Optional[int] = None

    # API only
    api_processing: Optional[str] = "sync"

    # Role names for filename construction
    assistant_model: Optional[str] = None
    user_model: Optional[str] = None

    # Generation function (generation.utils.generate_api or generation_utils.generate_locally)
    generate_fn: Optional[Callable] = None

    def __post_init__(self):
        """Validate mode after initialization."""
        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode!r}. "f"Must be one of {VALID_MODES}")
        if self.reasoner_prompt is not None and self.reasoner_prompt not in VALID_REASONER_PROMPTS:
            raise ValueError(f"Invalid reasoner_prompt: {self.reasoner_prompt!r}. "f"Must be one of {VALID_REASONER_PROMPTS}")
        if self.final_prompt is not None and self.final_prompt not in VALID_FINAL_PROMPTS:
            raise ValueError(f"Invalid final_prompt: {self.final_prompt!r}. "f"Must be one of {VALID_FINAL_PROMPTS}")


    def to_dict(self, resolved_params: Optional[dict] = None) -> dict:
        """Convert config to dict, replacing generate_fn with its name.

        Args:
            resolved_params: Optional dict with actual generation parameter values used
        """
        config_dict = asdict(self)
        if self.generate_fn is not None:
            config_dict['generate_fn'] = self.generate_fn.__name__

        # Add resolved params if provided
        if resolved_params is not None:
            config_dict['resolved_generation_params'] = resolved_params

        return config_dict


def build_output_fname(mode, assistant_model, direct_prompt, belief_sampling, seed, reasoner_prompt=None, user_model=None, judge_model=None, judge_branch=None, final_prompt=None):
    """Build output filename stem for a pipeline step."""
    direct_prompt_key = "" if (not direct_prompt or direct_prompt == "vanilla") else f"_{direct_prompt}"
    sampling_key = "" if (belief_sampling is None or belief_sampling == "unbiased") else f"_{belief_sampling}"

    if mode in ("direct", "disambiguated"):
        return f"{mode}_{assistant_model}{direct_prompt_key}{sampling_key}_{seed}"

    # Downstream steps: only carry these keys when reasoner builds on belief state
    downstream_sampling_key = sampling_key if (reasoner_prompt and "belief" in reasoner_prompt) else ""
    downstream_direct_prompt_key = direct_prompt_key if (reasoner_prompt and "belief" in reasoner_prompt) else ""

    final_prompt_key = f"_{final_prompt}" if final_prompt else ""

    if mode == "clarify":
        if not reasoner_prompt:
            raise ValueError(f"Loading/saving reasoner output: reasoner_prompt cannot be None")
        return f"{mode}_{assistant_model}{downstream_direct_prompt_key}{downstream_sampling_key}_{seed}_{reasoner_prompt}"
    elif mode == "user":
        if not reasoner_prompt or not user_model:
            raise ValueError(f"Loading/saving {mode} output: reasoner_prompt or user_model cannot be None")
        return f"{mode}_{assistant_model}_{user_model}{downstream_direct_prompt_key}{downstream_sampling_key}_{seed}_{reasoner_prompt}"
    elif mode == "final":
        if not reasoner_prompt or not user_model:
            raise ValueError(f"Loading/saving {mode} output: reasoner_prompt or user_model cannot be None")
        return f"{mode}_{assistant_model}_{user_model}{downstream_direct_prompt_key}{downstream_sampling_key}_{seed}_{reasoner_prompt}{final_prompt_key}"
    elif mode == "judge":
        if not judge_model:
            raise ValueError("Loading/saving judge output: judge_model cannot be None")
        rp_key = f"_{reasoner_prompt}" if reasoner_prompt else ""
        if judge_branch is not None:
            reasoner_prompt_suffix = rp_key if judge_branch in ("clarify", "final") else ""
            # direct/disambig/belief judge branches correspond to direct pipeline steps,
            # so they carry direct_prompt/sampling unconditionally (not downstream-filtered)
            if judge_branch in ("direct", "disambig", "belief"):
                branch_direct_prompt_key = direct_prompt_key
                branch_sampling_key = sampling_key
            else:
                branch_direct_prompt_key = downstream_direct_prompt_key
                branch_sampling_key = downstream_sampling_key
            if judge_branch == "final":
                if not user_model:
                    raise ValueError("Loading/saving judge_final output: user_model cannot be None")
                return f"judge_final_{assistant_model}_{judge_model}_{user_model}{branch_direct_prompt_key}{branch_sampling_key}_{seed}{reasoner_prompt_suffix}{final_prompt_key}"
            return f"judge_{judge_branch}_{assistant_model}_{judge_model}{branch_direct_prompt_key}{branch_sampling_key}_{seed}{reasoner_prompt_suffix}"
        return f"{mode}_{assistant_model}_{judge_model}{direct_prompt_key}{sampling_key}_{seed}{rp_key}"
    elif mode == "claim_variation":
        if not judge_model:
            raise ValueError("Loading/saving claim_variation output: judge_model cannot be None")
        return f"claim_variation_{assistant_model}_{judge_model}{direct_prompt_key}{sampling_key}_{seed}"
    elif mode == "interpretation_variation":
        if not judge_model:
            raise ValueError("Loading/saving interpretation_variation output: judge_model cannot be None")
        return f"interpretation_variation_{assistant_model}_{judge_model}{direct_prompt_key}{sampling_key}_{seed}"
    elif mode == "claim_variation_final":
        if not judge_model or not user_model or not reasoner_prompt:
            raise ValueError("claim_variation_final requires judge_model, user_model, reasoner_prompt")
        return (
            f"claim_variation_final_{assistant_model}_{user_model}_{judge_model}"
            f"{downstream_direct_prompt_key}{downstream_sampling_key}_{seed}_{reasoner_prompt}"
        )
