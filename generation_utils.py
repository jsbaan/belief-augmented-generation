import datetime
import json
import os
from dataclasses import asdict
from typing import List, Dict, Tuple, Literal, Callable, Optional, Any
import logging

from tqdm import tqdm

from api_utils import get_api_client
from config import Config, build_output_fname
from dataloader import load_jsonl, load_ambigqa

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Latest open source models as of mid 2025
SUPPORTED_MODELS = {
    "tinyllama-1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "olmo2-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo2-13b-instruct": "allenai/OLMo-2-1124-13B-Instruct",
    "olmo2-32b-instruct": "allenai/OLMo-2-0325-32B-Instruct",
    "olmo3-7b-think": "allenai/Olmo-3-7B-Think",
    "olmo3-7b-instruct": "allenai/Olmo-3-7B-Instruct",
    "qwen3-8b-think": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b-think": "Qwen/Qwen3-14B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B-Instruct",
    "phi-4": "microsoft/phi-4",
    "ministral-3-14b": "mistralai/Ministral-3-14B-Instruct-2512",

    # Claude models
    "haiku3": "claude-3-haiku-20240307",
    "haiku4.5": "claude-haiku-4-5",

    # Google Gemini models
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3-flash": "gemini-3-flash",
}

def tile_prompts(prompts: List[List[Dict]], n_samples: int) -> Tuple[List[int], List[List[Dict]]]:
    """Tile prompts to generate n_samples samples per question."""
    tiled_idxs, tiled_prompts = zip(
        *[(q_idx, prompt) for (q_idx, prompt) in enumerate(prompts) for _ in range(n_samples)]
    )
    return tiled_idxs, tiled_prompts


def estimate_batch_size(model, max_input_tokens: int, max_new_tokens: int, safety_factor: Optional[float] = None) -> int:
    import torch
    if not torch.cuda.is_available():
        return 1
    if safety_factor is None:
        safety_factor = 0.70 if max_new_tokens > 512 else 0.85
    cfg = model.config
    try:
        num_layers   = cfg.num_hidden_layers
        hidden_size  = cfg.hidden_size
        num_heads    = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    except AttributeError:
        logger.warning("Model config missing architecture attributes; defaulting to batch_size=1")
        return 1

    head_dim = hidden_size // num_heads
    kv_bytes_per_seq = (max_input_tokens + max_new_tokens) * num_layers * num_kv_heads * head_dim * 2 * 2  # K+V, fp16
    prefill_bytes_per_seq = num_heads * max_input_tokens**2 * 2  # attention scores during prefill (one layer at a time), fp16
    memory_per_seq = kv_bytes_per_seq + prefill_bytes_per_seq
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    batch_size = max(1, int(safety_factor * free_bytes / memory_per_seq))
    logger.info(f"estimate_batch_size: free={free_bytes / 1e9:.1f}GB, kv_bytes_per_seq={kv_bytes_per_seq / 1e6:.1f}MB, "
                f"prefill_bytes_per_seq={prefill_bytes_per_seq / 1e6:.1f}MB, estimated batch_size={batch_size}")
    # Cap for long sequences: the C-level CUDA kernel for BitsAndBytes ops crashes with an
    # unrecoverable "invalid configuration argument" error at large batch_size * seq_len products.
    # This cannot be caught as a Python exception, so we prevent it upfront.
    # Empirically: batch_size=58 crashes at max_input_tokens=3400; 16 works fine.
    # Safe threshold: 16 * 3400 = 54400, so we use 50000 to be conservative.
    cuda_cap = max(1, 110000 // max_input_tokens)
    batch_size = min(batch_size, cuda_cap)
    logger.info(f"batch size after applying cuda cap to avoid BitsAndBytes error: {batch_size}")
    return batch_size


def generate_locally(
        model,
        tokenizer,
        prompts,
        n_samples: int,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
        batch_size: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        enable_thinking: Optional[bool] = None
) -> tuple[list[list[Any]], list[str], int]:
    import torch

    sampling_strategy = "greedy" if temperature == 0.0 else "sample"

    logger.info(f"Generating {n_samples} {sampling_strategy} for {len(prompts)} inputs")

    tiled_idxs, tiled_prompts = tile_prompts(prompts, n_samples)

    # Generate answers in batches
    generations = [[] for _ in range(len(prompts))]
    processed_prompts = [None for _ in range(len(prompts))]

    if batch_size is None:
        current_batch_size = estimate_batch_size(model, max_input_tokens, max_new_tokens)
    else:
        current_batch_size = batch_size

    with tqdm(total=len(tiled_prompts), desc=f"Generating {sampling_strategy}", unit="seq") as pbar:
        i = 0
        while i < len(tiled_prompts):
            batch_prompts = tiled_prompts[i: i + current_batch_size]
            batch_tiled_idxs = tiled_idxs[i: i + current_batch_size]

            # Batch tokenize
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = enable_thinking
            batch_prompts_formatted = tokenizer.apply_chat_template(batch_prompts, **chat_template_kwargs)
            inputs = tokenizer(
                batch_prompts_formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding=True,
                padding_side="left",
            ).to(model.device)

            # Generate tokens
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": sampling_strategy == "sample",
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }

                if sampling_strategy == "sample":
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = top_p
                    generation_kwargs["top_k"] = top_k
                    generation_kwargs["min_p"] = min_p

                try:
                    outputs = model.generate(**inputs, **generation_kwargs)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    del inputs
                    new_bs = max(1, int(current_batch_size * 0.85))
                    logger.warning(f"OOM at batch_size={current_batch_size}; retrying with batch_size={new_bs}")
                    if new_bs == current_batch_size:
                        raise
                    current_batch_size = new_bs
                    continue

            # Decode and distribute generations back to questions
            for j, (output, prompt_idx) in enumerate(zip(outputs, batch_tiled_idxs)):
                # Capture the processed (truncated) prompt by decoding the input
                if processed_prompts[prompt_idx] is None:
                    processed_prompt = tokenizer.decode(inputs["input_ids"][j], skip_special_tokens=True).strip()
                    processed_prompts[prompt_idx] = processed_prompt

                # Extract only the newly generated tokens (after the input)
                input_length = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_length:]
                # Decode only the generated part
                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                # Handle models that prefill <think> in the prompt (e.g., OLMo3-Think)
                if batch_prompts_formatted[j].rstrip().endswith('<think>'):
                    answer = '<think>' + answer
                generations[prompt_idx].append(answer)
            pbar.update(len(batch_prompts))
            i += current_batch_size

    return generations, processed_prompts, current_batch_size

def generate_api(
        model_name: str,
        prompts: List[List[Dict]],
        n_samples: int,
        max_new_tokens: int,
        api_processing: Literal["sync", "async"],
        temperature: float = None
) -> Tuple[List[List[str]], List[List[Dict]]]:

    model_name = SUPPORTED_MODELS[model_name]
    logger.info(f"Generating {len(prompts) * n_samples} responses using {model_name} API")

    generations = [[] for _ in range(len(prompts))]
    tiled_idxs, tiled_prompts = tile_prompts(prompts, n_samples)
    use_batch_api = api_processing == 'async'
    api_client = get_api_client(model_name, use_batch_api=use_batch_api)
    tiled_generations = api_client.generate(tiled_prompts, max_tokens=max_new_tokens, temperature=temperature)

    # Distribute generations back to questions
    for prompt_idx, gen in zip(tiled_idxs, tiled_generations):
        generations[prompt_idx].append(gen)

    # Unlike local models, the API never truncate, so we can just return the original prompts as message dicts
    return generations, prompts

def load_model_and_tokenizer(model_name: str, use_8bit: bool = True):
    """Load model and tokenizer with GPU support."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(SUPPORTED_MODELS.keys())}")

    model_id = SUPPORTED_MODELS[model_name]
    logger.info(f"Loading model: {model_id}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except ValueError as e:
        if "llama" in model_name:
            logger.warning(f"Failed to load fast tokenizer for {model_id}, falling back to slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=False, legacy=False, tokenizer_class="LlamaTokenizer"
            )
        else:
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading
    use_cuda = torch.cuda.is_available()
    model_kwargs = {}

    if use_cuda:
        # Use float16 when quantizing to match bitsandbytes requirements
        dtype = torch.float16 if use_8bit else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model_kwargs.update({"torch_dtype": dtype, "device_map": "auto"})

        if use_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # CPU configuration
        model_kwargs.update({"torch_dtype": torch.float32, "device_map": "cpu"})

        if use_8bit:
            logger.warning("8-bit quantization not supported on CPU, ignoring use_8bit flag")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    attn_implementation = getattr(model.config, '_attn_implementation', 'default')
    logger.info(f"Model {model_name} loaded successfully on device: {model.device} with attention implementation: {attn_implementation}")

    return model, tokenizer

def return_generation_fn(model_name: str):
    """
    Decide on local or API generation function based on model name that we will use later on in the generate_*.py files
     so we load a local model on the GPU only once for the entire pipeline (e.g., direct, clarify, and final).

    The returned generation function is low-level and takes a list of prompts and returns a list of generated answers.

    This is also the place where we decide on the actual generation parameters (hf/model/user), like temperature.
    """
    if model_name.startswith("haiku") or model_name.startswith("gemini"):
        # Create API generation function
        def generate_fn(prompts, cfg):
            # Manually unpack the parameters needed for API generation
            api_params = {
                "n_samples": cfg.n_samples,
                "max_new_tokens": cfg.max_new_tokens,
                "api_processing": cfg.api_processing,
                "temperature": cfg.temperature
            }
            generations, processed_prompts = generate_api(
                cfg.model_name,
                prompts,
                **api_params
            )
            return generations, processed_prompts, api_params
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, use_8bit=True)
        # Create local generation function bound to pre-loaded model
        def generate_fn(prompts, cfg):
            local_model_params = {
                "batch_size": cfg.batch_size,
                "n_samples": cfg.n_samples,
                "max_input_tokens": cfg.max_input_tokens,
                "max_new_tokens": cfg.max_new_tokens,
            }

            # Resolve temperature
            # Priority: user explicit > sampling strategy > model default
            if cfg.temperature is not None:
                local_model_params["temperature"] = cfg.temperature
            elif cfg.sampling == "unbiased":
                local_model_params["temperature"] = 1.0
            else:
                local_model_params["temperature"] = getattr(model.generation_config, "temperature", 1.0)

            # Add sampling parameters only if temperature is nonzero
            if local_model_params["temperature"] > 0:
                if cfg.sampling == "unbiased":
                    defaults = {"top_p": 1.0, "top_k": 0, "min_p": 0.0}
                else:
                    defaults = {
                        "top_p": getattr(model.generation_config, "top_p", 1.0),
                        "top_k": getattr(model.generation_config, "top_k", 0),
                        "min_p": getattr(model.generation_config, "min_p", 0.0),
                    }
                local_model_params["top_p"] = cfg.top_p if cfg.top_p is not None else defaults["top_p"]
                local_model_params["top_k"] = cfg.top_k if cfg.top_k is not None else defaults["top_k"]
                local_model_params["min_p"] = cfg.min_p if cfg.min_p is not None else defaults["min_p"]

            # Hardcoding Qwen3 settings for non-thinking mode (model default reflect recommended thinking mode settings)
            if 'qwen3' in cfg.model_name.lower():
                local_model_params['enable_thinking'] = 'think' in cfg.model_name.lower()
                # Manual override with recommended sampling settings for qwen3 non-thinking mode
                if (not local_model_params['enable_thinking']
                        and local_model_params["temperature"] > 0
                        and cfg.sampling != "unbiased"):
                    local_model_params.update({'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0})

            generations, processed_prompts, effective_bs = generate_locally(model, tokenizer, prompts, **local_model_params)
            local_model_params["batch_size"] = effective_bs  # replace None with actual value for logging
            return generations, processed_prompts, local_model_params

    return generate_fn

def run_generation_mode(generate_fn: Callable, config: Config, input_data=None):
    if input_data is None:
        if config.mode in ['direct', 'disambiguated']:
            # Load, shuffle, and limit ambigQA dataset
            input_data = load_ambigqa(config.input_path, config.max_examples, config.filter_conflicts, config.data_seed)
        else:
            # Load the output of the previous generation steps - no need to shuffle since they follow the above order
            input_data = load_jsonl(config.input_path)
            if config.max_examples:
                input_data = input_data[:config.max_examples]

    outputs = generate_fn(input_data, config)

    return save_output(outputs, config)

def save_output(outputs: List[Dict], config: Config):
    if config.output_fname:
        fname = config.output_fname.replace(".jsonl", "")
    else:
        fname = build_output_fname(
            mode=config.mode,
            assistant_model=config.assistant_model,
            direct_prompt=config.direct_prompt,
            belief_sampling=config.belief_sampling if config.belief_sampling is not None else config.sampling,
            seed=config.data_seed,
            reasoner_prompt=config.reasoner_prompt,
            user_model=config.user_model,
            final_prompt=config.final_prompt,
        )

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "timestamped"), exist_ok=True)

    def _write_atomic(fpath, data):
        tmp = fpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        os.replace(tmp, fpath)  # atomic on POSIX

    # Save with datetime ID
    today = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    output_fpath_timestamped = os.path.join(config.output_dir, "timestamped", f"{fname}_{today}.jsonl")
    _write_atomic(output_fpath_timestamped, outputs)

    # Overwrite the last file for simple access to the latest version
    output_fpath = os.path.join(config.output_dir, f"{fname}.jsonl")
    _write_atomic(output_fpath, outputs)

    logger.info(f"Results saved to: {output_fpath} and {output_fpath_timestamped}")
    return output_fpath