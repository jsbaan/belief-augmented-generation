"""
Generic API client utilities for LLM generation.
Supports multiple API providers with both sync and batch processing.
"""
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional
import anthropic
from google import genai
from google.genai import types
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Claude client for LLM generation."""

    def __init__(self, model_name: str = "claude-3-haiku-20240307", use_batch_api: bool = True, batch_threshold: int = 50):
        api_key_path = Path.home() / ".secrets" / "anthropic_api_key"
        api_key = api_key_path.read_text().strip()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.use_batch_api = use_batch_api
        self.batch_threshold = batch_threshold
        logger.info(f"Initialized Anthropic client with model: {model_name}, batch_api: {use_batch_api}")

    def generate(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Generate responses using Anthropic API."""
        # Use batch API for large batches, sync for small batches
        if self.use_batch_api and len(prompts) >= self.batch_threshold:
            return self._generate_batch_async(prompts, max_tokens, temperature)
        else:
            return self._generate_batch_sync(prompts, max_tokens, temperature)

    def _generate_batch_sync(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Synchronous generation."""
        responses = []
        logger.info(f"Generating {len(prompts)} responses with Anthropic API (sync)")

        for i, messages in enumerate(tqdm(prompts, desc="API generation")):
            try:
                kwargs = dict(model=self.model_name, max_tokens=max_tokens, messages=messages)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = self.client.messages.create(**kwargs)
                responses.append(response.content[0].text)
            except Exception as e:
                logger.error(f"Error generating response {i}: {e}")
                responses.append("error")  # Fallback response

        return responses

    def _generate_batch_async(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Asynchronous batch generation using Anthropic's batch API."""
        logger.info(f"Generating {len(prompts)} responses with Anthropic Batch API")

        # Prepare batch requests
        batch_requests = []
        for i, messages in enumerate(prompts):
            params = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
            if temperature is not None:
                params["temperature"] = temperature
            batch_requests.append({"custom_id": str(i), "params": params})

        # Submit batch
        batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        logger.info(f"Submitted batch {batch_id}, waiting for completion...")

        # Poll for completion
        with tqdm(desc="Waiting for batch completion", unit="poll") as pbar:
            while True:
                batch_status = self.client.messages.batches.retrieve(batch_id)

                if batch_status.processing_status == "ended":
                    pbar.set_description("Batch ended, downloading results")
                    break
                elif batch_status.processing_status in ["canceling", "canceled"]:
                    raise Exception(f"Batch {batch_id} was canceled during processing")

                # Update progress bar with batch status
                if hasattr(batch_status, 'request_counts'):
                    succeeded = getattr(batch_status.request_counts, 'succeeded', 0)
                    errored = getattr(batch_status.request_counts, 'errored', 0)
                    processing = getattr(batch_status.request_counts, 'processing', 0)
                    total = len(prompts)
                    completed = succeeded + errored
                    pbar.set_postfix(completed=f"{completed}/{total}", processing=processing)

                pbar.update(1)
                time.sleep(10)  # Poll every 10 seconds

        # Download results using the API client
        batch_results = self.client.messages.batches.results(batch_id)

        # Parse results back into list format, maintaining order
        responses = ["error"] * len(prompts)  # Initialize with fallback

        for result_entry in batch_results:
            custom_id = int(result_entry.custom_id)
            result = result_entry.result

            if result.type == 'succeeded':
                # Successful response
                message = result.message
                if message and message.content and len(message.content) > 0:
                    responses[custom_id] = message.content[0].text
            elif result.type == 'errored':
                # Error response - raise exception instead of logging
                raise Exception(f"Batch request {custom_id} failed: {result.error}")
            elif result.type in ['canceled', 'expired']:
                raise Exception(f"Batch request {custom_id} was {result.type}")

        logger.info(f"Batch processing completed for {batch_id}")
        return responses


class GeminiClient:
    """Google Gemini client for LLM generation."""

    def __init__(self, model_name: str = "gemini-2.5-flash", use_batch_api: bool = True, batch_threshold: int = 500):
        api_key_path = Path.home() / ".secrets" / "google_api_key"
        api_key = api_key_path.read_text().strip()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.use_batch_api = use_batch_api
        self.batch_threshold = batch_threshold
        logger.info(f"Initialized Gemini client with model: {model_name}, batch_api: {use_batch_api}")

    @staticmethod
    def _convert_messages(messages: List[Dict]) -> List[Dict]:
        """Convert Claude-format messages to Gemini format.

        Claude:  {"role": "user"/"assistant", "content": "text"}
        Gemini:  {"role": "user"/"model", "parts": [{"text": "text"}]}
        """
        converted = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            content = msg.get("content", "")
            converted.append({
                "role": role,
                "parts": [{"text": content}]
            })
        return converted

    def generate(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Generate responses using Gemini API."""
        if self.use_batch_api and len(prompts) >= self.batch_threshold:
            return self._generate_batch_async(prompts, max_tokens, temperature)
        else:
            return self._generate_batch_sync(prompts, max_tokens, temperature)

    @staticmethod
    def _retry(fn, label, max_attempts=5, initial_delay=60):
        """Retry fn on 429/503, raising RuntimeError after max_attempts."""
        delay = initial_delay
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                if code in (429, 500, 502, 503, 504):
                    jitter = random.uniform(0, delay * 0.2)
                    details = getattr(e, 'message', None) or str(e)
                    logger.warning(
                        f"{label} got {code}, retrying in {delay:.0f}s "
                        f"(attempt {attempt+1}/{max_attempts}) | detail: {details}"
                    )
                    time.sleep(delay + jitter)
                    delay = min(delay * 2, 600)
                else:
                    raise
        raise RuntimeError(f"{label} failed after {max_attempts} retries")

    def _generate_batch_sync(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Synchronous generation."""
        responses = []
        logger.info(f"Generating {len(prompts)} responses with Gemini API (sync)")

        for i, messages in enumerate(tqdm(prompts, desc="Gemini API generation")):
            gemini_messages = self._convert_messages(messages)
            response = self._retry(
                lambda msgs=gemini_messages: self.client.models.generate_content(
                    model=self.model_name,
                    contents=msgs,
                    config=types.GenerateContentConfig(
                        **({"temperature": temperature} if temperature is not None else {}),
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(mode="NONE")
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )
                ),
                label=f"sync response {i}",
                initial_delay=15,
            )
            text = response.text
            if text is None:
                logger.warning(
                    f"Gemini sync response {i} was filtered/empty (safety block?), using empty string. "
                    f"Prompt: {messages}"
                )
                text = ""
            responses.append(text)

        return responses

    def _generate_batch_async(
        self,
        prompts: List[List[Dict]],
        max_tokens: int = 100,
        temperature: float = None,
    ) -> List[str]:
        """Batch generation using Gemini's batch API with inline requests."""
        logger.info(f"Generating {len(prompts)} responses with Gemini Batch API")

        # Prepare inline requests
        inline_requests = []
        for messages in prompts:
            gemini_messages = self._convert_messages(messages)
            inline_requests.append({
                "contents": gemini_messages,
                "config": {
                    **({"temperature": temperature} if temperature is not None else {}),
                    "max_output_tokens": max_tokens,
                    "thinking_config": {"thinking_budget": 0},
                    "tool_config": {
                        "function_calling_config": {"mode": "NONE"}
                    },
                }
            })

        # Submit batch job (batch API requires models/ prefix)
        batch_job = self._retry(
            lambda: self.client.batches.create(
                model=f"models/{self.model_name}",
                src=inline_requests,
                config={"display_name": f"batch-{self.model_name}-{len(prompts)}"},
            ),
            "batches.create"
        )
        logger.info(f"Submitted Gemini batch {batch_job.name}, waiting for completion...")

        # Poll for completion
        completed_states = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"
        }
        with tqdm(desc="Waiting for Gemini batch completion", unit="poll") as pbar:
            while batch_job.state.name not in completed_states:
                pbar.update(1)
                time.sleep(30)  # Gemini recommended polling interval
                batch_job = self._retry(
                    lambda: self.client.batches.get(name=batch_job.name),
                    "batches.get"
                )

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise Exception(f"Gemini batch job {batch_job.name} ended with state: {batch_job.state.name}")

        # Extract results - inline responses preserve input order
        responses = []
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            if inline_response.error:
                logger.warning(f"Gemini batch request {i} failed, using empty string: {inline_response.error}")
                responses.append("")
            else:
                responses.append(inline_response.response.text or "")

        logger.info(f"Gemini batch processing completed for {batch_job.name}")
        return responses

def get_api_client(model_name: str, use_batch_api: bool = True, **kwargs):
    """Factory function to get the appropriate API client.

    Args:
        model_name: Name of the model (e.g., "claude-3-haiku-20240307")
        use_batch_api: Whether to use batch API when available
        **kwargs: Additional arguments passed to the client constructor
    """
    if model_name.startswith("claude-"):
        return AnthropicClient(model_name, use_batch_api=use_batch_api, **kwargs)
    elif model_name.startswith("gemini-"):
        return GeminiClient(model_name, use_batch_api=use_batch_api, **kwargs)
    else:
        raise ValueError(f"Unsupported API model: {model_name}")
