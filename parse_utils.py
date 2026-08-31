import re
from typing import Optional, Dict


def parse_direct_response(response: str) -> Dict:
    # Handle thinking-model outputs: <think>...</think> followed by the actual answer.
    # If </think> is missing the model was cut off mid-thought and no answer was produced.
    if response.startswith('<think>'):
        think_end = response.find('</think>')
        if think_end != -1:
            reasoning = response[len('<think>'):think_end].strip()
            raw_response = response[think_end + len('</think>'):].strip() or None
        else:
            reasoning = response[len('<think>'):].strip()
            raw_response = None
        return {
            "reasoning": reasoning,
            "response": None,
            "raw_response": raw_response,
        }

    reasoning_match = re.search(r"reasoning:\s*(.+?)(?=final answer:)", response.lower(), re.DOTALL)
    answer_match = re.search(r"final answer:\s*(.+?)(?:\n|$)", response.lower(), re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    return {
        "reasoning": reasoning,
        "response": answer,
        "raw_response": response
    }

def parse_clarification_response(response: str) -> Optional[Dict[str, str]]:
    """Parse LLM response to extract strategy, reasoning, and user-facing content."""

    if isinstance(response, list):
        if len(response) > 1:
            raise ValueError("Expecting single response.")
        else:
            response = response[0]
    response = response.strip()

    strategy_match = None
    reasoning_match = None
    content_match = None
    summary_match = None

    lines = response.split('\n')

    # When first line is not prefixed with "STRATEGY" but just contains a strategy name
    if lines and lines[0].strip().upper() in ["CLARIFICATION_QUESTION", "CLARIFICATION QUESTION", "CLARIFICATION", "DIRECT_ANSWER", "DIRECT ANSWER", "ABSTAIN"]:
        first_line = lines[0].strip().upper()
        strategy_match = first_line.lower().replace(" ", "_")
        # Normalize bare "CLARIFICATION" to "clarification_question"
        if strategy_match == "clarification":
            strategy_match = "clarification_question"
        start_idx = 1
    else:
        start_idx = 0

    # Handle "LABEL: content" format (e.g. "CLARIFICATION_QUESTION: Could you...")
    if not strategy_match and lines:
        first_upper = lines[0].strip().upper()
        for label, normalized in [
            ("CLARIFICATION_QUESTION:", "clarification_question"),
            ("CLARIFICATION:", "clarification_question"),
            ("DIRECT_ANSWER:", "direct_answer"),
            ("ABSTAIN:", "abstain"),
        ]:
            if first_upper.startswith(label):
                strategy_match = normalized
                content_after = lines[0].strip()[len(label):].strip()
                if content_after:
                    content_match = content_after
                    if len(lines) > 1:
                        content_match += "\n" + "\n".join(lines[1:])
                start_idx = len(lines)  # skip loop since we consumed everything
                break

    # Parse all lines for STRATEGY:, REASONING:, and FINAL RESPONSE:
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        line_upper = line.upper()
        if line_upper.startswith("STRATEGY:") and not strategy_match:
            strategy_text = line.split(":", 1)[1].strip().lower()
            # Some models (e.g. olmo2-13b, qwen3-14b) put the value on the next line:
            #   STRATEGY:
            #   DIRECT_ANSWER
            # Peek at the next line only when the same-line value is empty AND the next
            # line is a bare strategy keyword (to avoid false positives from REASONING
            # lines that mention strategy names in passing).
            if not strategy_text and idx + 1 < len(lines):
                next_stripped = lines[idx + 1].strip().upper().rstrip(".")
                if next_stripped in ("DIRECT_ANSWER", "CLARIFICATION_QUESTION", "ABSTAIN"):
                    strategy_text = lines[idx + 1].strip().lower()
            if "direct" in strategy_text:
                strategy_match = "direct_answer"
            elif "clarification" in strategy_text:
                strategy_match = "clarification_question"
            elif "abstain" in strategy_text:
                strategy_match = "abstain"
            # else:
            #     strategy_match = f"Freeform: {strategy_text}"
        elif line_upper.startswith("REASONING:"):
            reasoning_match = line.split(":", 1)[1].strip()
        elif line_upper.startswith("SUMMARY:"):
            summary_match = line.split(":", 1)[1].strip()
        elif line_upper.startswith("RESPONSE:") or line_upper.startswith("FINAL RESPONSE:") or line_upper.startswith("FINAL ANSWER:"):
            for prefix in ["FINAL RESPONSE:", "FINAL ANSWER:", "RESPONSE:"]:
                if line_upper.startswith(prefix):
                    content_match = line[len(prefix):].strip()
                    if idx + 1 < len(lines):
                        remaining_lines = lines[idx + 1:]
                        if remaining_lines:
                            content_match += "\n" + "\n".join(remaining_lines)
                    break
            break

    return {
        "strategy": strategy_match,
        "reasoning": reasoning_match,
        "summary": summary_match,
        "response": content_match,
        "raw_response": response
    }


def parse_llm_judge_eval_response(raw: str) -> Dict:
    """Parse LLM judge eval response (REASONING / VERDICT format).

    Returns a dict with:
        verdict:      1 = correct, 0 = incorrect, -1 = unparseable
        reasoning:    the model's one-sentence explanation (or None)
        raw_response: the full raw string
    """
    if not raw:
        return {'verdict': -1, 'reasoning': None, 'raw_response': raw}

    reasoning = None
    verdict = -1

    for line in raw.strip().split('\n'):
        line = line.strip()
        upper = line.upper()
        if upper.startswith('REASONING:'):
            reasoning = line.split(':', 1)[1].strip()
        elif upper.startswith('VERDICT:'):
            v = line.split(':', 1)[1].strip().lower()
            if v.startswith('yes'):
                verdict = 1
            elif v.startswith('no'):
                verdict = 0

    # Fallback: check the last non-empty line if the structured format was not followed
    if verdict == -1:
        last = next((l.strip().lower() for l in reversed(raw.strip().split('\n')) if l.strip()), '')
        if last in ('yes', 'yes.') or last.startswith('yes,') or last.startswith('yes '):
            verdict = 1
        elif last in ('no', 'no.') or last.startswith('no,') or last.startswith('no '):
            verdict = 0

    return {'verdict': verdict, 'reasoning': reasoning, 'raw_response': raw}


def parse_claim_variation_response(raw: str) -> Dict:
    """Parse faithfulness prompt output (N_DISTINCT_ANSWERS / ANSWERS format).

    Returns:
        n_distinct_claims: int (>=0), or -1 if unparseable — counts only type=content clusters
        n_clarifying:      int — total samples in type=clarify clusters
        n_refusing:        int — total samples in type=refuse clusters
        claims:            list of {label, representative, n_samples, sample_indices, type}
        raw_response:      str
    """
    if not raw:
        return {'n_distinct_claims': -1, 'n_clarifying': 0, 'n_refusing': 0, 'claims': [], 'raw_response': raw}

    n_distinct_claims = -1
    claims = []
    in_claims_block = False
    claim_line_re = re.compile(
        r'^([A-Z])\s*\(n=(\d+),\s*indices=([0-9,\s]*),\s*type=(content|clarify|refuse)\):\s*(.+)$'
    )
    # Fallback: format without type field
    claim_line_re_legacy = re.compile(
        r'^([A-Z])\s*\(n=(\d+),\s*indices=([0-9,\s]*)\):\s*(.+)$'
    )

    for line in raw.strip().split('\n'):
        line = line.strip()
        upper = line.upper()
        if upper.startswith('N_DISTINCT_ANSWERS:') or upper.startswith('N_DISTINCT_CLAIMS:'):
            try:
                n_distinct_claims = int(line.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif upper.startswith('ANSWERS:') or upper.startswith('CLAIMS:'):
            in_claims_block = True
        elif in_claims_block and line:
            m = claim_line_re.match(line)
            if m:
                label, n_str, indices_str, claim_type, rep = m.groups()
            else:
                m = claim_line_re_legacy.match(line)
                if m:
                    label, n_str, indices_str, rep = m.groups()
                    rep_lower = rep.strip().lower()
                    if rep_lower in ('clarify', 'clarification', 'asks for clarification',
                                     'ask clarification', 'asks clarification'):
                        claim_type = 'clarify'
                    elif rep_lower in ('refuse', 'refusal', 'refuses', 'refuses to answer',
                                       "don't know", "doesn't know", 'unanswerable',
                                       'unknown', 'no answer'):
                        claim_type = 'refuse'
                    else:
                        claim_type = 'content'
                else:
                    continue
            try:
                n_samples = int(n_str)
                sample_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
            except ValueError:
                n_samples = 0
                sample_indices = []
            if n_samples > 0:
                claims.append({
                    'label': label,
                    'representative': rep.strip(),
                    'n_samples': n_samples,
                    'sample_indices': sample_indices,
                    'type': claim_type,
                })

    # Fallback: infer n_distinct_claims from content cluster count
    content_claims = [c for c in claims if c['type'] == 'content']
    if n_distinct_claims == -1 and content_claims:
        n_distinct_claims = len(content_claims)

    n_clarifying = sum(c['n_samples'] for c in claims if c['type'] == 'clarify')
    n_refusing = sum(c['n_samples'] for c in claims if c['type'] == 'refuse')

    return {
        'n_distinct_claims': n_distinct_claims,
        'n_clarifying': n_clarifying,
        'n_refusing': n_refusing,
        'claims': claims,
        'raw_response': raw,
    }


def parse_contextualisation_response(raw: str) -> Dict:
    """Parse contextualisation prompt output (CLASSIFICATIONS format).

    Returns:
        n_contextualised:   int — responses labelled yes
        n_uncontextualised: int — responses labelled no
        n_clarifying:       int — responses labelled clarify
        n_refusing:         int — responses labelled refuse
        classifications:    list of {index, label, scope}
        raw_response:       str
    """
    if not raw:
        return {
            'n_contextualised': 0, 'n_uncontextualised': 0,
            'n_clarifying': 0, 'n_refusing': 0,
            'classifications': [], 'raw_response': raw,
        }

    classifications = []
    in_block = False
    cls_re = re.compile(
        r'^(\d+):\s*(yes|no|clarify|refuse)(?:\s*[-—]+\s*(.+))?$',
        re.IGNORECASE,
    )

    for line in raw.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('CLASSIFICATIONS:'):
            in_block = True
        elif in_block and line:
            m = cls_re.match(line)
            if m:
                idx_str, label, scope = m.groups()
                classifications.append({
                    'index': int(idx_str),
                    'label': label.lower(),
                    'scope': (scope or '').strip(),
                })

    n_contextualised = sum(1 for c in classifications if c['label'] == 'yes')

    n_uncontextualised = sum(1 for c in classifications if c['label'] == 'no')
    n_clarifying       = sum(1 for c in classifications if c['label'] == 'clarify')
    n_refusing         = sum(1 for c in classifications if c['label'] == 'refuse')

    return {
        'n_contextualised':   n_contextualised,
        'n_uncontextualised': n_uncontextualised,
        'n_clarifying':       n_clarifying,
        'n_refusing':         n_refusing,
        'classifications':    classifications,
        'raw_response':       raw,
    }


def parse_user_answer(response: str) -> Dict[str, str]:
    """Parse LLM response to extract reasoning and user answer."""
    if isinstance(response, list):
        if len(response) > 1:
            raise ValueError("Expecting single response.")
        else:
            response = response[0]
    response = response.strip()

    # Try to extract reasoning and user answer using the expected format
    reasoning_match = None
    user_answer_match = None

    if "USER ANSWER:" in response:
        lines = response.split('\n')

        for idx, line in enumerate(lines):
            if line.startswith("REASONING:"):
                reasoning_match = line.split("REASONING:", 1)[1].strip()
            elif line.startswith("USER ANSWER:"):
                user_answer_match = line.split("USER ANSWER:", 1)[1].strip()
                # Continue reading subsequent lines as part of user answer
                if idx + 1 < len(lines):
                    remaining_lines = lines[idx + 1:]
                    if remaining_lines:
                        user_answer_match += "\n" + "\n".join(remaining_lines)
                break

    return {
        "reasoning": reasoning_match,
        "response": user_answer_match,
        "raw_response": response
    }
