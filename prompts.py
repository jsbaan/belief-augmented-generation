from typing import List, Dict

def create_direct_answer_prompt(question: str, direct_prompt: str) -> List[Dict]:
    """Create QA prompt as message array"""
    if direct_prompt == "concise":
        content = f"Please provide a concise answer to the following question: {question}"
    elif direct_prompt == "sentence":
        content = f"Please provide a short answer of at most 1 sentence to the following question: {question}"
    elif direct_prompt == "cot":
        content = f"""{question}
        
Think through this step by step, then provide your final answer. 

Format your response as:
REASONING: [Your reasoning]
FINAL ANSWER: [Your final answer]"""
    else:
        content = question
    return [{"role": "user", "content": content}]


def create_user_answer_prompt(question: str, clarification_question: str, disambig_question: str, reference: str) -> List[Dict]:
    if disambig_question:
        context_descr = "a disambiguated version of"
        context = f"Secret disambiguated user question: {disambig_question}"
        postfix = ""
    else:
        context_descr = "the final reference answer to"
        context = f"Secret final reference answer: {reference}"
        postfix = f" (DON'T MENTION {reference})"

    prompt = f"""Pretend you are roleplaying a user that asked a question to an AI assistant.

The AI assistant asked a clarification question.
Your task is to formulate the user's response to this clarification question given secret additional context.
You will be given {context_descr} the user question to help resolve any ambiguity or underspecification.

Rules:
- Never reveal the final answer to the user question, only answer the AIs clarification question.
- If the AI's clarification question is impossible to answer or not helpful, please respond "I don't know".

User question (turn 1): {question}
{context}
AI clarification question (turn 2): {clarification_question}
User answer (turn 3): [your answer]

Format your response as:
REASONING: [A compact reasoning chain explaining your answer]
USER ANSWER: [Your simulated user answer to the AIs clarification question {postfix}]
"""
    return [{"role": "user", "content": prompt}]

def create_clarification_prompt_no_belief(question: str) -> List[Dict]:
    """ "prompt": the prompt-only baseline """
    prompt = f"""You are an AI assistant and your goal is to answer a user's question directly, ask a clarification question, or abstain.

- A direct answer is effective when the question is clear and you know the right answer
- A clarification question is effective if it resolves ambiguity or underspecification in the user question, and if the user can answer the clarification question without needing to know the answer to their own question.
- Abstaining is effective if the question is clear but you don't know the right answer.

User question: {question}

Format your response as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
FINAL RESPONSE: [Your response to the user's question. The user will not see your strategy or reasoning.]
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief": almost identical to the prompt-only baseline but adds belief state."""

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are an AI assistant and your goal is to answer a user's question directly, ask a clarification question, or abstain.

Below, you are given {len(belief_samples)} candidate answers to the user's question.
Consider these your state of knowledge and decide on your next action based on the following criteria:

- A direct answer is effective when the question is clear and the answers consistent
- A clarification question is effective if it resolves ambiguity or underspecification in the user question, and if the user can answer the clarification question without needing to know the answer to their own question.
- Abstaining is effective if the question is clear but individual answers are too factually distinct or contradictory to be reliable.

User question: {question}
Candidate answers:
{belief_state_text}

Format your response as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
FINAL RESPONSE: [Your response to the user's question. The user will not see your strategy or reasoning.]

Please provide your response.
"""
    return [{"role": "user", "content": prompt}]

def create_clarification_prompt1(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief1": instruct using semantic variation specifically."""

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question. 

Below, you are given {len(belief_samples)} candidate answers to the user's question. Consider these your state of knowledge.

Your job is to analyze the (semantic) variation between these candidate answers and respond to the user following one of three strategies:
- Direct answer: all candidate answers are semantically equivalent and do not contradict each other.
- Clarification question: The candidate answers contain multiple semantically distinct but plausible answers due to underspecification or ambiguity in the question. Only ask clarification questions that the user can answer by specifying their intent. Never just ask the user to answer their own question. 
- Abstain: the candidate answers are completely factually inconsistent and contradictory.

User question: {question}
Candidate answers:
{belief_state_text}

Format your response as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
RESPONSE: [Your response to the user following your strategy. The user will not see your strategy or reasoning.]
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt2(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief2": candidate answers -> generations"""

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question. 

Below, you are given {len(belief_samples)} sampled generations from an LLM given the user's question. Consider these your belief state.

Your job is to analyze the (semantic) variation between these generations and respond to the user following one of three strategies:
- Direct answer: all generations are semantically equivalent and do not contradict each other.
- Clarification question: The generations contain multiple semantically distinct but plausible answers due to underspecification or ambiguity in the question. Only ask clarification questions that the user can answer by further specifying their intent. Never just ask the user to answer their own question. 
- Abstain: the generations are completely semantically inconsistent and contradictory.

User question: {question}
Candidate answers:
{belief_state_text}

Format your response as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
RESPONSE: [Your response to the user following your strategy. The user will not see your strategy or reasoning.]
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt3(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief3": =belief1, with different phrasing as: "the user shouldn't know about the candidate answers" """

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question. 

Below, you are given {len(belief_samples)} candidate answers to the user's question. Consider these your state of knowledge.

Your job is to analyze the (semantic) variation between these candidate answers and respond to the user following one of three strategies:
- Direct answer: all candidate answers are semantically equivalent and do not contradict each other.
- Clarification question: The candidate answers contain multiple semantically distinct but plausible answers due to underspecification or ambiguity in the question. Only ask clarification questions that the user can answer by specifying their intent. Never just ask the user to answer their own question. 
- Abstain: the candidate answers are completely factually inconsistent and contradictory.

User question: {question}
Candidate answers:
{belief_state_text}

Format your response as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
RESPONSE: [Your response to the user following your strategy. The user shouldn't know about the candidate answers, your strategy or reasoning.]

Please provide your response.
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt4(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief4": mix of belief1 (and thus 3) and also 2 (mix of generations and cand answers)
     but NO ABSTAIN, and with a SUMMARY instead of reasoning before the strategy."""

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question. 

Below, you are given {len(belief_samples)} sampled generations from an LLM. The user cannot see these answers.

Your job is to analyze the (semantic) variation between these generations and either directly answer the user or ask for clarification: 
- Direct answer: all candidate answers are semantically equivalent and do not contradict each other.
- Clarification question: The candidate answers contain multiple semantically distinct answers due to underspecification or ambiguity in the question. Only ask clarification questions that the user can answer by specifying their intent. Never just ask the user to answer their own question. 

User question: {question}
Candidate answers:
{belief_state_text}

Format your response as:
SUMMARY: [A very short summary extracting the main answers from the candidate answers]
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION]
RESPONSE: [Your response to the user following your strategy. The user will not see your strategy or reasoning.]

Please provide your response.
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt5(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief5": similar to belief1/2/3 and talk about uncertainty and further specify abstain to avoid it containing clarification questions too """

    # Format samples as numbered list
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question. 

Below, you are given {len(belief_samples)} candidate answers to the user's question. Consider these your state of knowledge.

Your job is to analyze the (semantic) variation between these candidate answers and respond to the user following one of three strategies:
- DIRECT_ANSWER: all candidate answers are semantically equivalent and do not contradict each other - there is no uncertainty.
- CLARIFICATION_QUESTION: The candidate answers contain multiple semantically distinct answers due to underspecification or ambiguity in the question, and there is a clear clarification question that would help you identify the right answer AND that a user would be able to answer. 
- ABSTAIN: the candidate answers are completely factually inconsistent and contradictory due to your lack of knowledge. There is no useful possible clarification question (otherwise use the "clarification question strategy") so you politely refuse to answer because you are too uncertain. 

User question: {question}
Candidate answers:
{belief_state_text}

Format your output as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
RESPONSE: [Your response to the user's question based on the (hidden) candidate answers, following your chosen strategy.]

Please provide your output.
"""
    return [{"role": "user", "content": prompt}]

def create_clarification_prompt6(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief6": best-effort BAG prompt — principled abstain (belief5-style) + MBR-inspired direct generation. """
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))

    prompt = f"""You are a helpful AI assistant in conversation with a user that asked a question.

Below, you are given {len(belief_samples)} sampled generations from a language model given the user's question. Consider these your belief state — a representation of your uncertainty about the answer.

Your job is to analyze these generations and choose one of three strategies:

- DIRECT_ANSWER: The generations converge on the same answer — there is no meaningful uncertainty. Synthesize the most representative response from your belief state, at roughly the same length as the individual generations.
- CLARIFICATION_QUESTION: The generations reflect multiple semantically distinct but plausible answers, indicating the question is ambiguous or underspecified. Ask a clarification question that the user can answer by specifying their intent. Never ask the user to answer their own question.
- ABSTAIN: The generations are factually contradictory and inconsistent due to your lack of knowledge, AND no useful clarification question exists that would resolve the uncertainty. Politely decline to answer.

User question: {question}
Candidate answers:
{belief_state_text}

Format your output as:
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [A compact reasoning chain explaining your decision]
RESPONSE: [Your response to the user's question based on the (hidden) candidate answers, following your chosen strategy.]

Please provide your output.
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt7(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief7": explicit cluster→interpret→route structure.
    Forces the model to name semantic groups and map each to a question interpretation
    before choosing a strategy. This distinguishes ambiguity (coherent clusters with
    distinct interpretations → clarify) from knowledge gaps (incoherent scatter → abstain).

    NOTE: the CLUSTERS and INTERPRETATIONS fields are not parsed by parse_clarification_response
    but are preserved in raw_response for analysis.

    Token budget: CLUSTERS + INTERPRETATIONS add ~100-175 tokens of output overhead.
    Recommend max_new_tokens=700 for this variant.

    """
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))
    n = len(belief_samples)

    prompt = f"""You are a helpful AI assistant in conversation with a user.

Below are {n} candidate answers representing your belief state — your uncertainty about the answer to the user's question.

Analyze them in two steps, then choose a strategy and respond.

Step 1 — Cluster by meaning: Group the answers by what they assert. Ignore surface variation (wording, punctuation); group by the underlying fact or claim. Give each group a letter label and a representative answer.

Step 2 — Interpret each cluster: For each group, ask: "What interpretation of the user's question would naturally lead to this answer?" Write one interpretation per group. If you cannot identify any coherent interpretation for a group, mark it "uninterpretable."

Step 3 — Choose a strategy:
- DIRECT_ANSWER: there is only 1 meaningful cluster, or all clusters share a single interpretation. Give a complete, direct answer synthesized from the dominant cluster.
- CLARIFICATION_QUESTION: there are 2–3 clusters, each with a distinct coherent interpretation, and the user can specify their intent without already knowing the answer. Ask about the user's context or purpose — do NOT simply ask "do you mean X or Y?", instead ask about the situation or intent that would determine which interpretation applies.
- ABSTAIN: clusters are uninterpretable, too numerous to resolve, or no clarification question could realistically distinguish them. Politely decline.

User question: {question}
Candidate answers:
{belief_state_text}

Format your output as:
CLUSTERS: [one line per group, e.g. "A ({{k}}/{n}): <representative answer>"]
INTERPRETATIONS: [one line per group, e.g. "A → <what interpretation of the question leads here>"]
STRATEGY: [DIRECT_ANSWER / CLARIFICATION_QUESTION / ABSTAIN]
REASONING: [one compact sentence explaining your routing decision]
RESPONSE: [your response to the user — they do not see the candidate answers, clusters, or reasoning]
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt8(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief8": faithful two-signal routing operationalising interpretation_variation and
    claim_variation judges, with MBR-style synthesis for direct answers.

    Routing (checked in order):
      1. Any answer contains an explicit scope marker → CLARIFICATION_QUESTION about that scope
      2. More than half of answers agree on the same claim → DIRECT_ANSWER (synthesise from majority)
      3. Otherwise (scattered factual claims, no scope) → ABSTAIN

    NOTE: SCOPE_MARKERS field is output before STRATEGY to force explicit scope-checking
    before majority routing. Not parsed by parse_clarification_response.
    """
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))
    n = len(belief_samples)

    prompt = f"""You are a helpful AI assistant.

Below are {n} candidate answers representing your uncertainty about the user's question.

Two signals determine your strategy:

- Interpretation variation: one or more answers contain an explicit scope marker that narrows the question to a particular interpretation not stated in the question — for example a country, time period, tradition, or domain (e.g. "In the United States, ..." or "As of 2023, ..."). Only count this when the scope marker is explicitly written in the answer — do not infer scope from differing facts alone. Note: scope embedded inside a noun phrase (e.g. "13 Premier League titles") counts just as much as a leading marker.

- Claim variation: answers assert different facts with no explicit scope markers — the model is uncertain about the answer itself.

Step 1 — Check for scope markers: Do any answers contain an explicit scope marker that narrows the question to a particular interpretation not stated in the question? Examples: a country ("In the US, ..."), time period ("As of 2023, ..."), tradition ("In the Catholic Church, ..."), domain, or version.

Step 2 — Choose a strategy:
- CLARIFICATION_QUESTION: any answer contains an explicit scope marker (whether one or many distinct scopes). Ask the user directly about that specific scope dimension — name it from the candidate answers.
- DIRECT_ANSWER: no scope markers found, and more than half of the answers assert the same claim. Synthesise a response from those answers.
- ABSTAIN: no scope markers found, and no single claim accounts for more than half the answers. Politely decline to answer.

User question: {question}
Candidate answers:
{belief_state_text}

SCOPE_MARKERS: [yes/no — and if yes, name the scope(s) found]
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [one sentence]
RESPONSE: [your response to the user — they do not see the candidate answers or reasoning]
"""
    return [{"role": "user", "content": prompt}]


def create_clarification_prompt9(question: str, belief_samples: List[str]) -> List[Dict]:
    """ "belief9": response-type routing — reads the types of responses in the belief state
    rather than analysing variation between them.

    Three response types in the belief state are actionable signals:
      - CQ: a sample asks a clarification question → the model itself wants clarification
      - Scoped answer: a sample adds scope not present in the question → ask about that scope
      - Refusal: a sample refuses or says it doesn't know → signal of genuine uncertainty

    Routing (checked in order):
      1. Any sample is a CQ or a scoped answer → CLARIFICATION_QUESTION
      2. Most samples are refusals → ABSTAIN
      3. Otherwise → DIRECT_ANSWER (synthesise majority claim)

    NOTE: RESPONSE_TYPES field is output before STRATEGY to force explicit type-checking.
    Not parsed by parse_clarification_response.
    """
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))
    n = len(belief_samples)

    prompt = f"""You are a helpful AI assistant.

Below are {n} candidate answers representing your uncertainty about the user's question.

Step 1 — Classify each answer into one of these response types:
- cq: the answer asks a clarification question instead of answering
- scoped: the answer adds an explicit scope not present in the question (a country, time period, domain, version, etc.) and answers within that scope
- refusal: the answer refuses to answer, says it doesn't know, or says the question is unanswerable
- direct: the answer gives a direct factual response without the above

Step 2 — Choose a strategy:
- CLARIFICATION_QUESTION: any answer is type cq or scoped. Ask the user the clarification question the model is already asking, or ask about the scope the model is already assuming — name it specifically.
- ABSTAIN: no cq or scoped answers, but most answers are refusals. Politely decline.
- DIRECT_ANSWER: no cq, scoped, or dominant refusals. Synthesise a response from the majority claim.

User question: {question}
Candidate answers:
{belief_state_text}

RESPONSE_TYPES: [list each answer's type, e.g. "1=direct, 2=scoped(US), 3=direct, ..."]
STRATEGY: [DIRECT_ANSWER/CLARIFICATION_QUESTION/ABSTAIN]
REASONING: [one sentence]
RESPONSE: [your response to the user — they do not see the candidate answers or reasoning]
"""
    return [{"role": "user", "content": prompt}]


def create_final_answer_prompt(question: str, clarification_question: str, user_answer: str, direct_prompt: str) -> List[Dict]:
    """Create multi-turn QA prompt with clarification conversation as message array"""

    messages = create_direct_answer_prompt(question, direct_prompt)
    messages.extend([
        {"role": "assistant", "content": clarification_question},
        {"role": "user", "content": user_answer},
    ])

    return messages

def create_final_answer_prompt1(question: str, clarification_question: str, user_answer: str, direct_prompt: str) -> List[Dict]:
    """Prompt-only baseline: instructs the model to answer only if it can, otherwise abstain.

    No belief state is used. The model sees the full conversation and decides for itself
    whether the clarification exchange gave it enough information to answer confidently.
    Uses the same routing framing as create_final_belief_reasoner_prompt for fair comparison.
    """
    messages = create_direct_answer_prompt(question, direct_prompt)
    messages.extend([
        {"role": "assistant", "content": clarification_question},
        {"role": "user", "content": user_answer + """

Now answer the original question based on our conversation so far.

- DIRECT_ANSWER: You are confident you know the answer. Provide it.
- ABSTAIN: You are still uncertain and should not guess. Politely decline.

Format your output as:
STRATEGY: [DIRECT_ANSWER / ABSTAIN]
REASONING: [one compact sentence explaining your decision]
RESPONSE: [your answer to the user if DIRECT_ANSWER, or a polite decline if ABSTAIN]"""},
    ])

    return messages


def create_final_belief_reasoner_prompt(question: str, clarification_question: str, user_answer: str, direct_prompt: str, belief_samples: List[str]) -> List[Dict]:
    """Post-clarification BAG reasoner: decides answer vs abstain based on belief-state consensus.

    Unlike the pre-clarification reasoner (belief8) which detects scope markers and routes
    via majority-vote synthesis, this prompt focuses purely on whether the model's belief state
    after clarification converges on a single factual claim. The question of ambiguity has
    already been addressed by the clarification exchange — what matters now is whether the
    model actually *knows* the answer.

    Routing logic:
    - High consensus (most samples agree on the same factual claim) → DIRECT_ANSWER
    - High diversity (samples make different factual claims) → ABSTAIN
      (diversity signals the model is guessing, not that the question is ambiguous)
    """
    belief_state_text = "\n".join(f"{i + 1}. {sample}" for i, sample in enumerate(belief_samples))
    n = len(belief_samples)

    messages = create_direct_answer_prompt(question, direct_prompt)
    messages.extend([
        {"role": "assistant", "content": clarification_question},
        {"role": "user", "content": user_answer + f"""

Below are {n} candidate answers representing your belief state — independent attempts at answering the original question given our conversation so far.

Determine whether your belief state shows enough consensus to give a confident answer, or whether the candidates diverge too much (indicating you are guessing rather than knowing).

Step 1 — Assess consensus: Do the {n} candidate answers agree on the same core factual claim? Ignore minor surface differences in wording — focus on whether they assert the same fact. Count how many samples support the most common claim.

Step 2 — Route:
- DIRECT_ANSWER: The candidates largely agree (≥ 70% support the same claim). Synthesize a single confident answer based on the consensus.
- ABSTAIN: The candidates make multiple different factual claims with no clear majority. This means you are uncertain and should not guess. Politely decline.

Candidate answers:
{belief_state_text}

Format your output as:
CONSENSUS: [one line: "X/{n} candidates agree on: <the claim>" or "no consensus — candidates split across N different claims"]
STRATEGY: [DIRECT_ANSWER / ABSTAIN]
REASONING: [one compact sentence explaining your decision]
RESPONSE: [your answer to the user if DIRECT_ANSWER, or a polite decline if ABSTAIN]"""},
    ])

    return messages


def create_llm_judge_eval_prompt_anyref(question: str, candidate: str, all_refs: List[List[str]]) -> List[Dict]:
    """LLM judge prompt for anyref evaluation — correct if the candidate asserts ANY valid interpretation.

    Uses a single call listing all interpretation groups, which is cheaper than one call per reference
    and lets the judge reason across all valid answers at once.

    When there is only one reference group this is equivalent to create_llm_judge_eval_prompt.

    Args:
        question:  the original user question
        candidate: the model's answer to evaluate
        all_refs:  list of reference groups; each group contains surface forms for one interpretation
    """
    if len(all_refs) == 1:
        return create_llm_judge_eval_prompt(question, candidate, all_refs[0])

    ref_lines = []
    for i, ref_group in enumerate(all_refs, 1):
        ref_text = " / ".join(r for r in ref_group if r)
        ref_lines.append(f"  Interpretation {i}: {ref_text}")
    refs_block = "\n".join(ref_lines)

    prompt = f"""Your task is to evaluate whether a model answer correctly responds to a factual question.

This question has multiple valid interpretations. The model answer is correct if it directly asserts ANY ONE of the following correct answers as its main claim:

{refs_block}

The correct answers above are ground truth — treat them as authoritative regardless of your own knowledge.

Guidelines:
- CORRECT ("yes"): the model directly asserts any one correct answer as its main claim. Semantically equivalent forms are accepted (abbreviations, name variants, different date formats, singular/plural).
- INCORRECT ("no"): the model (a) states a different answer as its main claim for none of the interpretations, (b) mentions a correct answer only incidentally (e.g. as one item in a list, in a historical aside, or as a counterexample without endorsing it), or (c) is too vague or non-committal to constitute a clear answer.

Question: {question}
Model answer: {candidate}

Respond in this exact format:
REASONING: [one sentence explaining your decision]
VERDICT: yes / no"""
    return [{"role": "user", "content": prompt}]


def create_llm_judge_eval_prompt(question: str, candidate: str, reference: List[str]) -> List[Dict]:
    """LLM judge prompt for pipeline evaluation.

    Uses a reasoning-then-verdict format for reliability. Explicitly guards against
    giving credit for incidental mentions of the reference — the main failure mode
    of ROUGE-L recall on long or hedging answers.

    Args:
        question:  the original user question
        candidate: the model's answer to evaluate
        reference: surface forms for one specific reference answer, e.g. ['1971'] or ['June 18, 1968']
    """
    ref_text = " / ".join(r for r in reference if r) if len(reference) > 1 else reference[0]
    prompt = f"""Your task is to evaluate whether a model answer correctly responds to a factual question.

Correct answer: {ref_text}

The correct answer above is ground truth — treat it as authoritative regardless of your own knowledge.

Guidelines:
- CORRECT ("yes"): the model directly asserts the correct answer as its main claim. Semantically equivalent forms are accepted (abbreviations, name variants, different date formats, singular/plural).
- INCORRECT ("no"): the model (a) states a different answer as its main claim, (b) mentions the correct answer only incidentally (e.g. as one item in a list, in a historical aside, or as a counterexample without endorsing it), or (c) is too vague or non-committal to constitute a clear answer.

Question: {question}
Model answer: {candidate}

Respond in this exact format:
REASONING: [one sentence explaining your decision]
VERDICT: yes / no"""
    return [{"role": "user", "content": prompt}]




def create_claim_variation_prompt(question: str, belief_samples: List[str]) -> List[Dict]:
    """One-call belief-state diversity prompt.

    Sends all K samples to the LLM and asks it to cluster them by main factual claim.
    Returns structured output with N_DISTINCT_CLAIMS count + per-cluster details.
    """
    n = len(belief_samples)
    belief_state_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(belief_samples))
    prompt = f"""Your task is to cluster {n} responses to the question below by their main answer.

Ignore surface variation in wording, punctuation, hedging, or additional context — group by the underlying answer.
Each response belongs to exactly one cluster, so the total number is at most {n}. 

Responses that refuse (say they don't know, say the question is unanswerable, or say the thing being asked about doesn't exist) or 2) ask a clarification question should have their own cluster. 

Write a short representative label (a name, date, or brief phrase - not a full sentence - or clarify or refuse). 

Only include a cluster if it contains at least one answer.

Question: {question}
Responses:
{belief_state_text}

Respond in this exact format:
N_DISTINCT_ANSWERS: <integer>
ANSWERS:
A (n=<count>, indices=<1-based indices>): <short label>
B (n=<count>, indices=<1-based indices>): <short label>
[continue for each cluster]"""
    return [{"role": "user", "content": prompt}]


def create_contextualisation_prompt(question: str, belief_samples: List[str]) -> List[Dict]:
    """Per-response contextualisation classifier for belief-state analysis.

    Classifies each sample as contextualised (contains an explicit scope marker
    not present in the question), uncontextualised, clarifying, or refusing.
    Replaces the clustering-based create_interpretation_variation_prompt.
    """
    n = len(belief_samples)
    belief_state_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(belief_samples))
    prompt = f"""Your task is to classify each of {n} responses to the question below.

For each response, decide whether it is contextualised: does it explicitly assume a specific interpretation of the question that is not stated in the question itself?

Examples of contextualising:
  - Geographic: "In the United States, ..." / "For the UK, ..."
  - Temporal: "As of 2023, ..." / "At the time of writing, ..."
  - Demographic: "For adults, ..." / "In the context of children, ..."
  - Domain or tradition: "In the Catholic Church, ..." / "Under US law, ..."
  - Version or edition: "In the 1993 film, ..." / "For the 2024 model, ..."
  - etc...

A response is NOT contextualised if it answers without taking any explicit interpretation (even if its answer differs from other responses), or if it only restates terms already present in the question (e.g. if the question asks about "The Greatest Showman", a response that says "The Greatest Showman (2017)" is not adding a new interpretation).

Use these labels:
- yes: the response explicitly adds a specific interpretation of the question
- no: no explicit interpretation
- clarify: asks a clarification question instead of answering
- refuse: refuses to answer, says it doesn't know, or says the question is unanswerable

Question: {question}
Responses:
{belief_state_text}

Respond in this exact format:
CLASSIFICATIONS:
1: yes — <brief scope, e.g. "US law">
2: no
3: clarify
[one line per response, 1 to {n}]"""
    return [{"role": "user", "content": prompt}]
