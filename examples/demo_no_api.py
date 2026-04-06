"""Demo that generates trace data WITHOUT needing any API key.

Usage:
    python examples/demo_no_api.py
    agentlife ui
"""

import time
import agentlife
from agentlife.collector import Collector
from agentlife.models import SpanType

agentlife.init(patch_openai=False)


@agentlife.trace
def plan(question: str) -> str:
    time.sleep(0.1)
    return "1. Research the topic\n2. Find examples\n3. Synthesize answer"


@agentlife.trace
def search(query: str) -> str:
    time.sleep(0.15)
    return f"Found 3 results for: {query}"


@agentlife.trace(name="llm-call")
def fake_llm_call(prompt: str) -> str:
    """Simulate an LLM call with manual span data."""
    collector = Collector.get()
    span = collector.start_span(
        name="chat.completions.create(gpt-4o-mini)",
        span_type=SpanType.LLM,
        input_data={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": prompt},
            ],
            "model": "gpt-4o-mini",
        },
        model="gpt-4o-mini",
    )
    time.sleep(0.2)
    collector.end_span(
        span,
        output_data={
            "role": "assistant",
            "content": f"Here is a simulated response to: {prompt[:50]}...",
            "finish_reason": "stop",
        },
        prompt_tokens=150,
        completion_tokens=80,
    )
    return f"Simulated response for: {prompt[:50]}"


@agentlife.trace
def synthesize(findings: list[str]) -> str:
    time.sleep(0.1)
    return "Final synthesized answer based on all findings."


def main():
    # Session 1: successful multi-step agent
    with agentlife.session("research-agent"):
        result = plan("What is reinforcement learning from human feedback?")
        print(f"Plan: {result}")

        r1 = search("RLHF overview")
        r2 = search("RLHF examples in practice")
        print(f"Search done: {r1}, {r2}")

        llm_result = fake_llm_call("Explain RLHF in simple terms")
        print(f"LLM: {llm_result}")

        answer = synthesize([r1, r2, llm_result])
        print(f"Answer: {answer}")

    time.sleep(0.3)

    # Session 2: with an error
    with agentlife.session("failing-agent"):
        try:
            plan("What causes bugs?")
            search("common software bugs")
            raise ValueError("Simulated API timeout error")
        except ValueError:
            pass

    time.sleep(0.3)

    # Session 3: multiple LLM calls
    with agentlife.session("chat-bot"):
        fake_llm_call("Hello, who are you?")
        fake_llm_call("Tell me a joke about programming")
        fake_llm_call("Now explain the joke")

    print("\nDone! Run 'agentlife ui' to view traces.")


if __name__ == "__main__":
    main()
