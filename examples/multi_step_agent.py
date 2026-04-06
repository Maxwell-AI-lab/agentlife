"""Multi-step agent example: shows nested spans with @trace decorator.

Usage:
    pip install agentlife[openai]
    export OPENAI_API_KEY=sk-...
    python examples/multi_step_agent.py
    agentlife ui   # open http://localhost:8777
"""

from openai import OpenAI
import agentlife

agentlife.init()

client = OpenAI()


@agentlife.trace
def plan(question: str) -> str:
    """First step: ask the LLM to create a plan."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Break the question into 2-3 sub-tasks. Be concise."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


@agentlife.trace
def research(subtask: str) -> str:
    """Second step: research each sub-task."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer this research sub-task concisely."},
            {"role": "user", "content": subtask},
        ],
    )
    return response.choices[0].message.content or ""


@agentlife.trace
def synthesize(question: str, findings: list[str]) -> str:
    """Final step: combine findings into an answer."""
    context = "\n---\n".join(findings)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Synthesize these findings into a clear answer.\n\nFindings:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def main():
    question = "What are the pros and cons of microservices vs monolithic architecture?"

    with agentlife.session("multi-step-agent"):
        plan_text = plan(question)
        print(f"Plan:\n{plan_text}\n")

        subtasks = [line.strip("- ").strip() for line in plan_text.split("\n") if line.strip()]
        findings = []
        for task in subtasks[:3]:
            result = research(task)
            findings.append(result)
            print(f"Research: {task[:50]}... done")

        answer = synthesize(question, findings)
        print(f"\nFinal answer:\n{answer}")


if __name__ == "__main__":
    main()
