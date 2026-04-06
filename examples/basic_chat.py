"""Basic example: trace a simple OpenAI chat completion.

Usage:
    pip install agentlife[openai]
    export OPENAI_API_KEY=sk-...
    python examples/basic_chat.py
    agentlife ui   # open http://localhost:8777
"""

from openai import OpenAI
import agentlife

agentlife.init()

client = OpenAI()

with agentlife.session("basic-chat"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in 3 sentences."},
        ],
    )
    print(response.choices[0].message.content)
