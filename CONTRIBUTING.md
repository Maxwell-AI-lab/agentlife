# Contributing to AgentLife

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Maxwell-AI-lab/agentlife.git
cd agentlife

# Create a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install in dev mode
uv pip install -e ".[openai,dev]"
```

## Running Locally

```bash
# Run the demo to generate test data
python examples/demo_no_api.py

# Start the dashboard
agentlife ui
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Running Tests

```bash
pytest
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests if applicable
4. Run `ruff check .` and `pytest` to ensure nothing is broken
5. Submit a PR with a clear description of the change

## What to Contribute

Check the [Roadmap](README.md#roadmap) for planned features, or look at open issues. Good first contributions:

- Add auto-patchers for other LLM SDKs (Anthropic, Google GenAI)
- Add cost data for more models
- Improve the Web UI
- Write more examples
- Fix bugs and improve docs

## Code of Conduct

Be respectful. We're all here to build something useful together.
