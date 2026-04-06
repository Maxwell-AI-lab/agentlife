# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-04-06

### Added

- Initial release
- OpenAI client auto-patcher (sync + async `chat.completions.create`)
- `@agentlife.trace` decorator for custom functions
- `agentlife.session()` context manager for grouping spans
- SQLite-based local storage (`~/.agentlife/traces.db`)
- Web dashboard with session list, call tree, and detail panel
- Token and cost tracking for 10+ popular models
- CLI commands: `agentlife ui`, `agentlife sessions`, `agentlife clear`
