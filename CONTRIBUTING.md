# Contributing to Armenian Video Dubbing AI

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Be respectful, inclusive, and constructive. We are building tools for the Armenian language community and welcome contributors of all backgrounds.

---

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/Edmon02/armenian-video-dubbing.git
   cd armenian-video-dubbing
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

```bash
# Install dependencies
bash scripts/setup_environment.sh

# Or use pip
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Verify setup
python scripts/verify_setup.py
```

---

## Making Changes

### Branch Naming

- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation changes
- `refactor/description` — Code refactoring
- `test/description` — Test additions/changes

### Commit Messages

Use clear, descriptive commit messages:

```
Add dialect selector to Gradio UI

- Add dropdown for Eastern/Western Armenian
- Pass dialect parameter through pipeline
- Update config validation
```

---

## Pull Request Process

1. Ensure your code passes linting and tests:
   ```bash
   make lint
   make test
   ```
2. Update documentation if your change affects usage
3. Create a pull request with:
   - Clear title describing the change
   - Description of what and why
   - Link to related issue(s) if applicable
4. Wait for review — maintainers will provide feedback

---

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

- **Line length**: 100 characters
- **Target**: Python 3.11
- **Rules**: E, F, I, W (with E501 ignored)

```bash
# Check style
make lint

# Auto-fix issues
make lint-fix

# Format code
make format
```

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

When adding new features, include tests in the `tests/` directory.

---

## Reporting Issues

Use [GitHub Issues](../../issues) with the appropriate template:

- **Bug Report** — Something isn't working
- **Feature Request** — Suggest an enhancement
- **Documentation** — Docs improvement needed

Include as much context as possible: error messages, environment details, steps to reproduce.

---

## Project Areas

Here are the main areas where contributions are welcome:

| Area | Directory | Description |
|------|-----------|-------------|
| Core Pipeline | `src/pipeline.py` | Dubbing orchestrator |
| Inference | `src/inference.py` | Model wrappers |
| API | `src/api/` | FastAPI endpoints |
| UI | `src/ui/` | Gradio interface |
| Evaluation | `scripts/evaluation/` | Quality metrics |
| Training | `scripts/training/` | Model fine-tuning |
| Documentation | `docs/` | Guides and references |
| Tests | `tests/` | Test coverage |

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
