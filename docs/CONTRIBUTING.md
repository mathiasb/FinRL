# Contributing to FinRL Alpha-FX

Thank you for your interest in contributing!

## Project Overview
This project targets the development of a robust FX Trading reinforcement learning system under a strict **Test-Driven Development (TDD)** and **Documentation-First** approach.

## Getting Started
1.  **Clone**: `git clone ...`
2.  **Environment**: 
    -   Create `.venv`: `python -m venv .venv`
    -   Activate: `source .venv/bin/activate`
    -   Install: `pip install -r requirements.txt` & `pip install -e .`

## Development Standards
We follow a strict Red-Green-Refactor process. Please read the detailed guidelines in:
-   [**Development Workflow & TDD Guide**](development_workflow.md)
-   [**System Architecture**](system_architecture.md)

### Checklist for Pull Requests
-   [ ] **Issue**: Linked to an existing issue.
-   [ ] **TDD**: Include a new test case that fails without your changes.
-   [ ] **Tests**: All tests pass locally (`pytest`).
-   [ ] **Docs**: Updated relevant diagrams or markdown files if architecture changed.
-   [ ] **Lint**: Code is formatted (Ruff/Black).

## Reporting Bugs
Please open an issue with the template provided in `.github/ISSUE_TEMPLATE/bug_report.md`.
