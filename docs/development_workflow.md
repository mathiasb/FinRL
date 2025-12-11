# Development Workflow & Standards

This document outlines the standard operating procedures for the FinRL Alpha-FX project, emphasizing Test-Driven Development (TDD) and continuous documentation.

## 1. Test-Driven Development (TDD) Hierarchy
We strictly adhere to a **Red-Green-Refactor** cycle. No feature code is written without a failing test.

### Levels of Testing
1.  **Unit Tests (`tests/`)**:
    -   **Scope**: Individual classes/methods (e.g., `FXDataPipeline.clean_data`, `FXPortfolioEnv.step`).
    -   **Dependencies**: Must use **Mocking** or **Static Fixtures**. No external API calls allowed.
    -   **Execution Speed**: < 100ms per test.
    -   *Example*: Verifying that `clean_data` calculates a moving average correctly using a fixed numpy array.

2.  **Infrastructure/Integration Tests**:
    -   **Scope**: Interaction between modules or external services (if absolutely necessary).
    -   **Dependencies**: Allowed (e.g., File system I/O, Database).
    -   **Location**: Separate suite, run less frequently.

### TDD Process
1.  **Write Failing Test**: Create a test case in `tests/test_<module>.py` that defines the expected behavior (Inputs -> Outputs/State Change).
2.  **Verify Failure**: Run the test to confirm it fails (Red). `python -m unittest tests/test_...`
3.  **Implement Minimal Code**: Write just enough code to pass the test (Green).
4.  **Refactor**: Improve code quality while keeping tests passing.
    -   *Philosophy*: **"Make it work, then make it good, then make it fast."**
        1.  **Make it work**: Pass the test (Green).
        2.  **Make it good**: Clean up code, improve readability, apply patterns (Refactor).
        3.  **Make it fast**: Optimize performance only if necessary and measured.

## 2. CI/CD Pipeline
Continuous Integration is managed via GitHub Actions.

-   **Triggers**:
    -   Push to your active development branch (e.g., `fxmvp1`, `develop`).
    -   Pull Requests targeting your stable branch.
-   **Branching Strategy (Fork)**:
    -   Keep `main` synced with upstream (optional).
    -   Maintain a long-lived stable branch (e.g., `develop` or `fx-trading`) for your project.
    -   Work in feature branches off your stable branch.
-   **Jobs**:
    -   **Lint**: Checks code style and syntax.
    -   **Test**: Runs the full unit test suite with coverage reporting.

## 3. Documentation "As Code"
Documentation is treated as a first-class citizen and lives in the repository.

-   **Diagrams**: Use MermaidJS within Markdown files.
-   **Architecture**: `docs/system_architecture.md` must be updated if class structures or data flows change.
-   **Walkthroughs**: After major milestones, update `walkthrough.md` to summarize changes and validation results.

## 4. Environment Management
-   **Dependencies**: Managed in `requirements.txt` (and potentially `setup.py` / `pyproject.toml`).
-   **Virtual Env**: Always activate `.venv` before developing.
