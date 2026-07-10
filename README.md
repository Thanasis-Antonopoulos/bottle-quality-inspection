# Bottle Quality Inspection

A production-oriented computer vision system for automated bottle quality inspection.

## Project status

This repository is an active modernization of a Machine Vision project originally developed in 2022.

The original implementation inspected bottles moving through an ouzo bottling process and detected quality problems using classical computer vision techniques.

The new implementation aims to provide a clean, configurable, tested, and deployment-ready inspection pipeline.

## Planned inspection capabilities

- Bottle detection
- Cap or cork inspection
- Liquid fill-level inspection
- Foreign-object detection
- Bottle damage and crack detection
- Annotated video generation
- Bottle-level inspection reports
- Performance and quality metrics

## Project objectives

The project will:

1. Reproduce the working behaviour of the original 2022 implementation.
2. Refactor the original scripts into reusable Python modules.
3. Add bottle tracking and bottle-level decisions.
4. Add automated tests and code-quality checks.
5. Provide configurable inspection rules.
6. Support deployment through a command-line application.
7. Provide a demonstration interface after the inspection engine is stable.

## Repository structure

```text
bottle-quality-inspection/
├── docs/                     Project documentation
├── legacy/                   Preserved historical implementations
│   ├── 2022/                 Original university implementation
│   └── 2025/                 Experimental revised implementation
├── src/
│   └── bottle_quality_inspection/
│                              Production application source code
├── tests/                    Automated tests
├── pyproject.toml            Python project configuration
└── README.md
```

## Development setup

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package and development tools:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the application:

```powershell
python -m bottle_quality_inspection
```

Run the tests:

```powershell
pytest
```

Run code-quality checks:

```powershell
ruff check .
```

## Historical implementations

The original 2022 and experimental 2025 implementations will be preserved under `legacy/`.

The production application will not directly import code from the legacy folders. Relevant logic will be reviewed, tested, and migrated into the new package.

## Current version

```text
0.1.0 — Initial project structure
```