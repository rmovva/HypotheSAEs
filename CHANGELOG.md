# Changelog

New version releases for HypotheSAEs will be documented here.

## [0.3.0] - 2025-07-28

### Added
- `llm_local.py` module for local LLM inference with vLLM, `tests/test_local_llm.py` for unit tests
- `quickstart_local.ipynb` notebook to get started with local LLMs
- `local_llm_experiments` contains experiments benchmarking local LLMs for autointerp and concept annotation

### Changed
- `interpret_neurons.py`, `annotate.py`, `requirements.txt` modified to support local LLM inference
- `quickstart.ipynb` notebook now uses matryoshka SAE by default

### Fixed
- `requirements.txt` forces scipy==1.15.3 to avoid bug with Python 3.13

## [0.2.0] - 2025-05-03

### Added
- Matryoshka SAEs: https://github.com/rmovva/HypotheSAEs/pull/1

### Fixed
- Account for the changed parameter name `print_examples_n` in `quickstart.interpret_sae()` (from `print_examples`) in the `quickstart.ipynb` notebook.

## [0.1.0] - 2025-04-22

### Added
- Basic unit tests for embeddings and quickstart functions

### Changed
- Add param for users to include more characters when printing examples in `quickstart.interpret_sae()`
- Only catch API errors for timeout + rate limit errors

### Fixed
- Ensure that in `sae.get_activations()` and all functions in `quickstart.py`, we never load the full dataset onto the GPU
- When sampling examples in `interpret_neurons.py`, we handle the case where there are not enough examples to sample

## [0.0.5] - 2025-03-18

This was the initial release of HypotheSAEs (with small bug fixes).
