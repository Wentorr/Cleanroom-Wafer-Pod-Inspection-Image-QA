# 300 mm FOUP Load-Port Handoff Panel QA

Complete Eris-style challenge package for a synthetic multimodal computer vision task grounded in semiconductor FOUP/load-port handoff operations.

- `raw/generate_raw.py` creates deterministic rendered inspection-panel PNGs and trace-derived QA rows with explicit easy/medium/hard visual tiers.
- `raw/questions.csv` contains all raw questions, answers, and hidden evaluation groups.
- `raw/images/` contains source PNG images.
- `dataset_description_eris_upload.md` documents the dataset.
- `prepare.py` creates public train/test files and private answers.
- `problem.md` is the solver-facing challenge statement.
- `grade.py` computes weighted accuracy with worst-group penalties.
- `rubrics.yaml` contains task-specific rubric criteria.
- `reference_solution.py` and `solution.ipynb` provide a solvability baseline.

## Submission Mapping

Dataset upload:
- Title: `Synthetic 300 mm FOUP Load-Port Handoff Panel QA Dataset`
- Description: paste `dataset_description_eris_upload.md`
- Data file: upload `cleanroom-wafer-pod-vqa-raw.tar.gz` or the refreshed `cleanroom-wafer-pod-vqa-raw.zip`
- License: `CC0 1.0 Public Domain`

Challenge:
- Domain: `Computer Vision`
- Difficulty: `Medium`
- Tags: `image`, `multimodal`, `small-data`, `feature-engineering`
- Title: `300 mm FOUP Load-Port Handoff Panel QA`
- Grade direction: `Maximize`
- Min score: `0`
- Max score: `1`
- Problem description: paste `problem.md`
- Grading script: paste `grade.py`
- Prepare script: paste `prepare.py`
- Rubrics: use `rubrics.yaml`
- Reference solution: upload `solution.ipynb`
