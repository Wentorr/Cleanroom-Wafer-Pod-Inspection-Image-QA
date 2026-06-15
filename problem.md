# Cleanroom Wafer Pod Inspection Image QA

## Overview

Semiconductor cleanroom technicians inspect wafer pod handling stations using camera feeds and station overlays before releasing a carrier to the next process step. Your task is to answer multiple-choice questions about synthetic inspection-panel images.

Each PNG image shows a wafer cassette, colored wafer slots, a two-digit pressure gauge, status LEDs, cable ports, a robot arm, a route strip, a lot-code display, and a visible particle/defect marker. Questions are objective and trace-derived: they ask about concrete visible details such as which slot contains a colored wafer, which module has a red LED, which port is connected to a yellow cable, what the gauge reads, where the defect marker is located, or which waypoint follows another in the route strip.

The challenge is intentionally multimodal. Solvers must combine the image, natural-language question, and answer options. The hidden test set includes low-light panels, glare, dense visual clutter, and partial occlusion, so relying on text priors or majority answers is not enough.

Visual difficulty is structured into easy, medium, and hard tiers. Easy panels have clean visibility, medium panels include light clutter or mild lighting shifts, and hard panels include stronger interference such as glare, low light, dense clutter, or partial occlusion. Critical answer-bearing regions such as gauge digits, lot-code digits, route digits, port labels, module labels, and cassette slot markers remain readable; the challenge is to parse the scene reliably, not to guess through unreadable text.

## Evaluation

Submissions are scored using weighted balanced accuracy:

`score = 0.50 * overall_accuracy + 0.18 * worst_query_type_accuracy + 0.10 * worst_split_group_accuracy + 0.07 * worst_ood_axis_accuracy + 0.05 * worst_layout_family_accuracy + 0.03 * worst_lighting_condition_accuracy + 0.02 * worst_clutter_level_accuracy + 0.05 * worst_difficulty_tier_accuracy`

All terms are accuracies in `[0, 1]`, so the final score is also in `[0, 1]`. Higher is better.

The worst-group terms make the benchmark harder to game. A submission that answers common easy questions but fails under glare, low light, route reasoning, cable matching, or hard-tier scenes will receive a lower score.

## Dataset

The prepared public dataset contains:

| File or Folder | Description |
|---|---|
| `train.csv` | Labeled image-question rows with answer options and the correct `answer`. |
| `test.csv` | Held-out image-question rows with answer options but no answer. |
| `sample_submission.csv` | Example submission with the required columns. |
| `images/` | PNG inspection-panel images referenced by train and test rows. |

## Columns

| Column | Type | Present In | Description |
|---|---:|---|---|
| `id` | string | train, test, submission | Unique question id. |
| `image` | string | train, test | Relative path to the PNG under `public/`. |
| `query_type` | string | train, test | Question category. |
| `question` | string | train, test | Natural-language question. |
| `option_A` | string | train, test | Answer option A. |
| `option_B` | string | train, test | Answer option B. |
| `option_C` | string | train, test | Answer option C. |
| `option_D` | string | train, test | Answer option D. |
| `option_E` | string | train, test | Answer option E. |
| `answer` | string | train only | Correct answer letter. |

## Query Types

| Query Type | Description |
|---|---|
| `slot_color` | Identify which cassette slot contains a named wafer color. |
| `gauge_reading` | Read the two-digit pressure gauge. |
| `led_state` | Identify the module with a named LED color. |
| `cable_connection` | Identify the port connected to a named cable color. |
| `robot_alignment` | Identify the robot arm target station. |
| `defect_location` | Identify the zone containing the visible magenta defect marker. |
| `lot_code` | Read the lot-code digits on the pod label. |
| `route_order` | Identify the waypoint that follows another waypoint in the route strip. |

## Modeling Considerations

Strong solutions should inspect the image pixels and parse visual structure. Useful approaches include template matching for seven-segment digits, color segmentation for wafer slots and cables, spatial reasoning over fixed station zones, and robustness to lighting or occlusion.

Do not treat this as a text-only multiple-choice problem. The options are balanced and randomized, and the same wording can map to different answers depending on the image.

The hidden score includes worst-group terms across query types, OOD axes, layouts, lighting conditions, clutter levels, and difficulty tiers. Models should avoid specializing only in the easiest normal-light panels.

## Submission

Submit `submission.csv` with exactly these columns:

| Column | Type | Description |
|---|---:|---|
| `id` | string | Question id from `test.csv`. |
| `answer` | string | Predicted answer letter. |

Valid answers are exactly `A`, `B`, `C`, `D`, or `E`.

Example:

| id | answer |
|---|---|
| cwq_e317a3ea7e8b2d | C |
| cwq_b0b8ebd47751d5 | A |
| cwq_a84bcb53bbc532 | E |

Requirements:

- Include exactly one row for every row in `test.csv`.
- Do not include extra columns.
- Do not duplicate or omit any `id`.
- `answer` must be one of `A`, `B`, `C`, `D`, or `E`.
- Do not use private labels, hidden scene traces, external datasets, internet access, or manual labeling of the hidden test set.
