# Dataset Description

## Overview

This synthetic dataset contains rendered cleanroom inspection panels for wafer pod handling stations. Each PNG image shows a single inspection moment with a wafer cassette, slot map, digital gauge, module LEDs, robot arm, cable ports, route strip, lot code, and visual defect marker.

The dataset is a multimodal image question-answering benchmark. Each row pairs an image with a natural-language multiple-choice question and five answer options. Correct answers are generated from the hidden scene trace, not from manual annotation. Questions require visual inspection, spatial reasoning, small-digit reading, color/port matching, and robustness to lighting, glare, clutter, and partial occlusion.

The data is fully synthetic. It contains no real cleanroom, semiconductor, factory, customer, or proprietary equipment data.

## File Structure

| File or Folder | Rows / Count | Description |
|---|---:|---|
| `questions.csv` | 352 | Raw QA table with labels and hidden evaluation groups. The prepare script splits this into public train/test and private answers. |
| `images/` | 44 PNG files | Rendered inspection-panel images referenced by the `image` column. |

The prepare script copies all PNGs into `public/images/`, places labeled rows from 28 scenes into `public/train.csv`, places unlabeled rows from 16 held-out scenes into `public/test.csv`, and stores labels plus hidden evaluation groups in `private/answers.csv`.

## Public Columns After Preparation

| Column | Type | Description |
|---|---:|---|
| `id` | string | Unique question id, formatted as `cw_sceneNNN_qTT`. |
| `image` | string | Relative path to the PNG under `public/`, for example `images/scene_000.png`. |
| `query_type` | string | Question category. |
| `question` | string | Natural-language multiple-choice question. |
| `option_A` | string | Answer option A. |
| `option_B` | string | Answer option B. |
| `option_C` | string | Answer option C. |
| `option_D` | string | Answer option D. |
| `option_E` | string | Answer option E. |
| `answer` | string | Correct answer letter. Present only in `train.csv`. |

## Query Types

| Query Type | Description |
|---|---|
| `slot_color` | Identify which cassette slot contains a specified colored wafer. |
| `gauge_reading` | Read the two-digit digital pressure gauge. |
| `led_state` | Identify which module shows a specified status LED color. |
| `cable_connection` | Identify which port is connected to a specified cable color. |
| `robot_alignment` | Identify the station targeted by the robot arm. |
| `defect_location` | Identify the zone containing a magenta particle/defect marker. |
| `lot_code` | Read the two-digit lot code displayed on the pod label. |
| `route_order` | Determine which waypoint follows another in the route strip. |

## Hidden Evaluation Groups

The private answers include hidden grouping columns used by the grader:

| Group | Values |
|---|---|
| `query_type` | The eight query types listed above. |
| `split_group` | `standard_holdout`, `ood_low_light`, `ood_glare`, `ood_occlusion`, `ood_dense_clutter`. |
| `ood_axis` | `standard`, `low_light`, `glare`, `occlusion`, `dense_clutter`. |
| `layout_family` | `left_cassette`, `right_cassette`, `wide_ports`, `compact_panel`. |
| `lighting_condition` | `normal`, `low_light`, `glare`, `blue_cast`. |
| `clutter_level` | `low`, `medium`, `high`. |
| `task_family` | `wafer_tracking`, `instrument_reading`, `connection_audit`, `route_memory`, `defect_audit`. |

The score rewards balanced performance across these hidden groups, so solutions that only answer the easiest visible cases will not score well.

## Image Characteristics

- Resolution: 720x480 RGB PNG.
- Scene count: 44 inspection panels.
- QA rows: 8 questions per image, 352 total.
- Train/test split: scene-level split, so no question from a held-out scene appears in training.
- Visual variation: layout family, illumination, glare, occluding tape strips, particle clutter, scanner noise, and small display digits.
- Important: the answer must be inferred from the image and question/options. Hidden scene traces are not present in the public data.

## License and Source

License: CC0 1.0 Public Domain.

Source: Synthetic dataset generated locally with a deterministic renderer and fixed random seed. No external source data was used.

