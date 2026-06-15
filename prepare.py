from pathlib import Path
import hashlib
import shutil

import pandas as pd


ID_COLUMN = "id"
ANSWER_COLUMN = "answer"
PUBLIC_COLUMNS = [
    "id",
    "image",
    "query_type",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "option_E",
]
PRIVATE_GROUP_COLUMNS = [
    "query_type",
    "split_group",
    "ood_axis",
    "layout_family",
    "lighting_condition",
    "clutter_level",
    "difficulty_tier",
    "task_family",
]
SPLIT_SALT = "cleanroom-wafer-pod-vqa-split-v2"
PUBLIC_ID_SALT = "cleanroom-wafer-pod-vqa-public-id-v2"


def _digest(value: str, salt: str, length: int = 14) -> str:
    return hashlib.sha256(f"{salt}::{value}".encode("utf-8")).hexdigest()[:length]


def _find_images(raw: Path) -> dict[str, Path]:
    image_files = {}
    for png in raw.rglob("*.png"):
        image_files[png.name] = png
        parts = png.parts
        if "images" in parts:
            idx = parts.index("images")
            image_files["/".join(parts[idx:])] = png
    if not image_files:
        raise ValueError("No PNG files found. Upload the raw zip containing questions.csv and images/*.png")
    return image_files


def _copy_images(raw: Path, public: Path, questions: pd.DataFrame) -> pd.DataFrame:
    dst = public / "images"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    image_files = _find_images(raw)
    image_map = {}
    for image in sorted(questions["image"].unique()):
        source = image_files.get(image) or image_files.get(Path(image).name)
        if source is None:
            raise ValueError(f"missing image referenced by questions.csv: {image}")
        public_name = f"pod_{_digest(image, PUBLIC_ID_SALT, 16)}.png"
        shutil.copy2(source, dst / public_name)
        image_map[image] = f"images/{public_name}"

    questions = questions.copy()
    questions["image"] = questions["image"].map(image_map)
    return questions


def prepare(raw: Path, public: Path, private: Path) -> None:
    """Create deterministic scene-level train/test splits for image QA."""
    question_files = [raw / "questions.csv", raw / "raw" / "questions.csv"]
    question_files.extend(raw.rglob("questions.csv"))
    question_path = next((path for path in question_files if path.exists()), None)
    if question_path is None:
        raise ValueError("questions.csv is required in the raw dataset upload")

    questions = pd.read_csv(question_path)
    required = set(PUBLIC_COLUMNS + [ANSWER_COLUMN, "scene_id"] + PRIVATE_GROUP_COLUMNS)
    missing = required.difference(questions.columns)
    if missing:
        raise ValueError(f"questions.csv missing columns: {sorted(missing)}")
    if questions[ID_COLUMN].duplicated().any():
        raise ValueError("question ids must be unique")

    questions["public_id"] = questions[ID_COLUMN].map(lambda value: f"cwq_{_digest(str(value), PUBLIC_ID_SALT)}")
    questions[ID_COLUMN] = questions["public_id"]

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    questions = _copy_images(raw, public, questions)

    scene_scores = (
        questions[["scene_id"]]
        .drop_duplicates()
        .assign(split_score=lambda frame: frame["scene_id"].map(lambda value: _digest(str(value), SPLIT_SALT, 16)))
        .sort_values("split_score")
        .reset_index(drop=True)
    )
    test_scenes = set(scene_scores.tail(16)["scene_id"])
    train_mask = ~questions["scene_id"].isin(test_scenes)

    train = questions.loc[train_mask, PUBLIC_COLUMNS + [ANSWER_COLUMN]].copy()
    test_full = questions.loc[~train_mask].copy()
    test = test_full[PUBLIC_COLUMNS].copy()
    answers = test_full[[ID_COLUMN, ANSWER_COLUMN] + PRIVATE_GROUP_COLUMNS].copy()
    sample = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], ANSWER_COLUMN: "A"})

    train = train.sample(frac=1.0, random_state=411).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=512).reset_index(drop=True)
    sample = sample.set_index(ID_COLUMN).loc[test[ID_COLUMN]].reset_index()
    answers = answers.sample(frac=1.0, random_state=613).reset_index(drop=True)

    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)
