#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA-hyper}"
SIMPLER_ROOT="${SIMPLER_ROOT:-${ROOT}/third_party/SimplerEnv-maniskill3}"
PYTHON="${PYTHON:-${ROOT}/.venvs/simplerenv-eval-py310/bin/python}"

# Continue the existing success50 collection. This preserves the original raw
# rollouts and fingerprint logs, so new attempts start after the old attempts.
export SAVE_TRAJECTORY_DIR="${SAVE_TRAJECTORY_DIR:-${ROOT}/data/simpler_mvplam_noeval_success50_rollouts_npz}"
export RECORD_DIR="${RECORD_DIR:-${ROOT}/eval_logs/mvplam_noeval_success50_with_video}"
export FINGERPRINT_DIR="${FINGERPRINT_DIR:-${ROOT}/data/simpler_mvplam_noeval_success50_fingerprints}"

# Build a merged 100-success dataset into new output directories.
export CLEAN_DIR="${CLEAN_DIR:-${ROOT}/data/simpler_mvplam_noeval_success100_clean}"
export RLDS_DIR="${RLDS_DIR:-${ROOT}/data/simpler_mvplam_noeval_success100_rlds}"
export SUCCESS75_CLEAN_DIR="${SUCCESS75_CLEAN_DIR:-${ROOT}/data/simpler_mvplam_noeval_success75_clean}"
export SUCCESS75_RLDS_DIR="${SUCCESS75_RLDS_DIR:-${ROOT}/data/simpler_mvplam_noeval_success75_rlds}"
export SUCCESS_TARGET="${SUCCESS_TARGET:-100}"
export CHUNK_EPISODES="${CHUNK_EPISODES:-25}"
export MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-3000}"
export BUILD_CLEAN="${BUILD_CLEAN:-1}"
export BUILD_RLDS="${BUILD_RLDS:-1}"

echo "[success100-merge] continuing raw collection from: ${SAVE_TRAJECTORY_DIR}"
echo "[success100-merge] using fingerprints from:       ${FINGERPRINT_DIR}"
echo "[success100-merge] writing clean dataset to:      ${CLEAN_DIR}"
echo "[success100-merge] writing RLDS dataset to:       ${RLDS_DIR}"
echo "[success100-merge] writing success75 clean to:    ${SUCCESS75_CLEAN_DIR}"
echo "[success100-merge] writing success75 RLDS to:     ${SUCCESS75_RLDS_DIR}"

"${SIMPLER_ROOT}/collect_mvplam_noeval_success50.sh"

"${PYTHON}" - <<'PY'
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path("/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong")
sys.path.insert(0, str(root / "third_party/SimplerEnv-maniskill3"))

from fingerprint_utils import find_collision, load_jsonl

old_manifest = root / "data/simpler_mvplam_noeval_success50_clean/manifest.jsonl"
success100_clean = Path(os.environ["CLEAN_DIR"])
success75_clean = Path(os.environ["SUCCESS75_CLEAN_DIR"])
new_manifest = success100_clean / "manifest.jsonl"
summary_path = success100_clean / "summary.json"
eval_fingerprints_path = Path(os.environ["FINGERPRINT_DIR"]) / "eval_fingerprints.jsonl"

TASKS = [
    "PutSpoonOnTableClothInScene-v1",
    "PutCarrotOnPlateInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
]

def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def replace_symlink(link_path: Path, target: Path, *, target_is_directory: bool) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace non-symlink directory: {link_path}")
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link_path, target_is_directory=target_is_directory)

def clear_generated_links(directory: Path) -> None:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_symlink():
            path.unlink()

def load_fingerprint(row: dict) -> dict:
    fp_path = row.get("fingerprint")
    if not fp_path:
        raise RuntimeError(f"Missing fingerprint path for {row['raw_episode']}")
    with Path(fp_path).open() as f:
        return json.load(f)

def fingerprint_key(fp: dict) -> tuple:
    return (fp.get("env_id"), fp.get("state_hash"), fp.get("image_sha256"))

old_rows = load_jsonl(old_manifest)
new_rows = load_jsonl(new_manifest)
eval_fingerprints = load_jsonl(eval_fingerprints_path)

old_raw = {row["raw_episode"] for row in old_rows}
new_raw = [row["raw_episode"] for row in new_rows]
new_raw_set = set(new_raw)
missing_old = sorted(old_raw - new_raw_set)
duplicate_raw = sorted(path for path, count in Counter(new_raw).items() if count > 1)

by_task = Counter(row["task"] for row in new_rows)
bad_counts = {task: count for task, count in by_task.items() if count != 100}

fingerprints = []
missing_fingerprint = []
eval_collisions = []
for row in new_rows:
    try:
        fp = load_fingerprint(row)
    except RuntimeError:
        missing_fingerprint.append(row["raw_episode"])
        continue
    collision = find_collision(fp, eval_fingerprints)
    if collision is not None:
        eval_collisions.append(
            {
                "episode": row["raw_episode"],
                "collision_type": collision["type"],
                "collision_env_id": collision["match"].get("env_id"),
                "collision_seed": collision["match"].get("seed"),
                "collision_episode_id": collision["match"].get("episode_id"),
            }
        )
    fingerprints.append((fingerprint_key(fp), row["raw_episode"]))

fingerprint_counts = Counter(key for key, _ in fingerprints)
duplicate_fingerprints = [
    {"key": key, "episodes": [episode for fp_key, episode in fingerprints if fp_key == key]}
    for key, count in fingerprint_counts.items()
    if count > 1
]

print("[success100-merge] selected counts:", dict(sorted(by_task.items())))
print("[success100-merge] total:", len(new_rows))
print("[success100-merge] summary:", summary_path)

errors = []
if missing_old:
    errors.append(f"old success50 episodes missing from success100: {len(missing_old)}")
if duplicate_raw:
    errors.append(f"duplicate raw episodes in success100 manifest: {len(duplicate_raw)}")
if bad_counts:
    errors.append(f"task counts are not 100: {bad_counts}")
if missing_fingerprint:
    errors.append(f"rows missing fingerprints: {len(missing_fingerprint)}")
if duplicate_fingerprints:
    errors.append(f"duplicate init fingerprints in success100: {len(duplicate_fingerprints)}")
if eval_collisions:
    errors.append(f"success100 overlaps eval fingerprints: {len(eval_collisions)}")

if errors:
    for error in errors:
        print("[success100-merge][ERROR]", error)
    raise SystemExit(1)

print("[success100-merge] verification passed: old 50 included, 100/task, no raw/fingerprint duplicates")

rows_by_task = defaultdict(list)
old_raw_by_task = defaultdict(set)
for row in new_rows:
    rows_by_task[row["task"]].append(row)
for row in old_rows:
    old_raw_by_task[row["task"]].add(row["raw_episode"])

success75_rows = []
success75_summary = {}
success75_report = {
    "source_success50_manifest": str(old_manifest),
    "source_success100_manifest": str(new_manifest),
    "eval_fingerprints": str(eval_fingerprints_path),
    "errors": [],
}

for task in TASKS:
    task_rows = rows_by_task[task]
    old_task_rows = [row for row in task_rows if row["raw_episode"] in old_raw_by_task[task]]
    added_task_rows = [row for row in task_rows if row["raw_episode"] not in old_raw_by_task[task]]
    selected = old_task_rows + added_task_rows[:25]
    if len(old_task_rows) != 50:
        success75_report["errors"].append(f"{task}: expected 50 old rows, got {len(old_task_rows)}")
    if len(added_task_rows) < 50:
        success75_report["errors"].append(f"{task}: expected 50 added rows in success100, got {len(added_task_rows)}")
    if len(selected) != 75:
        success75_report["errors"].append(f"{task}: expected 75 selected rows, got {len(selected)}")
    success75_rows.extend((task, row) for row in selected)
    success75_summary[task] = {
        "selected_success": len(selected),
        "existing_success50": len(old_task_rows),
        "added_from_success100": min(25, len(added_task_rows)),
        "available_added_from_success100": len(added_task_rows),
        "target": 75,
        "ready": len(selected) == 75 and len(old_task_rows) == 50 and len(added_task_rows) >= 25,
    }

raw75 = [row["raw_episode"] for _, row in success75_rows]
duplicate75_raw = sorted(path for path, count in Counter(raw75).items() if count > 1)
if duplicate75_raw:
    success75_report["errors"].append(f"duplicate raw episodes in success75: {len(duplicate75_raw)}")

fp75 = []
eval75_collisions = []
for _, row in success75_rows:
    fp = load_fingerprint(row)
    collision = find_collision(fp, eval_fingerprints)
    if collision is not None:
        eval75_collisions.append(row["raw_episode"])
    fp75.append((fingerprint_key(fp), row["raw_episode"]))
duplicate75_fp = [
    {"key": key, "episodes": [episode for fp_key, episode in fp75 if fp_key == key]}
    for key, count in Counter(key for key, _ in fp75).items()
    if count > 1
]
if duplicate75_fp:
    success75_report["errors"].append(f"duplicate init fingerprints in success75: {len(duplicate75_fp)}")
if eval75_collisions:
    success75_report["errors"].append(f"success75 overlaps eval fingerprints: {len(eval75_collisions)}")

success75_clean.mkdir(parents=True, exist_ok=True)
manifest75_path = success75_clean / "manifest.jsonl"
summary75_path = success75_clean / "summary.json"
report75_path = success75_clean / "overlap_report.json"

manifest75 = []
for task in TASKS:
    clear_generated_links(success75_clean / task)
    clear_generated_links(success75_clean / "videos" / task)

for idx_by_task, (task, row) in enumerate(success75_rows):
    task_index = sum(1 for prev_task, _ in success75_rows[:idx_by_task] if prev_task == task)
    clean_name = f"episode_{task_index:06d}"
    clean_episode = success75_clean / task / clean_name
    replace_symlink(clean_episode, Path(row["raw_episode"]).resolve(), target_is_directory=True)

    clean_video = None
    video_path = row.get("video")
    if video_path and Path(video_path).is_file():
        clean_video_path = success75_clean / "videos" / task / f"{clean_name}.mp4"
        replace_symlink(clean_video_path, Path(video_path).resolve(), target_is_directory=False)
        clean_video = str(clean_video_path)

    new_row = dict(row)
    new_row["clean_episode"] = str(clean_episode)
    new_row["clean_video"] = clean_video
    manifest75.append(new_row)

success75_summary["ready"] = not success75_report["errors"] and all(
    value["ready"] for value in success75_summary.values() if isinstance(value, dict)
)
success75_summary["total_selected_success"] = len(manifest75)
success75_summary["target_total"] = 75 * len(TASKS)

with manifest75_path.open("w") as f:
    for row in manifest75:
        f.write(json.dumps(row) + "\n")
with summary75_path.open("w") as f:
    json.dump(success75_summary, f, indent=2)
success75_report["duplicate_raw"] = duplicate75_raw
success75_report["duplicate_fingerprints"] = duplicate75_fp
success75_report["eval_collisions"] = eval75_collisions
with report75_path.open("w") as f:
    json.dump(success75_report, f, indent=2)

print("[success75] selected counts:", dict(sorted(Counter(row["task"] for row in manifest75).items())))
print("[success75] total:", len(manifest75))
print("[success75] summary:", summary75_path)
if success75_report["errors"]:
    for error in success75_report["errors"]:
        print("[success75][ERROR]", error)
    raise SystemExit(1)
print("[success75] verification passed: old 50 + first 25 added/task, no eval/raw/fingerprint overlap")
PY

if [[ "${BUILD_RLDS}" == "1" ]]; then
  "${PYTHON}" "${SIMPLER_ROOT}/convert_success25_to_simpler_rlds.py" \
    --manifest "${SUCCESS75_CLEAN_DIR}/manifest.jsonl" \
    --out-dir "${SUCCESS75_RLDS_DIR}" \
    --target 75 \
    --overwrite
fi
