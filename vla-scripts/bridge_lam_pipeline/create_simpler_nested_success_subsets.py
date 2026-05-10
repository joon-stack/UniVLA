#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


TASKS = (
    "PutSpoonOnTableClothInScene-v1",
    "PutCarrotOnPlateInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def make_subset(rows_by_task: dict[str, list[dict]], source_manifest: Path, out_root: Path, size: int, overwrite: bool) -> Path:
    out_dir = out_root / f"simpler_mvplam_noeval_nested_success{size}_clean"
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    manifest_rows = []
    summary = {
        "source_manifest": str(source_manifest),
        "nested_from": "simpler_mvplam_noeval_success50_clean",
        "target_per_task": size,
        "target_total": size * len(TASKS),
        "ready": True,
    }

    for task in TASKS:
        available = rows_by_task.get(task, [])
        if len(available) < size:
            raise RuntimeError(f"{task} has only {len(available)} rows; need {size}")

        task_dir = out_dir / task
        video_dir = out_dir / "videos" / task
        task_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)

        selected = available[:size]
        for idx, row in enumerate(selected):
            new_row = dict(row)
            episode_name = f"episode_{idx:06d}"
            clean_episode = task_dir / episode_name
            clean_video = video_dir / f"{episode_name}.mp4"

            raw_episode = Path(row["raw_episode"])
            video_src = Path(row.get("video") or row.get("clean_video", ""))
            if not raw_episode.exists():
                raise FileNotFoundError(raw_episode)
            if video_src and not video_src.exists():
                raise FileNotFoundError(video_src)

            symlink_force(raw_episode, clean_episode)
            if video_src:
                symlink_force(video_src, clean_video)

            new_row["clean_episode"] = str(clean_episode)
            new_row["clean_video"] = str(clean_video)
            manifest_rows.append(new_row)

        summary[task] = {
            "selected_success": len(selected),
            "available_success": len(available),
            "linked_videos": len(selected),
            "target": size,
            "ready": True,
        }

    with (out_dir / "manifest.jsonl").open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    overlap_report = {
        "source_manifest": str(source_manifest),
        "subset_dir": str(out_dir),
        "note": "Nested subset preserving the first N manifest rows per task from the source success50 clean set.",
    }
    with (out_dir / "overlap_report.json").open("w") as f:
        json.dump(overlap_report, f, indent=2)

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-clean",
        type=Path,
        default=Path(
            "/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/"
            "simpler_mvplam_noeval_success50_clean"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = args.source_clean / "manifest.jsonl"
    rows = load_manifest(manifest)
    rows_by_task = defaultdict(list)
    for row in rows:
        task = row["task"]
        if task not in TASKS:
            raise KeyError(f"Unknown task in manifest: {task}")
        rows_by_task[task].append(row)

    for task in TASKS:
        if len(rows_by_task[task]) < max(args.sizes):
            raise RuntimeError(f"{task} has {len(rows_by_task[task])} rows, need {max(args.sizes)}")

    outputs = []
    for size in args.sizes:
        outputs.append(make_subset(rows_by_task, manifest, args.out_root, size, args.overwrite))

    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
