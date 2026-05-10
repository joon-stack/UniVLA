#!/usr/bin/env python3
"""Build a UniVLA finetune cache from a LeRobot v3 dataset.

Run this with a LeRobot-capable environment, not the UniVLA training env:

  /path/to/lerobot/.venv/bin/python \
    vla-scripts/bridge_lam_pipeline/build_lerobot_univla_cache.py \
    --repo-id joon-stack/pick_place_nanobanana \
    --output-dir /path/to/data/lerobot_univla_cache/pick_place_nanobanana_top_ws10
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image

try:
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - Pillow < 9
    _BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="joon-stack/pick_place_nanobanana")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", default="lerobot_pick_place_nanobanana")
    parser.add_argument("--image-key", default="observation.images.top")
    parser.add_argument("--window-size", default=10, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--max-episodes", default=None, type=int)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _download(repo_id: str, filename: str, cache_dir: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir))


def _list_repo_files(repo_id: str) -> List[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_parquets(repo_id: str, files: Iterable[str], cache_dir: str | None = None):
    import pandas as pd

    frames = [pd.read_parquet(_download(repo_id, filename, cache_dir)) for filename in files]
    if not frames:
        raise ValueError("No parquet files matched.")
    return pd.concat(frames, ignore_index=False)


def _task_map(tasks_df) -> Dict[int, str]:
    if "task" in tasks_df.columns and "task_index" in tasks_df.columns:
        return {int(row["task_index"]): str(row["task"]) for _, row in tasks_df.iterrows()}
    if "task_index" in tasks_df.columns:
        return {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}
    return {0: str(tasks_df.index[0])}


def _format_data_path(info: Dict[str, Any], chunk_index: int, file_index: int) -> str:
    return info["data_path"].format(chunk_index=int(chunk_index), file_index=int(file_index))


def _format_video_path(info: Dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> str:
    return info["video_path"].format(
        video_key=video_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )


def _episode_task(row, data_rows, task_by_index: Dict[int, str]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) > 0:
        return str(tasks[0])
    if len(data_rows) > 0 and "task_index" in data_rows.columns:
        return task_by_index.get(int(data_rows.iloc[0]["task_index"]), "")
    return ""


def _write_dataset_statistics(
    output_dir: Path,
    dataset_name: str,
    info: Dict[str, Any],
    stats: Dict[str, Any],
    num_episodes: int,
    num_transitions: int,
) -> Dict[str, Any]:
    action_stats = stats["action"]
    action_dim = len(action_stats["mean"])
    dataset_statistics = {
        dataset_name: {
            "action": {
                "mean": action_stats["mean"],
                "std": action_stats["std"],
                "min": action_stats["min"],
                "max": action_stats["max"],
                "q01": action_stats["q01"],
                "q99": action_stats["q99"],
                "mask": [True] * action_dim,
            },
            "num_transitions": int(num_transitions),
            "num_trajectories": int(num_episodes),
        }
    }
    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    with open(output_dir / "lerobot_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return dataset_statistics


def _decode_and_save_frames(
    repo_id: str,
    info: Dict[str, Any],
    episodes_df,
    output_dir: Path,
    image_key: str,
    image_size: int,
    cache_dir: str | None = None,
) -> None:
    import av

    fps = float(info["fps"])
    video_chunk_col = f"videos/{image_key}/chunk_index"
    video_file_col = f"videos/{image_key}/file_index"
    from_ts_col = f"videos/{image_key}/from_timestamp"
    if video_chunk_col not in episodes_df.columns or video_file_col not in episodes_df.columns:
        raise KeyError(f"Image key {image_key!r} not present in episode metadata.")

    grouped = episodes_df.groupby([video_chunk_col, video_file_col], sort=True)
    for (chunk_index, file_index), group in grouped:
        video_file = _format_video_path(info, image_key, chunk_index, file_index)
        video_path = _download(repo_id, video_file, cache_dir)
        ranges = []
        for _, row in group.sort_values(from_ts_col).iterrows():
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            start = int(round(float(row[from_ts_col]) * fps))
            ranges.append(
                {
                    "episode_index": episode_index,
                    "start": start,
                    "end": start + length,
                    "length": length,
                    "saved": 0,
                }
            )

        print(f"[decode] {video_file} -> {len(ranges)} episodes")
        current = 0
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                while current < len(ranges) and frame_index >= ranges[current]["end"]:
                    current += 1
                if current >= len(ranges):
                    break
                current_range = ranges[current]
                if frame_index < current_range["start"]:
                    continue
                local_index = frame_index - current_range["start"]
                episode_dir = output_dir / "frames" / f"episode_{current_range['episode_index']:06d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                out_path = episode_dir / f"{local_index:06d}.jpg"
                if not out_path.exists():
                    image = frame.to_image().convert("RGB").resize((image_size, image_size), resample=_BICUBIC)
                    image.save(out_path, quality=95)
                current_range["saved"] += 1

        missing = [r for r in ranges if r["saved"] != r["length"]]
        if missing:
            raise RuntimeError(
                "Video decode did not produce the expected number of frames: "
                + ", ".join(
                    f"episode={r['episode_index']} saved={r['saved']} expected={r['length']}" for r in missing[:5]
                )
            )


def _write_windows(
    output_dir: Path,
    dataset_name: str,
    episodes_df,
    data_df,
    task_by_index: Dict[int, str],
    window_size: int,
) -> int:
    windows_path = output_dir / "windows.jsonl"
    window_len = window_size + 1
    written = 0
    with open(windows_path, "w") as f:
        for _, row in episodes_df.sort_values("episode_index").iterrows():
            episode_index = int(row["episode_index"])
            episode_rows = data_df[data_df["episode_index"] == episode_index].sort_values("frame_index")
            if episode_rows.empty:
                raise RuntimeError(f"No data rows found for episode {episode_index}")
            actions = np.stack(episode_rows["action"].to_numpy()).astype(np.float32)
            if len(actions) != int(row["length"]):
                raise RuntimeError(
                    f"Episode {episode_index} action length mismatch: {len(actions)} vs metadata {int(row['length'])}"
                )
            task = _episode_task(row, episode_rows, task_by_index)
            if len(actions) < window_len:
                continue

            for start in range(0, len(actions) - window_size):
                frame_paths = [
                    f"frames/episode_{episode_index:06d}/{frame_index:06d}.jpg"
                    for frame_index in range(start, start + window_len)
                ]
                record = {
                    "dataset_name": dataset_name,
                    "episode_index": episode_index,
                    "start": start,
                    "task": task,
                    "frame_paths": frame_paths,
                    "actions": actions[start : start + window_len].tolist(),
                }
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
    return written


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists and is not empty; pass --overwrite to rebuild.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_files = _list_repo_files(args.repo_id)
    info = _read_json(_download(args.repo_id, "meta/info.json", args.hf_cache_dir))
    stats = _read_json(_download(args.repo_id, "meta/stats.json", args.hf_cache_dir))
    tasks_df = _read_parquets(args.repo_id, ["meta/tasks.parquet"], args.hf_cache_dir)
    episode_files = sorted(f for f in repo_files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    episodes_df = _read_parquets(args.repo_id, episode_files, args.hf_cache_dir).sort_values("episode_index")
    if args.max_episodes is not None:
        episodes_df = episodes_df.head(args.max_episodes)

    data_files = sorted(
        {
            _format_data_path(info, row["data/chunk_index"], row["data/file_index"])
            for _, row in episodes_df.iterrows()
        }
    )
    data_df = _read_parquets(args.repo_id, data_files, args.hf_cache_dir)
    selected_episode_ids = set(int(x) for x in episodes_df["episode_index"].to_numpy())
    data_df = data_df[data_df["episode_index"].map(lambda x: int(x) in selected_episode_ids)]

    _decode_and_save_frames(
        args.repo_id,
        info,
        episodes_df,
        output_dir,
        args.image_key,
        args.image_size,
        args.hf_cache_dir,
    )
    windows = _write_windows(output_dir, args.dataset_name, episodes_df, data_df, _task_map(tasks_df), args.window_size)
    _write_dataset_statistics(
        output_dir,
        args.dataset_name,
        info,
        stats,
        num_episodes=len(episodes_df),
        num_transitions=len(data_df),
    )

    action_feature = info["features"]["action"]
    metadata = {
        "format": "univla_lerobot_window_cache_v1",
        "repo_id": args.repo_id,
        "dataset_name": args.dataset_name,
        "image_key": args.image_key,
        "resize_strategy": "resize_naive",
        "image_size": args.image_size,
        "window_size": args.window_size,
        "outer_window_size": args.window_size + 1,
        "normalization_type": "bounds_q99",
        "action_dim": action_feature["shape"][0],
        "action_names": action_feature.get("names", []),
        "num_episodes": len(episodes_df),
        "num_transitions": len(data_df),
        "num_windows": windows,
        "task": next(iter(_task_map(tasks_df).values()), ""),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[done] wrote {windows} windows to {output_dir}")


if __name__ == "__main__":
    main()
