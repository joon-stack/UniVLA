import argparse
import os
import sqlite3
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms

from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.datasets import bridge_latent_cache_key


class CacheTransform:
    def __init__(self, action_tokenizer, device):
        self.action_tokenizer = action_tokenizer
        self.device = device
        self.to_tensor = transforms.ToTensor()

    def __call__(self, rlds_batch):
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        initial_arr = rlds_batch["observation"]["image_primary"][0]
        target_arr = rlds_batch["observation"]["image_primary"][-1]
        cache_key = bridge_latent_cache_key(dataset_name, lang, initial_arr, target_arr)

        initial = self.to_tensor(initial_arr)
        target = self.to_tensor(target_arr)
        with torch.no_grad():
            video = torch.stack([initial, target], dim=0).unsqueeze(0).to(self.device)
            indices = self.action_tokenizer.vq_encode(video)["indices"].reshape(-1).detach().cpu().tolist()

        return cache_key, ",".join(str(int(i)) for i in indices)


def load_lam(args, device):
    model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=args.lam_model_dim,
        latent_dim=args.lam_latent_dim,
        num_latents=args.codebook_size,
        patch_size=args.lam_patch_size,
        enc_blocks=args.lam_enc_blocks,
        dec_blocks=args.lam_dec_blocks,
        num_heads=args.lam_num_heads,
        dropout=0.0,
    )
    ckpt = torch.load(args.lam_path, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--data_mix", default="bridge")
    parser.add_argument("--lam_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--shuffle_buffer_size", type=int, default=20000)
    parser.add_argument("--commit_every", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--codebook_size", type=int, default=16)
    parser.add_argument("--lam_model_dim", type=int, default=768)
    parser.add_argument("--lam_latent_dim", type=int, default=128)
    parser.add_argument("--lam_patch_size", type=int, default=14)
    parser.add_argument("--lam_enc_blocks", type=int, default=12)
    parser.add_argument("--lam_dec_blocks", type=int, default=12)
    parser.add_argument("--lam_num_heads", type=int, default=12)
    args = parser.parse_args()

    if not args.lam_path.is_file():
        raise FileNotFoundError(args.lam_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    lam = load_lam(args, device)

    conn = sqlite3.connect(args.output)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS latent_actions (cache_key TEXT PRIMARY KEY, indices TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('lam_path', ?)", (str(args.lam_path),))
    conn.commit()

    dataset = RLDSDataset(
        args.data_root,
        args.data_mix,
        CacheTransform(lam, device),
        resize_resolution=(args.resolution, args.resolution),
        shuffle_buffer_size=args.shuffle_buffer_size,
        train=True,
        image_aug=False,
        training_phase="pre-training",
    )

    start = time.time()
    inserted = 0
    seen = 0
    for cache_key, indices in dataset:
        seen += 1
        cur = conn.execute(
            "INSERT OR IGNORE INTO latent_actions(cache_key, indices) VALUES (?, ?)",
            (cache_key, indices),
        )
        inserted += cur.rowcount
        if seen % args.commit_every == 0:
            conn.commit()
            elapsed = max(time.time() - start, 1e-6)
            print(f"[cache] seen={seen} inserted={inserted} rate={seen / elapsed:.2f}/s", flush=True)
        if args.max_samples and seen >= args.max_samples:
            break

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM latent_actions").fetchone()[0]
    conn.close()
    print(f"[cache] done seen={seen} inserted={inserted} total={total} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
