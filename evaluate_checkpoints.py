import argparse
import csv
import os

from torch.utils.data import DataLoader

from config.config_utils import LoadConfig
from dataloader.block_seg_datasets import BlockSegImageDataset
from dataloader.seg_datasets import SegImageDataset
from runnetworks import RunNetworks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints on one validation set without training."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--model", default="Release")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="checkpoint_validation.csv")
    return parser.parse_args()


def build_validation_loader(config, validation_path, batch_size, workers):
    common = {
        "root": [validation_path],
        "mode": "validation",
        "use_aug": False,
    }
    if config["run"]["data_crop"]["use"]:
        dataset = BlockSegImageDataset(
            **common,
            tile_size=config["run"]["data_crop"]["size"],
        )
    else:
        dataset = SegImageDataset(**common)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def main():
    args = parse_args()
    config = LoadConfig()
    config["run"]["type"] = "evaluate"
    config["run"]["model"] = args.model
    config["validation"]["dataset_path"] = args.validation_path
    config["validation"]["batch_size"] = args.batch_size
    config["validation"]["numberworks"] = args.workers

    runner = RunNetworks(config)
    validation_loader = build_validation_loader(
        config,
        args.validation_path,
        args.batch_size,
        args.workers,
    )

    rows = []
    for checkpoint in args.checkpoints:
        checkpoint = os.path.abspath(checkpoint)
        runner._load_checkpoint_flexible(
            runner.model,
            checkpoint,
            name=f"validation:{os.path.basename(checkpoint)}",
        )
        metrics = runner.test_epoch(validation_loader)
        rows.append({"checkpoint": checkpoint, **metrics})

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Checkpoint comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
