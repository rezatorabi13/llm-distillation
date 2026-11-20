"""Record the teacher's next-token distribution for a prompt (input to the soft-label figure).

Example:
    python scripts/soft_labels.py --prompt "Once upon a time, there was a little"
"""

from __future__ import annotations

import argparse

import torch

from lm_distill.models import load_teacher, load_tokenizer
from lm_distill.utils import get_device, save_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--teacher", default="roneneldan/TinyStories-33M")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="results/soft_labels.json")
    args = parser.parse_args()

    device = get_device()
    tokenizer = load_tokenizer(args.teacher)
    teacher = load_teacher(args.teacher, device)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    probs = teacher(**inputs).logits[0, -1].float().softmax(-1)

    # Entropy of the full distribution, in bits: how much information one soft label carries vs a one-hot label
    entropy_bits = -(probs * probs.clamp_min(1e-12).log2()).sum().item()
    top = probs.topk(args.top_k)
    record = {
        "prompt": args.prompt,
        "entropy_bits": entropy_bits,
        "top": [{"token": tokenizer.decode(i), "prob": p}
                for i, p in zip(top.indices.tolist(), top.values.tolist(), strict=True)],
    }
    save_json(record, args.out)
    for row in record["top"]:
        print(f"{row['token']!r:>12}  {row['prob']:.1%}")


if __name__ == "__main__":
    main()
