"""Sample stories from the teacher and/or saved students.

Examples:
    python scripts/generate.py --models teacher checkpoints/distill_n30000_T1_a1_s0 \
        --prompt "Once upon a time, there was a little dog named"
    python scripts/generate.py --models teacher checkpoints/distill_n30000_T1_a1_s0 \
        checkpoints/baseline_n30000_s0 --out results/samples.json
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from lm_distill.models import load_teacher, load_tokenizer
from lm_distill.utils import get_device, save_json

DEFAULT_PROMPTS = ["Once upon a time, there was a little dog named", "Lily was sad because her toy"]


@torch.no_grad()
def sample(model, tokenizer, prompt: str, max_new_tokens: int, seed: int, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=40, temperature=0.8,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="'teacher' and/or checkpoint directories")
    parser.add_argument("--prompt", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--teacher", default="roneneldan/TinyStories-33M")
    parser.add_argument("--max-new-tokens", type=int, default=45)  # the student's context is 64 tokens
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="also write the samples to this JSON file")
    args = parser.parse_args()

    device = get_device()
    tokenizer = load_tokenizer(args.teacher)
    samples = []
    for prompt in args.prompt:
        print("=" * 80 + f"\nPROMPT: {prompt}")
        for name in args.models:
            model = (load_teacher(args.teacher, device) if name == "teacher"
                     else AutoModelForCausalLM.from_pretrained(name).to(device).eval())
            text = sample(model, tokenizer, prompt, args.max_new_tokens, args.seed, device)
            samples.append({"prompt": prompt, "model": name, "text": text})
            print(f"\n[{name}]\n{text}")
        print()
    if args.out:
        save_json(samples, args.out)


if __name__ == "__main__":
    main()
