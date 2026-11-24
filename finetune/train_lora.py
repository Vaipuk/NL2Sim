"""
finetune/train_lora.py
-----------------------
LoRA fine-tune of an open-weight model on the NL->config dataset.

Tuned for a Quadro RTX 8000 (48 GB, Turing / sm_75), which constrains two
things that most recipes get wrong on this card:

  * Turing has no bfloat16. Training must be fp16 with loss scaling, which
    is what fp16=True sets up. Copying a bf16 recipe here fails or silently
    produces garbage gradients.
  * FlashAttention-2 needs Ampere or newer, so attention falls back to
    PyTorch SDPA. Slower, works fine.

8B weights in fp16 are ~16 GB, leaving room for LoRA optimizer state and
8192-token activations with gradient checkpointing on.

    python finetune/train_lora.py                         # defaults below
    python finetune/train_lora.py --epochs 3 --lora-r 32
    python finetune/train_lora.py --max-seq-len 4096      # if memory is tight

Run finetune/prepare_local_data.py first; it writes the train/val files and
keeps the evaluation test set out of them.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import json
import os
from pathlib import Path

# Must be set before torch is imported. The first OOM left 2.66 GB reserved
# but unallocated; expandable segments reuse that instead of stranding it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs" / "ft_data"
DEFAULT_OUT = ROOT / "outputs" / "lora_qwen3_8b"


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune for NL2Sim config generation.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # 8192 covers every sample: the longest is 7986 tokens. 4096 would truncate
    # about half of them, and truncation lands on the END of the config, which
    # teaches the model to stop early.
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-liger", action="store_true",
                    help="disable the fused cross-entropy kernel (expect OOM at 8192)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    print(f"  gpu        : {torch.cuda.get_device_name(0) if cap != (0,0) else 'NONE'}  sm_{cap[0]}{cap[1]}")
    if cap == (0, 0):
        raise SystemExit("no CUDA device visible")
    supports_bf16 = cap[0] >= 8
    print(f"  bf16       : {'yes' if supports_bf16 else 'no (Turing) -> using fp16'}")
    print("  attention  : sdpa")

    # Qwen3's vocabulary is ~152k, so plain cross-entropy builds a
    # seq_len x 152k logits tensor plus its gradient -- 7.5 GB at 8192 tokens,
    # which is what OOMs a 48 GB card on the first backward pass. Liger's
    # fused linear cross-entropy computes the loss in chunks and never
    # materialises it. Without Liger the only lever left is a shorter
    # sequence, and 4096 would truncate 52% of samples at the end of the
    # config -- teaching the model to stop early.
    use_liger = importlib.util.find_spec("liger_kernel") is not None and not args.no_liger
    print(f"  liger      : {'yes (fused cross-entropy)' if use_liger else 'NO -- expect OOM above ~6144 tokens'}")

    data = Path(args.data_dir)
    ds = load_dataset("json", data_files={"train": str(data / "train.jsonl"),
                                          "validation": str(data / "val.jsonl")})
    print(f"  train/val  : {len(ds['train'])} / {len(ds['validation'])}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if supports_bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False          # incompatible with gradient checkpointing

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # TRL's SFTConfig takes warmup_steps, not warmup_ratio, so work out 10%
    # of the run ourselves rather than hard-coding a step count that stops
    # being 10% the moment epochs or batch size change.
    steps_per_epoch = math.ceil(len(ds["train"]) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * 0.1))
    print(f"  schedule   : {total_steps} optimizer steps, {warmup} warmup")

    cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # Eval defaults to batch 8, which at this sequence length runs out of
        # memory on a 48 GB card at the first evaluation -- an hour in.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_len,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=supports_bf16,
        fp16=not supports_bf16,
        use_liger_kernel=use_liger,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # Record what produced this adapter -- seeds and hyperparameters are the
    # part of a training run that is impossible to reconstruct afterwards.
    (Path(args.out_dir) / "run_config.json").write_text(
        json.dumps({**vars(args), "bf16": supports_bf16,
                    "gpu": torch.cuda.get_device_name(0)}, indent=2),
        encoding="utf-8")
    print(f"\n  adapter saved -> {args.out_dir}")
    print(f"  serve it with: vllm serve {args.model} --enable-lora "
          f"--lora-modules nl2sim={args.out_dir} --dtype float16 --port 8000")


if __name__ == "__main__":
    main()
