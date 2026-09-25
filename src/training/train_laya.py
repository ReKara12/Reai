"""PyTorch Supervised Fine-Tuning (SFT) training script for Laya ModernBERT on OS reflex tasks.

Runs 100% in the background on GPU (RTX 3050 4GB VRAM) without touching the mouse or desktop windows.
"""

import os
import sys
import json
import time
import math
import random
import logging
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import laya
from laya.common import build_sequence, serialize_state, QTYPES

# Immediate flushing file handler
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        FlushFileHandler("training.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("train_laya")


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_sample_tensors(
    tok,
    sample: Dict[str, Any],
    max_len: int = 512,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    """Converts a synthetic OS reflex state sample into tokenized tensors for Laya."""
    compact_state = {
        "goal": sample["instruction"],
        "window": sample["active_window"],
        "elements": sample["elements"],
        "history": sample.get("history", [])[-3:],
    }

    # 1. Action Type Question
    action_criteria = {
        "CLICK": "Single left mouse click on targeted control",
        "DOUBLE_CLICK": "Double left click to open or select",
        "HOTKEY": "Execute keyboard shortcut combination",
        "NAVIGATE_URL": "Open URL in web browser",
        "FOCUS_WINDOW": "Bring target application to foreground",
        "TYPE": "Type input text into active edit field",
        "CALL_LLM": "Generative text synthesis or content drafting",
        "SCROLL": "Scroll viewport",
        "WAIT": "Pause briefly for UI asynchronous transition",
    }
    action_q = {
        "t": "choice",
        "ins": f"What discrete OS action should be performed for goal '{sample['instruction']}'?",
        "crit": action_criteria,
    }
    
    target_action = sample["target_action"]
    action_keys = list(action_criteria.keys())
    target_action_idx = action_keys.index(target_action) if target_action in action_keys else 0

    # 2. Completion Question (noul: false=0, true=1)
    comp_q = {
        "t": "noul",
        "ins": f"Has the goal '{sample['instruction']}' been fully completed based on current state?",
        "crit": {},
    }
    target_comp_idx = 1 if sample.get("is_completed", 0.0) >= 0.5 else 0

    questions_and_targets = [
        (action_q, target_action_idx),
        (comp_q, target_comp_idx),
    ]

    tensors = []
    for q, target_idx in questions_and_targets:
        ids, markers = build_sequence(tok, compact_state, q, max_len=max_len)
        if not markers:
            continue
        
        qtype_val = QTYPES.get(q["t"], 0)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        marker_pos = torch.tensor(markers, dtype=torch.long)
        marker_mask = torch.ones(len(markers), dtype=torch.bool)
        
        tensors.append((input_ids, attention_mask, marker_pos, marker_mask, qtype_val, target_idx))

    return tensors


def collate_fn(batch_items, pad_token_id: int):
    """Pads variable-length sequences and markers into a batched tensor dict."""
    max_seq_len = max(item[0].size(0) for item in batch_items)
    max_markers = max(item[2].size(0) for item in batch_items)
    batch_size = len(batch_items)

    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    marker_pos = torch.zeros((batch_size, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, max_markers), dtype=torch.bool)
    qtypes = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, (inp, att, mpos, mmask, qt, tgt) in enumerate(batch_items):
        seq_len = inp.size(0)
        n_m = mpos.size(0)

        input_ids[i, :seq_len] = inp
        attention_mask[i, :seq_len] = att
        marker_pos[i, :n_m] = mpos
        marker_mask[i, :n_m] = mmask
        qtypes[i] = qt
        targets[i] = tgt

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtypes,
        "targets": targets,
    }


def evaluate(agent, val_items, tok, device, batch_size=8, freeze_encoder: bool = True) -> Tuple[float, float]:
    """Evaluates loss and accuracy on validation dataset."""
    agent.model.eval()
    if not val_items:
        return 0.0, 0.0

    pad_id = tok.pad_token_id or 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(val_items), batch_size):
            batch = collate_fn(val_items[i : i + batch_size], pad_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, _ = agent.model(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=freeze_encoder)
                loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += len(targets)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    return avg_loss, accuracy


def train(
    train_path: str = "data/train_os_reflex.json",
    eval_path: str = "data/eval_os_reflex.json",
    output_dir: str = "models/laya-os-reflex",
    epochs: int = 25,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    freeze_encoder: bool = True,
    max_duration_hours: float = 6.0,
):
    """Executes the offline GPU training run for Laya."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info("Initializing SFT training on device: %s (%s)", device, device_name)

    train_data = load_dataset(train_path)
    val_data = load_dataset(eval_path)
    logger.info("Loaded raw dataset: %d train samples, %d eval samples.", len(train_data), len(val_data))

    agent = laya.Agent(
        model_id_or_path="convaiinnovations/laya",
        subfolder="typed-decisions",
        device=device,
        fast=False,
    )
    tok = agent.tok
    pad_id = tok.pad_token_id or 0

    # 1. Pre-tokenize all items ONCE to eliminate CPU tokenization bottleneck in epochs
    logger.info("Pre-tokenizing training and validation samples...")
    t_tok_start = time.time()
    train_items = []
    for sample in train_data:
        train_items.extend(prepare_sample_tensors(tok, sample))

    val_items = []
    for sample in val_data:
        val_items.extend(prepare_sample_tensors(tok, sample))

    logger.info(
        "Tokenization complete in %.1fs: %d training pairs, %d validation pairs.",
        time.time() - t_tok_start,
        len(train_items),
        len(val_items),
    )

    model = agent.model
    if freeze_encoder:
        logger.info("Freezing ModernBERT backbone (training 26.5M decision head params in 1.9GB VRAM)...")
        for p in model.encoder.parameters():
            p.requires_grad = False
    else:
        logger.info("Training full model (backbone + head)...")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %d (Total: %d)", sum(p.numel() for p in trainable_params), sum(p.numel() for p in model.parameters()))

    model.train()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = (len(train_items) // (batch_size * grad_accum_steps)) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_time = time.time()
    max_seconds = max_duration_hours * 3600.0
    best_val_loss = float("inf")
    global_step = 0

    logger.info(
        "Starting training loop: %d epochs (batch_size=%d, grad_accum=%d, max %.1f hours)...",
        epochs,
        batch_size,
        grad_accum_steps,
        max_duration_hours,
    )

    for epoch in range(1, epochs + 1):
        if time.time() - start_time >= max_seconds:
            logger.info("Reached maximum duration of %.1f hours. Stopping gracefully.", max_duration_hours)
            break

        random.shuffle(train_items)
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for step, i in enumerate(range(0, len(train_items), batch_size)):
            if time.time() - start_time >= max_seconds:
                break

            batch = collate_fn(train_items[i : i + batch_size], pad_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=freeze_encoder)
                loss = criterion(logits, targets) / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps * len(targets)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_items) // batch_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                global_step += 1

                if global_step % 20 == 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    curr_acc = correct / max(1, total) * 100.0
                    curr_loss = epoch_loss / max(1, total)
                    vram_mb = torch.cuda.memory_allocated() / (1024**2) if device == "cuda" else 0
                    logger.info(
                        f"Epoch {epoch:02d}/{epochs} | Step {global_step:04d} | "
                        f"Loss: {curr_loss:.4f} | Train Acc: {curr_acc:.1f}% | "
                        f"Elapsed: {elapsed_min:.1f}m | VRAM: {vram_mb:.0f}MB"
                    )

        # Validation at end of epoch
        val_loss, val_acc = evaluate(agent, val_items, tok, device, batch_size=batch_size, freeze_encoder=freeze_encoder)
        logger.info(
            f"=== End of Epoch {epoch:02d} === Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100.0:.1f}% ==="
        )

        # Always save latest checkpoint
        latest_path = os.path.join(output_dir, "latest_checkpoint.pt")
        torch.save(model.state_dict(), latest_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved new best checkpoint to %s (Val Loss: %.4f)", ckpt_path, val_loss)

    total_time_h = (time.time() - start_time) / 3600.0
    logger.info("Training completed in %.2f hours. Best Val Loss: %.4f.", total_time_h, best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Laya ModernBERT on OS reflex tasks")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head")
    parser.add_argument("--full", action="store_true", help="Train full backbone (default: freeze encoder)")
    parser.add_argument("--max-hours", type=float, default=6.0, help="Max duration in hours (e.g. 5.5 for overnight)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        freeze_encoder=not args.full,
        max_duration_hours=args.max_hours,
    )
