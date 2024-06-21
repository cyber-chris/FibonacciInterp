# %%
import transformer_lens

# %%
# Import stuff
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

import os
import datetime
import pickle

import wandb

# %%
DEVICE = "cuda"

# %%
N_CTX = 64
D_VOCAB = 11  # 10 digits and a comma
cfg = transformer_lens.HookedTransformerConfig(
    n_layers=2,
    d_model=128,
    n_ctx=N_CTX,
    d_head=32,
    n_heads=4,
    d_mlp=None,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=42,
    device=DEVICE,
    attn_only=True,
)
model = transformer_lens.HookedTransformer(cfg, move_to_device=True)

# %%
# data generation

torch.manual_seed(42)

MAX_DIGITS = 4


def tokenize(c: str):
    return ord(c) - ord("0") if c.isdigit() else 10  # 10 is comma


def untokenize(toks) -> str:
    return "".join([chr(tok + ord("0")) if tok < 10 else "," for tok in toks])


def str_to_tokens(seq_str):
    formatted_str = ",".join(
        [
            item.zfill(MAX_DIGITS) if item.isdigit() else item
            for item in seq_str.split(",")
        ]
    )
    return torch.tensor([tokenize(c) for c in formatted_str], device=DEVICE)


OPERAND_LIMIT = 600


def generate_addition_data():
    X, y = [], []

    for a in range(0, OPERAND_LIMIT):
        for b in range(0, OPERAND_LIMIT):
            x_str, y_str = f"{a},{b},", str(a + b)
            x_toks, y_toks = str_to_tokens(x_str), str_to_tokens(y_str)
            if len(x_toks) < cfg.n_ctx:
                x_toks = F.pad(x_toks, (cfg.n_ctx - len(x_toks), 0), value=10)

            for i in range(len(y_toks)):
                X.append(x_toks)
                y.append(y_toks[i])
                x_toks = torch.cat([x_toks[1:], y_toks[i].view(1)], dim=0)
            # also learn the end of a number by showing it the comma
            X.append(x_toks)
            y.append(torch.tensor(10))

    return torch.row_stack(X), torch.tensor(y, device=DEVICE)


if os.path.exists(f"data-cache/add-{OPERAND_LIMIT}-X.pt"):
    X = torch.load(f"data-cache/add-{OPERAND_LIMIT}-X.pt")
    y = torch.load(f"data-cache/add-{OPERAND_LIMIT}-y.pt")
else:
    X, y = generate_addition_data()
    print("Saving generated data to data-cache/add-*.pt")
    torch.save(X, f"data-cache/add-{OPERAND_LIMIT}-X.pt")
    torch.save(y, f"data-cache/add-{OPERAND_LIMIT}-y.pt")

random_indices = torch.randperm(len(X))[: int(0.75 * len(X))]
X, y = X[random_indices], y[random_indices]

dataset = TensorDataset(X, y)
train_ds, val_ds = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2])
print(len(train_ds), len(val_ds))

X_val, y_val = val_ds.dataset.tensors
X_val, y_val = X_val[val_ds.indices], y_val[val_ds.indices]
assert X_val.shape[0] == len(val_ds)

train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=128, shuffle=True)

for X, y in train_dataloader:
    for b, seq in enumerate(X):
        print(f"For example, {untokenize(seq)} predicts {y[b]}")
        if b > 10:
            break
    break

# %%
# Loss function (cross entropy loss)


def loss_fn(
    logits: torch.Tensor,  # [batch, pos, d_vocab]
    targets: torch.Tensor,
):
    # This not the optimal way to compute a loss function for an LLM, but for this task
    # I've crafted the x,y pairs in a certain way to hopefully better learn addition.
    # For the full fibonacci I'll probably do the usual c_e(logits, targets) method.
    return torch.nn.functional.cross_entropy(logits[:, -1, :], targets)


def correct_preds_count(
    logits: torch.Tensor,  # [batch, pos, d_vocab]
    targets: torch.Tensor,
):
    probs = logits[:, -1, :].softmax(dim=-1)
    predictions = probs.argmax(dim=-1)
    assert predictions.shape == targets.shape
    return (predictions == targets).float().sum()


def accuracy(
    logits: torch.Tensor,  # [batch, pos, d_vocab]
    targets: torch.Tensor,
):
    # using this as accuracy of the batch
    probs = logits[:, -1, :].softmax(dim=-1)
    predictions = probs.argmax(dim=-1)
    assert predictions.shape == targets.shape
    acc = (predictions == targets).float().mean()
    assert 0 <= acc <= 1
    return acc


# %%
# training

n_epochs = 5

# Optimization
lr = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=lr)


@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]


def train_model(model: transformer_lens.HookedTransformer) -> TrainingHistory:
    wandb.init(
        project="fibonacci-interp",
        config={
            "learning_rate": lr,
            "architecture": cfg.__dict__,
            "dataset": f"fib-padding-{len(train_ds)}",
            "epochs": n_epochs,
        },
    )

    losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        for i, (X, y) in enumerate(train_dataloader):
            optim.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)

            loss.backward()
            optim.step()

            if i == 0:
                losses.append(loss.item())
                train_batch_acc = accuracy(logits, y)
                train_accuracies.append(train_batch_acc.cpu())

                val_correct_counts = 0
                for X_val, y_val in val_dataloader:
                    val_correct_counts += correct_preds_count(model(X_val), y_val)
                val_acc = val_correct_counts / len(val_ds)
                assert 0 <= val_acc <= 1
                val_accuracies.append(val_acc)

                print(
                    f"({epoch}) loss = {loss.item():.4f}, batch accuracy = {train_batch_acc}, val accuracy = {val_acc}"
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch_loss": loss.item(),
                        "batch_acc": train_batch_acc,
                        "val_acc": val_acc,
                    }
                )

                # early terminate, we don't really care about perfect accuracy
                if val_acc > 0.999:
                    return TrainingHistory(losses, train_accuracies, val_accuracies)

    return TrainingHistory(losses, train_accuracies, val_accuracies)


history = train_model(model)


# %%
def save_model_state_dict(model: transformer_lens.HookedTransformer, filename=None):
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not filename:
        timestamp = datetime.datetime.now().isoformat("T", "minutes").replace(":", "-")
        filename = f"addition_model_state_dict_{timestamp}.pkl"
    with open(f"models/{filename}", "wb") as f:
        pickle.dump(model.state_dict(), f)


save_model_state_dict(model)
