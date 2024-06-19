# %%
import transformer_lens

# %%
# Import stuff
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

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
    n_layers=1,
    d_model=128,
    n_ctx=N_CTX,
    d_head=64,
    n_heads=2,
    d_mlp=128,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=42,
    device=DEVICE,
    attn_only=False,
)
model = transformer_lens.HookedTransformer(cfg, move_to_device=True)

# %%
# data generation

torch.manual_seed(42)


def tokenize(c: str):
    return ord(c) - ord("0") if c.isdigit() else 10  # 10 is comma


def untokenize(toks) -> str:
    return "".join([chr(tok + ord("0")) if tok < 10 else "," for tok in toks])


def str_to_tokens(seq_str):
    return torch.tensor([tokenize(c) for c in seq_str], device=DEVICE)


def seq_to_tokens(seq: np.ndarray) -> torch.Tensor:
    seq_str = ",".join(seq.astype(str))
    return str_to_tokens(seq_str)


def generate_addition_data():
    X, y = [], []

    for a in range(0, 150):
        for b in range(0, 150):
            x_str, y_str = f"{a},{b},", str(a + b)
            x_toks, y_toks = str_to_tokens(x_str), str_to_tokens(y_str)
            if len(x_toks) < cfg.n_ctx:
                x_toks = F.pad(x_toks, (cfg.n_ctx - len(x_toks), 0), value=10)

            for i in range(len(y_toks)):
                X.append(x_toks)
                y.append(y_toks[i])
                x_toks = torch.cat([x_toks[1:], y_toks[i].view(1)], dim=0)

    return torch.row_stack(X), torch.tensor(y, device=DEVICE)


X, y = generate_addition_data()

dataset = TensorDataset(X, y)
train_ds, val_ds = torch.utils.data.random_split(dataset, lengths=[0.9, 0.1])
print(len(train_ds), len(val_ds))

X_val, y_val = val_ds.dataset.tensors
X_val, y_val = X_val[val_ds.indices], y_val[val_ds.indices]
assert X_val.shape[0] == len(val_ds)

train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)

for X, y in train_dataloader:
    for b, seq in enumerate(X):
        print(f"For example, {untokenize(seq)} predicts {y[b]}")
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

n_epochs = 120

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
        config={"learning_rate": lr, "architecture": cfg.__dict__, "epochs": n_epochs},
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
                val_acc = accuracy(model(X_val), y_val)
                val_accuracies.append(val_acc.cpu())

                # early terminate, we don't really care about perfect accuracy
                if val_acc > 0.98:
                    break

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
