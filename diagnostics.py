"""Shared diagnostics: TensorBoard + (opt-in) Weights & Biases logging, norm
helpers, and a periodic checkpoint manager. Imported by the train_gpt_* scripts.

Design notes
------------
* TensorBoard is always on when the package is importable (local, no creds).
* wandb is opt-in and offline-safe: it activates only when requested via env and
  no-ops gracefully if the package is missing or unconfigured, so a training run
  never breaks because of logging. Control via env vars:
      WANDB=1|0            force on/off (default: on iff some WANDB_* config exists)
      WANDB_API_KEY=...    online logging
      WANDB_MODE=offline   local-only wandb run (default when no API key)
      WANDB_PROJECT=name   project name (default "rpb-optimizer")
* All logging is expected to be called on the master/rank-0 process only. The
  norm helpers, however, are collective when reduce=True and must then be called
  on *every* rank (they perform a dist all_reduce); log the returned scalar on
  the master only.
"""

import os
import math

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard optional
    SummaryWriter = None

try:
    import wandb
except Exception:  # pragma: no cover - wandb optional
    wandb = None


# -----------------------------------------------------------------------------
# distributed helpers

def dist_is_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


# -----------------------------------------------------------------------------
# norm helpers
#
# Each returns a Python float. With reduce=True the sum-of-squares is all_reduced
# (SUM) across ranks before the sqrt; this must be called on every rank. For
# weights/updates (identical across ranks) leave reduce=False. For grads under
# data-parallel (rank-local, different data) reduce=True reports the combined
# sqrt(sum_r ||g_r||^2) aggregate.

@torch.no_grad()
def _sumsq(tensors):
    acc = None
    for t in tensors:
        if t is None:
            continue
        s = t.detach().float().pow(2).sum()
        acc = s if acc is None else acc + s
    return acc


@torch.no_grad()
def _finish(acc, reduce):
    if acc is None:
        return 0.0
    if reduce and dist_is_initialized():
        import torch.distributed as dist
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    return float(acc.sqrt().item())


@torch.no_grad()
def global_grad_norm(params, reduce=False):
    return _finish(_sumsq([getattr(p, "grad", None) for p in params]), reduce)


@torch.no_grad()
def global_weight_norm(params, reduce=False):
    return _finish(_sumsq(list(params)), reduce)


@torch.no_grad()
def snapshot_params(params):
    """Clone params (for a later update-norm computation around opt.step())."""
    return [p.detach().clone() for p in params]


@torch.no_grad()
def global_update_norm(params, snaps, reduce=False):
    diffs = [p.detach() - s for p, s in zip(params, snaps)]
    return _finish(_sumsq(diffs), reduce)


# Convenience wrappers over a dict {name: [params...]}; each returns a flat
# metrics dict ready to hand to Diagnostics.log, including a "/global" entry.

@torch.no_grad()
def grad_norms(named_params, reduce=False):
    out = {f"grad_norm/{n}": global_grad_norm(ps, reduce=reduce) for n, ps in named_params.items()}
    allp = [p for ps in named_params.values() for p in ps]
    out["grad_norm/global"] = global_grad_norm(allp, reduce=reduce)
    return out


@torch.no_grad()
def weight_norms(named_params, reduce=False):
    out = {f"weight_norm/{n}": global_weight_norm(ps, reduce=reduce) for n, ps in named_params.items()}
    allp = [p for ps in named_params.values() for p in ps]
    out["weight_norm/global"] = global_weight_norm(allp, reduce=reduce)
    return out


@torch.no_grad()
def snapshot_named(named_params):
    return {n: snapshot_params(ps) for n, ps in named_params.items()}


@torch.no_grad()
def update_norms(named_params, snaps, reduce=False):
    out = {f"update_norm/{n}": global_update_norm(ps, snaps[n], reduce=reduce)
           for n, ps in named_params.items()}
    allp = [p for ps in named_params.values() for p in ps]
    alls = [s for n in named_params for s in snaps[n]]
    out["update_norm/global"] = global_update_norm(allp, alls, reduce=reduce)
    return out


# -----------------------------------------------------------------------------
# logging sink

def _wandb_requested():
    val = os.environ.get("WANDB", "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    # default: enable only if some wandb configuration is present in the env
    return bool(os.environ.get("WANDB_API_KEY")
                or os.environ.get("WANDB_MODE")
                or os.environ.get("WANDB_PROJECT"))


class Diagnostics:
    """TensorBoard + opt-in wandb scalar/histogram sink. Construct on master only."""

    def __init__(self, run_id, log_dir="logs", project="rpb-optimizer",
                 config=None, enabled=True, use_wandb=None):
        self.enabled = bool(enabled)
        self.run_id = str(run_id)
        self.tb = None
        self.wandb_run = None
        if not self.enabled:
            return

        if SummaryWriter is not None:
            tb_dir = os.path.join(log_dir, self.run_id, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb = SummaryWriter(tb_dir)
        else:
            print("[diagnostics] tensorboard not available; skipping TB logging")

        if use_wandb is None:
            use_wandb = _wandb_requested()
        if use_wandb and wandb is not None:
            mode = os.environ.get("WANDB_MODE")
            if mode is None:
                mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            try:
                self.wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", project),
                    name=self.run_id, id=self.run_id, dir=log_dir,
                    mode=mode, config=(config or {}), resume="allow",
                )
            except Exception as e:  # pragma: no cover - never break training
                print(f"[diagnostics] wandb init failed ({e}); continuing without wandb")
                self.wandb_run = None
        elif use_wandb and wandb is None:
            print("[diagnostics] wandb requested but not installed; continuing without it")

    def log(self, metrics: dict, step: int):
        if not self.enabled or not metrics:
            return
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(k, float(v), step)
        if self.wandb_run is not None:
            self.wandb_run.log({k: float(v) for k, v in metrics.items()}, step=int(step))

    def log_histogram(self, name: str, values, step: int):
        if not self.enabled:
            return
        v = values.detach().float().cpu()
        if self.tb is not None:
            self.tb.add_histogram(name, v, step)
        if self.wandb_run is not None:
            try:
                self.wandb_run.log({name: wandb.Histogram(v.numpy())}, step=int(step))
            except Exception:
                pass

    def close(self):
        if not self.enabled:
            return
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


# -----------------------------------------------------------------------------
# checkpointing (periodic, keep-all)

class CheckpointManager:
    """Saves model + optimizer states under logs/<run_id>/. Keeps every checkpoint."""

    def __init__(self, run_id, log_dir="logs", enabled=True):
        self.enabled = bool(enabled)
        self.dir = os.path.join(log_dir, str(run_id))
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def save(self, step, model, optimizers, extra=None):
        if not self.enabled:
            return None
        payload = dict(
            step=int(step),
            model=model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
        )
        if extra:
            payload.update(extra)
        path = os.path.join(self.dir, f"state_step{int(step):06d}.pt")
        torch.save(payload, path)
        return path
