import os
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
import torch.amp
from tqdm import tqdm


def lod2dol(lod):
    """
    Convert list of dicts -> dict of lists.
    """
    from collections import defaultdict

    dol = defaultdict(list)
    for d in lod:
        for k, v in d.items():
            dol[k].append(v.item())
    return dict(dol)


def make_json_serializable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: make_json_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_serializable(v) for v in x]
    return x


def clone_run_to_new_experiment(old_run_id: str, new_experiment_name: str, new_run_name: str | None = None) -> str:
    """
    Clone an existing MLflow run into a NEW experiment (fresh run + copied artifacts).
    """
    old_run = mlflow.get_run(old_run_id)
    old_run_name = old_run.data.tags.get("mlflow.runName", "unnamed_run")

    exp = mlflow.get_experiment_by_name(new_experiment_name)
    exp_id = mlflow.create_experiment(new_experiment_name) if exp is None else exp.experiment_id

    new_run = mlflow.start_run(experiment_id=exp_id, run_name=new_run_name or f"{old_run_name}_resumed")
    new_run_id = new_run.info.run_id

    for k, v in old_run.data.tags.items():
        mlflow.set_tag(k, v)
    mlflow.set_tag("resumed_from", old_run_id)

    src_dir = mlflow.artifacts.download_artifacts(run_id=old_run_id)
    _dst_dir = Path(mlflow.get_artifact_uri()).as_posix().replace("file://", "")
    dst_dir = Path(_dst_dir)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(f"Cloned run {old_run_id} → {new_run_id} (experiment: {new_experiment_name})")
    mlflow.end_run()
    return new_run_id


# ————————————————————————————————————————————————————————————————————————————————


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.is_improvement = False

    def set_patience(self, patience):
        self.patience = patience

    def step(self, current_score):
        score = -current_score if self.mode == "min" else current_score

        if self.best_score is None:
            self.best_score = score
            self.is_improvement = True
            return False, True

        if score < self.best_score + self.min_delta:
            self.counter += 1
            self.is_improvement = False
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.is_improvement = True

        return self.should_stop, self.is_improvement


class NullLogger:
    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path, artifact_path=None):
        pass

    def log_model(self, model, artifact_path="model"):
        pass

    def log_run_metadata(self, cwd=None):
        pass


class MLFlowLogger:
    """
    Simple MLflow wrapper that manages an active run and
    provides convenient methods for logging parameters, metrics, and artifacts.
    """

    def __init__(self, experiment_name, run_name=None, autostart=True, nested=False):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.active_run = None
        self.tracking_base = None
        if autostart:
            self.start(nested=nested)

    def start(self, nested=False, resume_run_id=None):
        mlflow.set_experiment(self.experiment_name)
        if self.active_run is None:
            if resume_run_id:
                self.active_run = mlflow.start_run(run_id=resume_run_id)
            else:
                self.active_run = mlflow.start_run(run_name=self.run_name, nested=nested)
        self.tracking_base = self._strip_file_prefix(mlflow.get_tracking_uri())
        return self.active_run

    def end(self):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path, artifact_path=None, relative_uri=True):
        mlflow.log_artifact(path, artifact_path=artifact_path)
        uri = mlflow.get_artifact_uri(artifact_path)
        if relative_uri and self.tracking_base:
            uri = self._strip_file_prefix(uri)
            if uri.startswith(self.tracking_base):
                uri = os.path.relpath(uri, self.tracking_base)
        return uri

    def log_df_as_artifact(self, df, filename="data.csv", artifact_path=None, relative_uri=True):
        tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(tmp_dir) / filename
        df.to_csv(tmp_path, index=False)
        try:
            uri = self.log_artifact(tmp_path, artifact_path=artifact_path, relative_uri=relative_uri)
        finally:
            shutil.rmtree(tmp_dir)
        return uri

    @staticmethod
    def _strip_file_prefix(uri: str) -> str:
        return urlparse(uri).path if uri.startswith("file://") else uri

    def build_state_dict(self, model, optimizer, scheduler=None, metadata=None):
        return {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metadata": metadata,
        }

    def log_run_metadata(self, cwd=None):
        """Log git commit hash, dirty flag, hostname and SLURM job ID as MLflow tags."""
        import socket
        import subprocess

        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
            git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd).decode().strip())
        except Exception:
            git_commit, git_dirty = "unknown", False

        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "git_dirty": str(git_dirty),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
            }
        )

    def save_model(self, model, optimizer, scheduler=None, metadata=None, filename=None, symlink_as=None):
        import torch

        metadata = {} if metadata is None else metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"checkpoint_{timestamp}.pt"

        tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(tmp_dir) / filename
        torch.save(self.build_state_dict(model, optimizer, scheduler, metadata), tmp_path)

        # If symlink requested, create it pointing to the checkpoint file
        if symlink_as is not None:
            symlink_path = Path(tmp_dir) / symlink_as
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(filename)  # relative symlink
            # Log both the checkpoint and the symlink
            mlflow.log_artifacts(str(tmp_dir), artifact_path="checkpoints")
        else:
            artifact_path = "checkpoints"
            mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)

        shutil.rmtree(tmp_dir)

        uri = mlflow.get_artifact_uri("checkpoints")
        return Path(uri) / filename


# ———————————————————————————————————————————————————————————————————————————————


class BaseTrainer:
    def shared_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def train_epoch(self) -> Any:
        pass

    def train(self):
        pass

    def val_epoch(self):
        pass

    def valid_epoch_end(self, *args: Any, **kwargs: Any) -> Any:
        pass


class Trainer(BaseTrainer):
    LOSSES_PER_EPOCH_FILEPATTERN = "losses_epoch{current_epoch}_{val_step}.csv"

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        scheduler,
        patience=3,
        n_train_batches: int | None = None,
        n_val_batches: int | None = None,
        n_validations_per_epoch=1,
        logger: NullLogger | MLFlowLogger | None = None,
        mlflow_params: dict | None = None,
        start_epoch=0,
        log_loss_per_disease=False,
        use_tqdm=True,
        use_amp=True,
        use_rich=True,
        checkpoint_every=None,
        optim_config=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper = EarlyStopping(patience=patience, min_delta=0.001, mode="min")

        assert isinstance(dataloaders, list), (
            "Argument 'dataloaders' should be a list of either 2 or 3 dataloaders (train/val[/test])"
        )

        if len(dataloaders) == 2:
            self.train_loader, self.valid_loader = dataloaders
        elif len(dataloaders) == 3:
            self.train_loader, self.valid_loader, self.test_loader = dataloaders
        else:
            raise ValueError(f"{len(dataloaders)=} ")

        self.n_train_batches: Literal["all"] | int = n_train_batches if n_train_batches is not None else "all"
        self.n_val_batches: Literal["all"] | int = n_val_batches if n_val_batches is not None else "all"

        self.train_outputs: list[dict[str, torch.Tensor]] = []
        self.valid_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []

        self.current_epoch = start_epoch
        self._validation_counter = 0
        self.n_validations_per_epoch = n_validations_per_epoch

        self.logger = logger if logger is not None else NullLogger()
        self.val_loss: dict[str, torch.Tensor] | None = None
        self.ema_alpha = 0.02

        self.additional_mlflow_params = (mlflow_params or {}) | {"ema_alpha": self.ema_alpha}

        self.use_tqdm = use_tqdm
        self.use_rich = use_rich
        self.checkpoint_every = checkpoint_every  # None = only save on improvement
        self.optim_config = optim_config

        self.ce_ema: torch.Tensor | None = None
        self.time_ema: torch.Tensor | None = None
        self.log_loss_per_disease = log_loss_per_disease

        # Precompute predicted domain IDs as a tensor for masking
        self._predicted_domain_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])

        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        # bfloat16 has the same exponent range as float32, so no gradient scaling needed
        self.scaler = torch.amp.grad_scaler.GradScaler("cuda", enabled=False)

        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True

    # ——————————————————————————————————————————————————————————————————————

    @property
    def device(self):
        return self.model.device

    def get_subject_ids_per_partition(self):
        return {
            "train_ids": sorted(self.train_loader.dataset.subject_list),
            "valid_ids": sorted(self.valid_loader.dataset.subject_list),
            "test_ids": sorted(self.test_loader.dataset.subject_list),
        }

    def _setup_display(self, epoch_rows: list[tuple[str, ...]]):
        """
        Returns (progress, live_ctx, refresh_fn).
        When use_rich=False, everything is a no-op.
        """
        if not self.use_rich:
            return None, nullcontext(), lambda: None

        try:
            from rich import box
            from rich.console import Group
            from rich.live import Live
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
            from rich.table import Table
        except ImportError:
            return None, nullcontext(), lambda: None

        def build_table():
            t = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style="bold")
            t.add_column("Epoch", justify="right", style="cyan", width=6)
            t.add_column("Train CE", justify="right", width=9)
            t.add_column("Train dt", justify="right", width=10)
            t.add_column("Train tot", justify="right", width=10)
            t.add_column("Val CE", justify="right", width=9)
            t.add_column("Val dt", justify="right", width=10)
            t.add_column("Val tot", justify="right", width=10)
            t.add_column("LR", justify="right", width=9)
            t.add_column("Improved?", justify="center", width=10)
            for row in epoch_rows:
                t.add_row(*row)
            return t

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        live = Live(Group(build_table(), progress), refresh_per_second=4)
        return progress, live, lambda: live.update(Group(build_table(), progress))

    def train(self, max_epochs=1000, patience=None):

        if patience is not None:
            self.early_stopper.set_patience(patience)

        n_batches_epoch = len(self.train_loader)
        # n_validations_per_epoch <= 1: end-of-epoch validation is enough, no mid-epoch eval
        eval_every = (
            None if self.n_validations_per_epoch <= 1 else max(1, n_batches_epoch // self.n_validations_per_epoch)
        )

        self.logger.log_params(self.model.config)
        self.logger.log_params(self.additional_mlflow_params)

        epoch_rows: list[tuple[str, ...]] = []
        progress, live_ctx, refresh_display = self._setup_display(epoch_rows)
        if self.use_rich and progress and not isinstance(live_ctx, nullcontext):
            _print = live_ctx.console.print
        else:
            _print = print

        with live_ctx:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch
                self.model.train()

                train_task = (
                    progress.add_task(f"[green]Ep {epoch:>4d}  train", total=len(self.train_loader))
                    if progress
                    else None
                )
                train_loss = self.train_epoch(eval_every=eval_every, _progress=progress, _task_id=train_task)
                self.on_train_epoch_end(epoch)
                if progress and train_task:
                    progress.remove_task(train_task)

                n_val = len(self.valid_loader) if self.n_val_batches == "all" else self.n_val_batches
                val_task = progress.add_task(f"[blue]Ep {epoch:>4d}  val  ", total=n_val) if progress else None

                self.val_loss, self.mean_val_loss = self.valid_epoch(
                    n_batches=self.n_val_batches, _progress=progress, _task_id=val_task
                )
                if progress and val_task:
                    progress.remove_task(val_task)

                metrics = {"train_loss": train_loss}
                if self.val_loss is not None:
                    metrics["val_loss"] = self.val_loss
                    if "val_ce_loss_per_disease" in metrics["val_loss"]:
                        metrics["val_loss"].pop("val_ce_loss_per_disease")

                self.logger.log_metrics(metrics["val_loss"], step=epoch)
                self.logger.log_metrics(metrics["train_loss"], step=epoch)

                should_stop, improved = self.early_stopper.step(self.mean_val_loss)

                val_loss_val = float(self.mean_val_loss.cpu())
                ckpt_metadata = {
                    "epoch": epoch,
                    "val_loss": val_loss_val,
                    "train_loss": float(train_loss["train_total"].cpu()),
                    "n_params": sum(p.numel() for p in self.model.parameters()),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "attention_scheme": getattr(self.model.config, "attention_scheme", None),
                    "date_cutoff": None,
                } | self.get_subject_ids_per_partition()

                if improved and not isinstance(self.logger, NullLogger):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ckpt_filepath = f"epoch{epoch}__valloss_{val_loss_val:.4f}__{timestamp}.pt"
                    ckpt_uri = self.logger.save_model(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        metadata=ckpt_metadata,
                        filename=ckpt_filepath,
                        symlink_as="best_model.pt",
                    )
                    _print(f"New best model → {ckpt_uri}")

                if (
                    self.checkpoint_every
                    and (epoch % self.checkpoint_every == 0)
                    and not isinstance(self.logger, NullLogger)
                ):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ckpt_filepath = f"epoch{epoch}__periodic__{timestamp}.pt"
                    ckpt_uri = self.logger.save_model(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        metadata=ckpt_metadata | {"checkpoint_type": "periodic"},
                        filename=ckpt_filepath,
                    )
                    _print(f"Periodic checkpoint → {ckpt_uri}")

                # Add row to epoch table
                lr = self.optimizer.param_groups[0]["lr"]
                epoch_rows.append(
                    (
                        str(epoch),
                        f"{train_loss['train_ce_loss'].item():.4f}",
                        f"{train_loss['train_time_loss'].item():.4f}",
                        f"{train_loss['train_total'].item():.4f}",
                        f"{self.val_loss['val_ce_loss'].item():.4f}",
                        f"{self.val_loss['val_time_loss'].item():.4f}",
                        f"{self.val_loss['val_total'].item():.4f}",
                        f"{lr:.2e}",
                        "[yellow]★[/]" if improved else "",
                    )
                )
                refresh_display()

                if should_stop:
                    _print(f"Early stopping at epoch {epoch}")
                    break

                self.epoch_end()

    def shared_step(
        self, batch, batch_idx, epoch, return_logits=False, return_att=False, stage="training", add_prefix=None
    ):
        """
        Unified train/val step.

        batch : DelphiBatch (already on device)
        """
        model = self.model

        # Move batch to device
        batch = batch.to(self.device)

        with torch.amp.autocast_mode.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # Forward pass
            logits_dict, att = model(batch, return_attention=return_att)

            # ── Build targets from the batch ──────────────────────────────
            target_global_ids = batch.global_token_ids[:, 1:]  # [B, T-1]
            target_domain_ids = batch.domain_ids[:, 1:]  # [B, T-1]
            input_ages = batch.ages[:, :-1]  # [B, T-1]
            target_ages = batch.ages[:, 1:]  # [B, T-1]

            # Concatenate logits from all predicted domains → [B, T, sum(vocab_sizes)]
            logits_cat = torch.cat(
                [logits_dict[dname] for dname in model.predicted_domains],
                dim=-1,
            )
            logits_cat = logits_cat[:, :-1, :]  # [B, T-1, V_total]

            # Mask: only compute loss on positions belonging to predicted domains
            predicted_ints = self._predicted_domain_ints.to(self.device)
            predict_mask = torch.isin(target_domain_ids, predicted_ints)  # [B, T-1]

            f_logits = logits_cat[predict_mask]  # [N_pred, V_total]
            f_global_ids = target_global_ids[predict_mask]  # [N_pred]

            # ── Cross-entropy loss ────────────────────────────────────────
            loss_ce = model.cross_entropy_loss(f_logits, f_global_ids)

            # ── Time-to-event loss ────────────────────────────────────────
            age_diff = (target_ages - input_ages)[predict_mask]
            time_loss = model.time_to_event_loss(f_logits, age_diff, t_min=1e-1, agg="mean")

        # ── Package losses (outside autocast, already float32 scalars) ──
        prefix = "" if add_prefix is None else add_prefix + "_"

        loss: dict[str, Any] = {
            f"{prefix}ce_loss": loss_ce,
            f"{prefix}time_loss": time_loss,
            f"{prefix}total": loss_ce + time_loss,
        }

        if self.log_loss_per_disease and stage == "validation":
            loss[f"{prefix}ce_loss_per_disease"] = model.cross_entropy_loss(
                f_logits.float(), f_global_ids, agg="per_disease"
            )  # pd.Series indexed by token_id

        if return_att and return_logits:
            return loss, logits_cat, att
        elif return_logits:
            return loss, logits_cat
        elif return_att:
            return loss, att
        else:
            return loss

    def valid_epoch_end(self, loss_outputs):
        self._validation_counter += 1

        if len(loss_outputs) and "val_ce_loss_per_disease" in loss_outputs[0]:
            # Each item is a pd.Series(mean_log_p, index=token_id).
            # Concat produces a Series with repeated token_id index entries
            # (one entry per batch where the token appeared).
            # groupby then computes, per token:
            #   log_p_total: sum of per-batch means  (contribution = frequency × difficulty)
            #   log_p_mean:  mean of per-batch means  (difficulty, independent of frequency)
            combined = pd.concat([x["val_ce_loss_per_disease"] for x in loss_outputs])
            self._val_ce_loss_per_disease_df = (
                combined.groupby(level=0)
                .agg(log_p_total="sum", log_p_mean="mean")
                .sort_values("log_p_total")
                .reset_index()
            )

        return 1

    def compute_mean(self, outputs: Sequence[dict[str, Any]]):
        out: dict[str, torch.Tensor] = {}
        if not outputs:
            return out
        for k in outputs[0]:
            try:
                vals = [(v[k].detach() if torch.is_tensor(v[k]) else torch.as_tensor(v[k])) for v in outputs if k in v]
                out[k] = torch.stack(vals).mean()
            except Exception as e:
                if not k.endswith("_per_disease"):  # DataFrames are handled separately
                    print(f"[compute_mean] Skipping key '{k}': {e}")
                out[k] = torch.tensor(float("nan"))
        return out

    def train_epoch(self, n_batches: Literal["all"] | int = "all", eval_every=None, _progress=None, _task_id=None):

        self.val_step = 0

        use_tqdm = (_progress is None) and self.use_tqdm
        pbar = (
            tqdm(
                total=len(self.train_loader.dataset) if n_batches == "all" else n_batches,
                disable=not use_tqdm,
            )
            if use_tqdm
            else None
        )

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self.shared_step(batch, batch_idx=i, epoch=self.current_epoch, stage="training", add_prefix="train")
            assert isinstance(loss, dict)

            self.scaler.scale(loss["train_total"]).backward()
            if self.optim_config is not None and self.optim_config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            with torch.no_grad():
                ce_val = loss["train_ce_loss"].detach()
                time_val = loss["train_time_loss"].detach()

                self.ce_ema = (
                    ce_val if self.ce_ema is None else (1 - self.ema_alpha) * self.ce_ema + self.ema_alpha * ce_val
                )

                self.time_ema = (
                    time_val
                    if self.time_ema is None
                    else (1 - self.ema_alpha) * self.time_ema + self.ema_alpha * time_val
                )

            assert isinstance(self.ce_ema, torch.Tensor)
            assert isinstance(self.time_ema, torch.Tensor)
            self.train_outputs.append(
                {
                    "train_ce_loss": ce_val,
                    "train_time_loss": time_val,
                    "train_ce_ema_loss": self.ce_ema,
                    "train_time_ema_loss": self.time_ema,
                    "train_total": ce_val + time_val,
                }
            )

            if eval_every is not None and (i % eval_every) == 0:
                self.val_loss, self.mean_val_loss = self.valid_epoch(n_batches=self.n_val_batches)
                self.val_step += 1

            current_loss = self.train_outputs[-1]
            if pbar:
                pbar.set_postfix(
                    {
                        "ce": f"{current_loss['train_ce_loss'].item():.3f} ({current_loss['train_ce_ema_loss'].item():.3f})",
                        "dt": f"{current_loss['train_time_loss'].item():.1f} ({current_loss['train_time_ema_loss'].item():.1f})",
                    }
                )
                pbar.update(self.train_loader.batch_size)

            if _progress is not None and _task_id is not None:
                _progress.advance(_task_id)
                _progress.update(
                    _task_id,
                    description=(
                        f"[green]Ep {self.current_epoch:>4d}  train  ce={current_loss['train_ce_ema_loss'].item():.3f}"
                    ),
                )

            if (n_batches is not None) and (i == n_batches):
                break

        return self.compute_mean(self.train_outputs)

    def valid_epoch(self, n_batches: Literal["all"] | int = "all", _progress=None, _task_id=None):

        self.model.eval()

        use_tqdm = (_progress is None) and self.use_tqdm
        pbar = (
            tqdm(
                total=len(self.valid_loader.dataset) if n_batches == "all" else n_batches,
                disable=not use_tqdm,
            )
            if use_tqdm
            else None
        )

        with torch.no_grad():
            loss_outputs: list[dict[str, Any]] = []
            ce_sum, time_sum = 0.0, 0.0

            for i, batch in enumerate(self.valid_loader):
                loss = self.shared_step(
                    batch, batch_idx=i, epoch=self.current_epoch, stage="validation", add_prefix="val"
                )
                assert isinstance(loss, dict)
                loss_outputs.append(loss)

                ce_sum += loss["val_ce_loss"].item()
                time_sum += loss["val_time_loss"].item()
                n = i + 1

                if pbar:
                    pbar.set_postfix(
                        {
                            "ce": f"{loss['val_ce_loss'].item():.3f} ({ce_sum / n:.3f})",
                            "dt": f"{loss['val_time_loss'].item():.1f} ({time_sum / n:.1f})",
                        }
                    )
                    pbar.update(self.valid_loader.batch_size)

                if _progress is not None and _task_id is not None:
                    _progress.advance(_task_id)
                    _progress.update(
                        _task_id, description=(f"[blue]Ep {self.current_epoch:>4d}  val    ce={ce_sum / n:.3f}")
                    )

                if n_batches != "all" and i == n_batches:
                    break

        if pbar:
            pbar.close()

        self.valid_epoch_end(loss_outputs)

        if hasattr(self, "_val_ce_loss_per_disease_df"):
            losses_per_epoch_file = self.LOSSES_PER_EPOCH_FILEPATTERN.format(
                current_epoch=self.current_epoch, val_step=self.val_step
            )
            if isinstance(self.logger, MLFlowLogger):
                self.logger.log_df_as_artifact(
                    df=self._val_ce_loss_per_disease_df,
                    filename=losses_per_epoch_file,
                    artifact_path="val_loss_per_disease",
                )

        mean_val = self.compute_mean(loss_outputs)

        self.model.train()

        return mean_val, mean_val["val_total"]

    def on_train_epoch_end(self, epoch):
        """Hook called after each train_epoch(). Override in subclasses."""
        pass

    def epoch_end(self):
        self.train_outputs = []
        self.valid_outputs = []

    def mlflow_logging(self):
        pass
