import datetime
import logging

import wandb
from omegaconf import OmegaConf


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None and wandb_cfg.run_path is None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    if (
        wandb_cfg.run_id is not None and run.resumed
    ):  # because wandb sweep forces resumed=True
        logging.info(f"Trying to resume run {wandb_cfg.run_id}")
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)
        checkpoint_name = run.summary["checkpoint"]
        if checkpoint_name is not None:
            logging.info(f"Restore checkpoint {checkpoint_name}")
            wandb.restore(checkpoint_name)
    elif wandb_cfg.run_path is not None:
        logging.info(f"Trying to start new run from {wandb_cfg.run_path}")
        api = wandb.Api()
        run.config = api.run(wandb_cfg.run_path).config
        run.config["old_config"] = run.config.copy()
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)
        checkpoint_name = run.summary.get("checkpoint")
        if checkpoint_name is not None:
            logging.info(f"Restore checkpoint {checkpoint_name}")
            wandb.restore(checkpoint_name, run_path=wandb_cfg.run_path)
    else:
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)
    if wandb_cfg.log_code is not None:
        run.log_code()
    return run
