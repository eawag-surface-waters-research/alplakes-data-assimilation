"""Run logging shared by the engines: a progress bar and the run header and footer, in one
layout so the logs of a native and an OpenDA run line up."""
import os
import logging

logger = logging.getLogger(__name__)


def display_path(path):
    """The absolute path with $HOME shown as '~'."""
    path = os.path.abspath(path)
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if home != "/" and path.startswith(home + os.sep) else path


class _NullBar:
    def update(self, *a, **k): pass
    def set_postfix_str(self, *a, **k): pass
    def close(self, *a, **k): pass


def make_progress(total, enabled, desc=None):
    """A tqdm bar, or a bar that does nothing when progress is off (server runs)."""
    if not enabled:
        return _NullBar()
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit="day", dynamic_ncols=True, leave=True)


def log_run_header(cfg):
    engine = cfg.get("engine", "python")
    logger.info(f"=== RUN | engine={engine} algo={cfg.get('algorithm')} "
                f"model={cfg.get('model')} lake={cfg.get('lake')} ===")
    logger.info(f"      window={cfg.get('start_date')}..{cfg.get('end_date')}  "
                f"n_members={cfg.get('n_members')}")
    knobs = [f"sigma_obs={cfg.get('sigma_obs')}"]
    if engine == "python" and cfg.get("inflation") is not None:
        knobs.append(f"inflation={cfg.get('inflation')}")
    knobs += [f"sigma_scale={cfg.get('sigma_scale', 1.0)}", f"rng_seed={cfg.get('rng_seed', 42)}"]
    logger.info("      " + "  ".join(knobs))
    logger.info(f"      obs={display_path(cfg['obs_file'])}")


def log_run_footer(cfg, skill, steps, updates, elapsed_s, out_path=None):
    """`skill` is report_summary's overall dict or None; `updates` None where the engine has no
    count separate from its steps."""
    logger.info(f"=== DONE | engine={cfg.get('engine', 'python')} algo={cfg.get('algorithm')} "
                f"lake={cfg.get('lake')} ===")
    line = f"      steps={steps}"
    if updates is not None:
        line += f"  updates={updates}"
    logger.info(f"{line}  elapsed={elapsed_s:.1f}s")
    if skill:
        logger.info(f"      rmse={skill['rmse']}  bias={skill['bias']}  n_obs={skill['n']}  (analysis fit)")
    if out_path:
        logger.info(f"      output={display_path(out_path)}")
