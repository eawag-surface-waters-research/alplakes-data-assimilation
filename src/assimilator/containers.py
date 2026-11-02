"""A long-lived container the members' model runs are exec'd into: creating a container per
member per window dominated the run time."""
import os
import subprocess


def start_persistent_container(cfg, name, mount, target, image):
    """Start container `name` from `image`, idle, with `mount` at `target`. A leftover container of
    the same name is replaced. With docker_dir and repo_dir in cfg, `mount` is translated to the
    path the Docker daemon sees."""
    if cfg.get("docker_dir") and cfg.get("repo_dir"):
        mount = os.path.join(cfg["docker_dir"], os.path.relpath(mount, cfg["repo_dir"]))
    mount = mount.replace("\\", "/")
    subprocess.run(f"docker rm -f {name}", shell=True, capture_output=True)
    result = subprocess.run(f"docker run -d --name {name} -v {mount}:{target} --entrypoint sleep {image} infinity",
                            shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"container start failed ({name}): {result.stderr.strip()}")


def stop_persistent_container(name):
    subprocess.run(f"docker stop {name}", shell=True, capture_output=True)
    subprocess.run(f"docker rm {name}", shell=True, capture_output=True)
