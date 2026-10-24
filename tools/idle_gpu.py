"""Pod-side idle shutdown. Standard library only.

Stops the pod after a stretch with no GPU work and no SSH session. It uses the pod-scoped RunPod key
that RunPod gives every pod (read from PID 1's environment), so no account key lives on the pod.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def pod_environment():
    raw = Path("/proc/1/environ").read_bytes().split(b"\0")
    return dict(item.decode(errors="replace").split("=", 1) for item in raw if b"=" in item)


def busy():
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15)
    if gpu.returncode or gpu.stdout.strip():
        return True
    utilization = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15)
    if utilization.returncode or any(float(v.strip()) > 0 for v in utilization.stdout.splitlines()):
        return True
    processes = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, timeout=15)
    if processes.returncode:
        return True
    # Interactive SSH sessions and jobs started with `gpu.py run --detach` (CPU-bound phases like downloads).
    return any(line.startswith("sshd: ") and ("@pts/" in line or "@notty" in line)
               or "uv run" in line for line in processes.stdout.splitlines())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("/workspace/.dolos-idle"))
    parser.add_argument("--idle-seconds", type=int, default=7200)
    parser.add_argument("--check", action="store_true", help="Report activity without stopping anything.")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.check:
        print(json.dumps({"busy": busy()}))
        return
    lock = (args.directory / "monitor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # already running
    env = pod_environment()
    if not env.get("RUNPOD_POD_ID") or not env.get("RUNPOD_API_KEY"):
        print("No pod-scoped RunPod key in the environment; idle stop is disabled.", flush=True)
        return
    last_activity = time.monotonic()
    print(json.dumps({"monitor_started": True, "idle_seconds": args.idle_seconds}), flush=True)
    while True:
        try:
            now = time.monotonic()
            if busy():
                last_activity = now
            (args.directory / "status.json").write_text(json.dumps(
                {"idle_seconds": round(now - last_activity), "timeout_seconds": args.idle_seconds}))
            if now - last_activity >= args.idle_seconds:
                print("Idle timeout: stopping the pod.", flush=True)
                subprocess.run(["runpodctl", "stop", "pod", env["RUNPOD_POD_ID"]],
                               env={**os.environ, **env}, check=True, timeout=60)
                return
        except Exception as exc:
            # Never stop because the activity check failed.
            last_activity = time.monotonic()
            print("Monitor check failed: " + type(exc).__name__, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
