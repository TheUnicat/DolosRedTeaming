"""A GPU on RunPod for this repo. The local repo is copied to the pod; jobs run there with uv.

    python tools/gpu.py up               rent a pod (RTX A4500, else RTX 4000 Ada, else A40), or start
                                         the stopped one; install the uv environment, copy the repo,
                                         and start the idle monitor
    python tools/gpu.py status           the pod, its GPU, price and SSH address
    python tools/gpu.py ssh              an interactive shell on the pod
    python tools/gpu.py sync             copy the repo's files over again
    python tools/gpu.py run <command>    sync, then run the command in the pod's repo (e.g. 'uv run neuron_diff.py');
                                         --detach keeps it running after you disconnect, logging to /workspace/logs
    python tools/gpu.py fetch <path>     copy a file from the pod's repo to the same path here
    python tools/gpu.py add-key <key>    let someone else SSH in (an OpenSSH public key, or a file holding one);
                                         prints the command to send them
    python tools/gpu.py down             stop the pod: the /workspace disk stays, the GPU meter stops
    python tools/gpu.py delete           delete the pod and its disk

The RunPod key is read from ~/.runpod_api_key and this machine's SSH key from ~/.ssh/id_ed25519.
Extra SSH keys added with add-key are kept in data/ssh_keys and re-installed on every `up`, since the
pod's home directory does not survive a stop. The idle monitor stops the pod after two hours with no
GPU work and no SSH session, using the pod's own scoped key, so no account key is copied to the pod.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
KEY = Path.home() / ".runpod_api_key"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
NAME = "dolos-redteaming"
STATE_FILE = ROOT / "data" / "gpu.json"
EXTRA_KEYS = ROOT / "data" / "ssh_keys"
# Tried in this order; the price cap guards against an unexpected allocation. No CUDA version filter:
# RunPod found no instances with one on 2026-10-24, and the CUDA check in `up` catches an unusable host.
GPUS = ["NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A40"]
MAX_PRICE = 0.65
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # for sshd and Python; torch comes from uv
WORK = "/workspace"  # the pod's volume, kept across stops
REPO = f"{WORK}/{NAME}"
ENV = f"export HF_HOME={WORK}/hf UV_CACHE_DIR={WORK}/uv-cache PATH=$HOME/.local/bin:$PATH"
SETUP = (f"set -e; {ENV}; mkdir -p {REPO} {WORK}/hf {WORK}/logs; "
         "command -v uv >/dev/null || python -m pip install -q --user uv; "
         f"cd {REPO} && uv sync --locked -q")
HEALTH = """import torch
assert torch.cuda.is_available(), 'no CUDA'
print(torch.__version__, torch.cuda.get_device_name(), 'ok')"""
IDLE = f"{WORK}/.dolos-idle"


def api(method, path, body=None):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {KEY.read_text().strip()}",
                                          "Content-Type": "application/json", "User-Agent": NAME})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            text = r.read().decode()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "GET":
            return None
        sys.exit(f"RunPod said {e.code} to {method} {path}: {e.read().decode()[:400] or 'nothing'}")


def pod():
    if not STATE_FILE.exists():
        return None
    return api("GET", f"/pods/{json.loads(STATE_FILE.read_text())['pod_id']}")


def address(p):
    port = (p.get("portMappings") or {}).get("22")
    return (p.get("publicIp"), port) if p.get("publicIp") and port else None


def ssh_args(p, tty=False):
    host, port = address(p)
    return ["ssh", *(["-t"] if tty else []), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-i", str(SSH_KEY), "-p", str(port), f"root@{host}"]


def ssh(p, command, quiet=False, stdin=None, tty=False):
    return subprocess.run(ssh_args(p, tty) + ([command] if command else []), stdin=stdin,
                          stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.DEVNULL if quiet else None).returncode


def public_keys():
    keys = [SSH_KEY.with_suffix(".pub").read_text().strip()]
    if EXTRA_KEYS.exists():
        keys += [k.strip() for k in EXTRA_KEYS.read_text().splitlines() if k.strip()]
    return keys


def up():
    p = pod()
    if p is None:
        print(f"renting the first available of: {', '.join(GPUS)}", flush=True)
        p = api("POST", "/pods", {
            "name": NAME, "imageName": IMAGE, "cloudType": "SECURE", "computeType": "GPU",
            "gpuCount": 1, "gpuTypeIds": GPUS, "gpuTypePriority": "custom",
            "containerDiskInGb": 30, "volumeInGb": 50, "volumeMountPath": WORK, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": "\n".join(public_keys())}})
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"pod_id": p["id"], "gpu": (p.get("machine") or {}).get("gpuTypeId")}))
        if p.get("costPerHr", 999) > MAX_PRICE:
            api("POST", f"/pods/{p['id']}/stop")
            sys.exit(f"Got ${p.get('costPerHr')}/h, above the ${MAX_PRICE} cap; stopped the new pod.")
    elif p.get("desiredStatus") != "RUNNING":
        print("starting the pod", flush=True)
        api("POST", f"/pods/{p['id']}/start")
    for _ in range(120):
        p = api("GET", f"/pods/{p['id']}")
        if p.get("desiredStatus") == "RUNNING" and address(p) and ssh(p, "true", quiet=True) == 0:
            break
        time.sleep(5)
    else:
        sys.exit("The pod did not come up in time; `python tools/gpu.py status` says where it is.")
    install_keys(p)
    sync(p)
    print("installing the uv environment (first time takes a few minutes)", flush=True)
    if ssh(p, SETUP):
        sys.exit("Setup failed; the pod is still running for diagnosis (`python tools/gpu.py ssh`).")
    if ssh(p, f"{ENV}; cd {REPO} && uv run --locked python -c {shlex.quote(HEALTH)}"):
        sys.exit("CUDA check failed; the pod is still running. Inspect it, or `delete` and rent again.")
    install_idle(p)
    status(p)


def install_keys(p):
    """Append any missing public keys to the pod's authorized_keys."""
    script = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
              "while read -r k; do grep -qxF \"$k\" ~/.ssh/authorized_keys || echo \"$k\" >> ~/.ssh/authorized_keys; done "
              "&& chmod 600 ~/.ssh/authorized_keys")
    subprocess.run(ssh_args(p) + [script], input="\n".join(public_keys()) + "\n", text=True, check=True)


def sync(p):
    """The repo's files, tracked or not yet (minus ignored ones), as a tar stream onto the pod."""
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, check=True).stdout
    tar = subprocess.Popen(["tar", "--null", "-T", "-", "-czf", "-"], cwd=ROOT, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE)
    tar.stdin.write(files)
    tar.stdin.close()
    code = ssh(p, f"mkdir -p {REPO} && tar --no-same-owner -xzf - -C {REPO}", stdin=tar.stdout)
    tar.stdout.close()
    if code or tar.wait():
        sys.exit("Copying the repo over failed.")


def install_idle(p):
    command = (f"mkdir -p {IDLE} && {{ setsid nohup python3 -u {REPO}/tools/idle_gpu.py --directory {IDLE} "
               f">> {IDLE}/monitor.log 2>&1 < /dev/null & }}")
    if ssh(p, command):
        sys.exit("Could not start the idle monitor.")


def run(command, detach=False):
    p = pod()
    if p is None or p.get("desiredStatus") != "RUNNING" or not address(p):
        sys.exit("No running pod; `python tools/gpu.py up` first.")
    sync(p)
    script = f"{ENV}; cd {REPO} && {command}"
    if not detach:
        return ssh(p, script, tty=True)
    log = f"{WORK}/logs/{time.strftime('%Y%m%d-%H%M%S')}.log"
    code = ssh(p, f"setsid nohup bash -c {shlex.quote(script)} > {log} 2>&1 < /dev/null &")
    print(f"detached; follow it with: python tools/gpu.py run 'tail -f {log}'")
    return code


def fetch(path):
    p = pod()
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        code = subprocess.run(ssh_args(p) + [f"cat {shlex.quote(f'{REPO}/{path}')}"], stdout=out).returncode
    if code:
        target.unlink(missing_ok=True)
        sys.exit(f"Could not fetch {path}.")
    print(f"fetched {path}")


def add_key(key):
    key = Path(key).read_text().strip() if Path(key).is_file() else key.strip()
    if not key.startswith(("ssh-", "ecdsa-", "sk-")):
        sys.exit("That does not look like an OpenSSH public key (it should start with ssh-ed25519, ssh-rsa, ...).")
    EXTRA_KEYS.parent.mkdir(parents=True, exist_ok=True)
    existing = EXTRA_KEYS.read_text().splitlines() if EXTRA_KEYS.exists() else []
    if key not in existing:
        EXTRA_KEYS.write_text("\n".join(existing + [key]) + "\n")
    p = pod()
    if p and p.get("desiredStatus") == "RUNNING" and address(p):
        install_keys(p)
        host, port = address(p)
        print(f"Added. Send them:\n\n    ssh root@{host} -p {port}\n\n"
              "(add -i ~/.ssh/<their private key> if it is not their default key). The address changes if the "
              "pod is stopped and started again; `python tools/gpu.py status` shows the current one.")
    else:
        print("Saved; it will be installed on the next `python tools/gpu.py up`.")


def status(p=None):
    p = p or pod()
    if p is None:
        print("no pod")
        return
    p = api("GET", f"/pods/{p['id']}")
    gpu = json.loads(STATE_FILE.read_text()).get("gpu") or "GPU unknown"
    where = address(p)
    print(f"pod {p['id']}: {p.get('desiredStatus')}, {gpu}, ${p.get('costPerHr', 0):.2f}/h"
          + (f", ssh root@{where[0]} -p {where[1]}" if where else ", no SSH address yet"))


def down():
    p = pod()
    if p is None or p.get("desiredStatus") != "RUNNING":
        print("nothing running")
        return
    api("POST", f"/pods/{p['id']}/stop")
    print("stopped: the disk stays, the GPU meter does not")


def delete():
    p = pod()
    if p is None:
        print("no pod")
        return
    api("DELETE", f"/pods/{p['id']}")
    STATE_FILE.unlink(missing_ok=True)
    print(f"deleted pod {p['id']} and its disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action")
    for name in ("up", "status", "ssh", "sync", "down", "delete"):
        sub.add_parser(name)
    launch = sub.add_parser("run")
    launch.add_argument("--detach", action="store_true")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("fetch").add_argument("path")
    sub.add_parser("add-key").add_argument("key")
    args = parser.parse_args()
    what = args.action or "status"
    if what == "up":
        up()
    elif what == "status":
        status()
    elif what == "ssh":
        sys.exit(ssh(pod(), None, tty=True))
    elif what == "sync":
        sync(pod())
    elif what == "run":
        sys.exit(run(" ".join(args.command), args.detach))
    elif what == "fetch":
        fetch(args.path)
    elif what == "add-key":
        add_key(args.key)
    elif what == "down":
        down()
    elif what == "delete":
        delete()
