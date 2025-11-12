import os, glob, subprocess
from pathlib import Path

TIMEOUT_SEC = 180

def parse_metrics(stderr: str):
    err_rate = acc = ""
    for line in reversed(stderr.splitlines()):
        if not err_rate and "Error Rate:" in line:
            err_rate = line.split("Error Rate:", 1)[1].strip()
        if not acc and "Accuracy:" in line:
            acc = line.split("Accuracy:", 1)[1].strip()
        if err_rate and acc:
            break
    return err_rate, acc

def run_one(gen_model: str, spam_model: str, test_glob: str, prior: float):
    test_files = sorted(glob.glob(test_glob))
    if not test_files:
        raise RuntimeError(f"No test files matched {test_glob!r}")

    cmd = ["./textcat.py", gen_model, spam_model, str(prior), *test_files]

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_NUM_THREADS", "1")

    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=os.getcwd(),
        env=env, timeout=TIMEOUT_SEC
    )
    err_rate, acc = parse_metrics(proc.stderr)

    return {
        "gen_model": gen_model,
        "spam_model": spam_model,
        "returncode": proc.returncode,
        "error_rate": err_rate,
        "accuracy": acc,
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-5:]),
    }