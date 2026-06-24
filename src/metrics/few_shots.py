import os
import subprocess
import sys


DEFAULT_FEW_SHOT_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "openbookqa",
    "piqa",
    "truthfulqa",
)


def eval_few_shots(
    model_name,
    task_list=None,
    output_path=None,
    output_dir=None,
    batch_size=16,
    cache_dir="./.cache",
    device=None,
):
    if task_list is None:
        task_list = DEFAULT_FEW_SHOT_TASKS
    if isinstance(task_list, str):
        task_list = [task.strip() for task in task_list.split(",") if task.strip()]
    if not task_list:
        raise ValueError("eval_few_shots requires at least one lm-eval task")

    if output_path is None:
        if output_dir is not None:
            output_path = os.path.join(output_dir, "few_shots.json")
        else:
            output_path = "."
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    command = [sys.executable, "-m", "lm_eval"]
    tasks = ",".join(task_list)
    model_args = f"pretrained={model_name},cache_dir={cache_dir},dtype=auto"
    args = [
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        f"{tasks}",
        "--batch_size",
        str(batch_size),
        "--output_path",
        f"{output_path}",
    ]
    if device is not None:
        args.extend(["--device", str(device)])

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    harness_root = os.path.join(repo_root, "lm-evaluation-harness")
    env = os.environ.copy()
    if os.path.isdir(harness_root):
        env["PYTHONPATH"] = os.pathsep.join(
            [harness_root, env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

    full_command = command + args
    print(f"[lm-eval] Running tasks={tasks} output={output_path}")
    subprocess.run(full_command, check=True, env=env)
    return output_path
