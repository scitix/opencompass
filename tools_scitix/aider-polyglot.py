from siflow import SiFlow
from siflow.types import TaskVolume, TaskUserSelectedInstance

client = SiFlow(
    region="cn-shanghai",
    cluster="mhysh",
    access_key_id="",
    access_key_secret="",
)

MODELS = [
    ("deepseek-v3-exp91", "172.16.161.31:30000"),
    ("deepseek-v3-exp93", "172.16.82.68:30000"),
    ("deepseek-v3-exp103", "172.16.148.82:30000"),
    ("deepseek-v3-exp108", "172.16.45.236:30000"),
    ("deepseek-v3-exp110", "172.16.177.180:30000"),
    ("deepseek-v3-exp111", "172.16.65.148:30000"),
    ("deepseek-v3-exp260", "172.16.189.143:30000"),
    ("deepseek-v3-exp262", "172.16.63.126:30000"),
    ("deepseek-v3-1", "eval-deepseek-v3-1.t-ai-infra-ylsun.svc"),
    # ("deepseek-v3-1", "172.16.239.229:30000"),  # replica1
    # ("deepseek-v3-1", "172.16.176.245:30000"),  # replica2
]
for model, model_api_base in MODELS:
    cmd = f"""
set -euo pipefail

export AIDER_DOCKER=1
export AIDER_BENCHMARK_DIR="/everything/ylsun/aider-benchmarks"
export OPENAI_API_BASE="http://{model_api_base}/v1"
export OPENAI_API_KEY="EMPTY"

./benchmark/benchmark.py aider-polyglot \
  --model "openai/{model}" \
  --edit-format diff \
  --exercises-dir polyglot-benchmark \
  --num-tests -1 \
  --threads 8 \
  --new
""".strip()
    uuid = client.tasks.create(
        name_prefix=f"aider-polyglot-{model}",
        image="code-evaluator",
        image_version="aider-0.86.2",
        image_url="registry-cn-shanghai.siflow.cn/ai-infra/code-evaluator:aider-0.86.2-dev",
        image_type="custom",
        type="pytorchjob",
        priority="medium",
        cmd=cmd,
        workers=0,
        resource_pool="cn-shanghai-mhysh-ai-infra-ondemand-shared",
        instances=[
            TaskUserSelectedInstance(name="sci.c22-2", count_per_pod=16),
        ],
        volumes=[
            TaskVolume(volume_name="everything", mount_dir="/everything"),
        ],
    )
    print(uuid)
