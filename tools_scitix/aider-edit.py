from siflow import SiFlow
from siflow.types import TaskVolume, TaskUserSelectedInstance

client = SiFlow(
    region="cn-shanghai",
    cluster="mhysh",
    access_key_id="",
    access_key_secret="",
)

MODELS = [
    "deepseek-v3-exp91",
    "deepseek-v3-exp93",
    "deepseek-v3-exp103",
    "deepseek-v3-exp108",
    "deepseek-v3-exp110",
    "deepseek-v3-exp111",
    "deepseek-v3-exp260",
    "deepseek-v3-exp262",
    "deepseek-v3-1",
]
for model in MODELS:
    cmd = f"""
set -euo pipefail

export AIDER_DOCKER=1
export AIDER_BENCHMARK_DIR="/everything/ylsun/aider-benchmarks"
export OPENAI_API_BASE="http://eval-{model}.t-ai-infra-ylsun.svc/v1"
export OPENAI_API_KEY="EMPTY"

./benchmark/benchmark.py aider-edit \
  --model "openai/{model}" \
  --edit-format diff \
  --exercises-dir edit-benchmark \
  --languages python \
  --num-tests -1 \
  --threads 8 \
  --new
""".strip()
    uuid = client.tasks.create(
        name_prefix=f"aider-edit-{model}",
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
            TaskUserSelectedInstance(name="sci.c22-2", count_per_pod=12),
        ],
        volumes=[
            TaskVolume(volume_name="everything", mount_dir="/everything"),
        ],
    )
    print(uuid)
