import json

OUTPUTS_DIR = "outputs/compass-academic-202510-simple/20251011_004222"
PREDS_DIR = f"{OUTPUTS_DIR}/predictions"
RESULTS_DIR = f"{OUTPUTS_DIR}/results"

NUM_SAMPLES = 10
DATASETS = [
    "aime-2025-n32",
    "gpqa-diamond_simple-evals",
    "ifeval",
    "lcb-code-generation-n6",
    "mmlu-pro",
]
MODELS = [
    "deepseek-v3.1",
    "deepseek-v3-exp91",
    "deepseek-v3-exp93",
    "deepseek-v3-exp103",
    "deepseek-v3-exp108",
    "deepseek-v3-exp110",
    "deepseek-v3-exp111",
    "deepseek-v3-exp260",
    "deepseek-v3-exp262",
]

datasets = {d: [dict() for i in range(NUM_SAMPLES)] for d in DATASETS}
for dataset_name, samples in datasets.items():
    print(dataset_name)
    for model in MODELS:
        print(model)
        # pred
        with open(f"{PREDS_DIR}/{model}/{dataset_name}_0.json") as fh:
            obj = json.load(fh)
            for i, (k, v) in enumerate(obj.items()):
                if i >= NUM_SAMPLES:
                    break
                samples[i]["origin_prompt"] = v["origin_prompt"]
                # some dataset does not have a golden standard
                if "gold" in v:
                    samples[i]["gold"] = v["gold"]
                samples[i][model] = {"prediction": v["prediction"]}
        # res
        with open(f"{RESULTS_DIR}/{model}/{dataset_name}.json") as fh:
            obj = json.load(fh)
            for i, v in enumerate(obj["details"]):
                if i >= NUM_SAMPLES:
                    break
                samples[i][model]["parsed_prediction"] = v["prediction"][0]
                # objective
                if "correct" in v:
                    samples[i][model]["correct"] = v["correct"][0]
                # code eval
                if "msg" in v:
                    samples[i][model]["msg"] = v["msg"][0]

    with open(f"{dataset_name}_10samples.json", "w") as f:
        json.dump(samples, f)
