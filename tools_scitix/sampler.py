import json

# OUTPUTS_DIR = "outputs/compass-bench-v1.3-t-eval/20251029_152546"
OUTPUTS_DIR = "outputs/compass-bench-v1.3-t-eval/20251029_163508"
PREDS_DIR = f"{OUTPUTS_DIR}/predictions"
RESULTS_DIR = f"{OUTPUTS_DIR}/results"

NUM_SAMPLES = 20
DATASETS = [
    "compass-bench-v1.3-t-eval-plan-en-legacy",
    "compass-bench-v1.3-t-eval-before-calling-en-legacy",
    "compass-bench-v1.3-t-eval-at-calling-en-legacy",
    "compass-bench-v1.3-t-eval-after-calling-en-legacy",
]
MODELS = [
    "Exp262",
    "Deepseek-V3.1",
    "DeepSeek-V3-0324",
    "Qwen2.5-72B-Instruct",
]

datasets = {d: [dict() for i in range(NUM_SAMPLES)] for d in DATASETS}
for dataset_name, samples in datasets.items():
    print(dataset_name)
    for model in MODELS:
        print(model)
        # pred
        with open(f"{PREDS_DIR}/{model}/{dataset_name}_0.json", encoding="utf-8") as fh:
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
        with open(f"{RESULTS_DIR}/{model}/{dataset_name}.json", encoding="utf-8") as fh:
            obj = json.load(fh)
            for i, v in enumerate(obj["details"]):
                if i >= NUM_SAMPLES:
                    break
                samples[i][model]["parsed_prediction"] = v["prediction"][0]

                metrics_to_extract = [
                    # objective
                    "correct",
                    # code eval
                    "msg",
                    # tool use - plan
                    "f1_score",
                    "parse_rate",
                    # tool use - before
                    "thought",
                    "name",
                    "args_f1_score",
                    "parse_rate",
                    # tool use - at
                    "string_format_metric",
                    "string_args_em_metric",
                    "json_format_metric",
                    "json_args_em_metric",
                    # tool use - after
                    "parse_rate",
                    "review_quality",
                ]
                for metric in metrics_to_extract:
                    if metric in v:
                        samples[i][model][metric] = v[metric][0]

    with open(f"{dataset_name}_10samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
