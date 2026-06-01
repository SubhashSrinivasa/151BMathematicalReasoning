# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |


## GPU Type 
We used A100 and it takes about 6 hours inference on the full private dataset. 

## Downloading the model

- follow the steps in the final_inference.py as it includes model downloading
- as a fall back, run this script:
"huggingface-cli download ig123/Susi-Qwen3-4b-Thinking-2507 \
    --local-dir ./models/Susi-Qwen3-4b-Thinking-2507"

## Running run_inference()
run "python final_inference.py --output my_submission.csv"

## Running End to End Pipeline 
### Step 1: SFT Training
Run "python train_sft.py"

The LoRA adapters are saved in this file path: data/lora_math_adapter/final_adapter

### Step 2: RL Training 
python train_RL.py --init-adapter data/lora_math_adapter/final_adapter

The adapter is located in checkpoints/qwen3-4b-grpo-l40s-skip-unboxed

To save the adapter: 
1. huggingface-cli login
2. from peft import PeftModel
   model.push_to_hub("your-username/your-model-name")

### Step 3: Inference 
To run the model you create run - python final_inference.py --hf-model your-username/your-model-name --output my_submission.csv
To run **our** model run - python final_inference.py --output my_submission.csv



