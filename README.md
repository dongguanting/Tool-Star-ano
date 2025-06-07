

<h1 align="center"> 🔧✨Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning</a></h1>






# 🏃 Quick Start for Training

## ❄️ Cold-Start SFT Stage

### 1. Environment Setup

In this step, we will describe how to perform a cold start for the SFT stage using the Llama Factory repository. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

### 2. Fine-Tuning Model


1. Download your SFT dataset, and place it in your folder. Define the dataset in `dataset_info.json`.

2. Complete the path information in `LLaMA-Factory-main/examples/train_full/qwen_sft_tool_star.yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: {your_path_to_model}/Qwen2.5-3B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: your_dataset_name
template: qwen
cutoff_len: 15000
max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {your_save_path}/Qwen2.5-3B-Instruct-sft
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 7.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

After completing the information, you can fine-tune the model using the following command:

```python
cd LLaMA-Factory-main
bash ./examples/train_full/train_sft.sh
```

---

## 🔥 Self-Critic RL Stage

In this step, we will load the cold-start data for GRPO training. We reference the [ReCall](https://github.com/Agent-RL/ReCall) and [VERL](https://github.com/volcengine/verl) frameworks for RL training.


### 1. Environment Setup

 you can install our additional environment as follow: 

```bash
#create env
conda create -n toolstar python==3.10
conda activate toolstar

# install torch & flash-atten
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# install RL basic env
cd Tool_Star_RL
pip3 install -e .

# This is our RL env freeze file. You can install it as a supplement or use it for checking.
pip install -r ./Tool-Star-main/requirements.txt

```
Please refer to requirements.txt carefully. It is important to note that **vLLM<= 0.6.3 and torch==2.4.0 (seem versions will not work.)**. You can also install a compatible flash_attention package from [here](https://github.com/Dao-AILab/flash-attention/releases).

If you encounter ray or other RL environment issues, we **highly recommend that you first try to run the RL training code for [ReCall](https://github.com/Agent-RL/ReCall/tree/re-search) or [Verl](https://github.com/volcengine/verl) successfully**, then further aligning with our requirements.txt.



### 2. Vanilla RL Training

Our training framework is based on [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). The training scripts can be found under `scripts/train`. First, you need to complete the information in `scripts/train/run_tool_star.sh`, 
we have provided both train parquet and test parquet for RL:

```bash
export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {your_actor_model_path} \
    --project_name {your_project_name} \
    --experiment_name {your_experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key {your_wandb_api_key} \
    --save_path {your_save_path} \
    --train_files {path_to_train_file}/your_train_file.parquet \
    --test_files {path_to_test_file}/your_test_file.parquet
```

Since the rollout process involves Bing web search calls, please configure the `deep_search_snippet()` function in `/src/verl/verl/workers/rollout/vllm_rollout/web_search/web_search_main.py` with your search API:

```python
def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key, 
        bing_endpoint=bing_endpoint, 
        eval=False,
        seed=123456789,
        concurrent_limit=200
    )
```

Replace `bing_subscription_key`, `bing_endpoint`, and `api_base_url` with your own values. Various web search modes are provided in this file for you to choose from.

You can then run the following script to start training:

```bash
cd ./Tool_Star_RL/scripts/train/
bash run_tool_star.sh
```

For the core code of the rollout process, please refer to `/src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, and for the reward calculation part, refer to `/Tool_Star_RL/src/verl/verl/utils/reward_score`. You can modify them according to your needs.

For the trained RL checkpoint, you can follow the code below to convert the weights to Hugging Face format：
```bash
# Merge RL weights and save in the same path.
python /Tool_Star_RL/model_merger.py \
    --local_dir /{your_checkpoint_path}/global_step_{your_RL_step}/actor/ \
```


### 3. Self-Critic DPO Training (Optional)

In our experiments, completing SFT + Vanilla RL has been sufficient to almost reproduce Tool-Star's performance (refer to the ablation study).

If you wish to proceed with Self-Critic DPO training, please refer to the training algorithm in **Appendix B.1** of the paper and the data format process in **Appendix E.2**. You can self-sample reward data using the saved checkpoints for RL and SFT training data. We also provide DPO training code based on [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for your reference.

Please complete the path information in `LLaMA-Factory-main/examples/train_lora/qwen_lora_dpo.yaml` and place the synthesized DPO data in `LLaMA-Factory-main/data/`. You can then run the following script for training:

```bash
cd LLaMA-Factory-main
bash ./examples/train_lora/train_dpo.sh
```

---

## ✅ TIR Evaluation

If you have already trained a model, you can refer to the following process for TIR capability evaluation. Of course, you can also download our checkpoint 🤗Tool-Star-Qwen-3B for directly testing.

### 1. Environment Setup

```bash
# Create conda environment
conda create -n tool_star python=3.10
conda activate tool_star

# Install requirements
cd Tool-Star-main
pip install -r requirements.txt
```

### 2. LLM Service Deployment

In this step, we will use the VLLM framework to deploy additional large language models (LLMs). This includes deploying an LLM as a judging model to evaluate the accuracy of the generated answers in the subsequent steps, as well as deploying inference-time tools such as code debugging and chain refinement.

- We use Qwen2.5-72B-Instruct as the judging model.

- We use Qwen2.5-3B-Instruct, which has the same parameter scale as the base model, as the foundation for the inference-time tools.

For the specific deployment, you can refer to the following script.

```bash
cd evaluation
bash vllm_server.sh
```

### 3. Retriever Serving Deployment

In this section, we will deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need to download the pre-indexed Wikipedia, Wikipedia corpus, and corresponding retriever models. The corpuses used can be found here, and Index construction method can be found here.

More details can be found in the FlashRAG documentation.

To start the retriever serving, first fill in `scripts/serving/retriever_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command to start the retriever serving:

```bash
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

### 4. Inference Your Model

In this section, we infer answers using a trained model. We support five types of mathematical reasoning datasets: AIME24, AIME25, GSM8K, MATH, and MATH500, as well as seven QA reasoning datasets: WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE. Due to resource constraints, all models and baselines will test a maximum of 500 samples for mathematical reasoning, 200 samples for all QA datasets, and 500 samples for HLE (please refer our code).

First, replace the API_URL and API key with your own in the following files:

In `evaluation/utils.py`:

```python
def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'your_search_api_url'
    ...

def batch_search(query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'your_search_api_url'
    ...
```

In `evaluation/tools/web_search_main.py`:

```python
def deep_search(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="xxxxx", bing_endpoint="xxxxx/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  
        bing_endpoint=bing_endpoint,  
        eval=False,
        seed=12345678,
        api_base_url='xxxxx',  
        model_name='search-agent',
        concurrent_limit=200
    )
    ...
```

In `evaluation/tools/debug_code.py`:

```python
def debug_code_function(code, error, api_key="your_api_key"):

    API_BASE_URL = api_key
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

Then, start the inference. We recommend that you use the default parameters as:

```bash
cd evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/path/to/your_path:$PYTHONPATH
module load cuda/11.8
python run.py \
    --model_path /path/to/your_model_path \
    --dataset_name math \
    --task math \
    --gpu_use 0.95 \
    --max_tokens 16384 \ #you can change this, 8192 is enough for most tasks
    --max_input_len 16384 \ #you can change this, 8192 is enough for most tasks
    --output_path /path/to/your_results/your_exp_math_result.json \
    --counts 500 \
    --batch_size 100 \
    --use_debug 
```

**Parameter Explanations:**
- `--model_path`: Path to your model.
- `--dataset_name`: Name of your dataset (supports AIME24, AIME25, GSM8K, MATH, MATH500, WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE).
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--gpu_use`: GPU memory utilization.
- `--max_tokens`: Maximum number of tokens the model can generate.
- `--max_input_len`: Maximum input tokens the model can accept.
- `--output_path`: Path to save the results.
- `--counts`: Number of samples to take from the test set during testing.
- `--batch_size`: Batch size for parallel inference.
- `--use_debug`: Enable the debug mechanism.

**Additional Parameters（Optional）:**

In practical, only in the cases of HLE and GAIA is there a possibility of exceeding the length limit, you can use refiner. Generally, it won't occur in other situations.  

- `--use_rollback`: Whether to use the rollback mechanism.
- `--use_refiner`: Whether to use the refine mechanism.


In `evaluation/tools/refine_code.py`:

```python
def refine(prompt, response):

    API_BASE_URL = "your_api_base_url"
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

### 5. Calculate Metrics

First, replace the API URL and API key with your own in the following file:

In `evaluation/evaluate/scripts/evaluate.py`:

```python
async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = "xxxxxxx"
    if model_name is None:
        model_name = "Qwen2.5-72B-Instruct"
    ...
```

Replace `api_base_url` with the API_URL of your deployed model.

Then, run the following command:

```bash
cd evaluation
python evaluate/scripts/evaluate.py \
    --output_path /path/to/your_results/your_exp_math_result.json \
    --task math \
    --dataset_name math \
    --use_llm \
    --extract_answer
```

**Parameter Explanations:**
- `--output_path`: Path to save the results.
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--dataset_name`: Name of your dataset.
- `--use_llm`: Whether to use the LLM-as-judge mechanism.
- `--extract_answer`: Whether to use exact matching (removes \text and other redundant symbols).






## 📄 License

This project is released under the [MIT License](LICENSE).
