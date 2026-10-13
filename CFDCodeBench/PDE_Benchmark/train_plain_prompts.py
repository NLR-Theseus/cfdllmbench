from utils import PromptGenerator, LLMCodeGenerator, Evaluator
import os
import pandas as pd

# STEP 1: generate the prompts
# Set the root directory (you can also use os.getcwd() if running from root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(ROOT_DIR, 'PDE_Benchmark')  # PDE_Benchmark root
json_file = "prompts.json"  # do not use mms
# json_file = "mms_prompts.json"  # use mms
# Initialize the PromptGenerator
generator_prompt = PromptGenerator(root_dir, json_file)

# Load the problem data from the input JSON
generator_prompt.load_problem_data()

# Create prompts from the data
prompts = generator_prompt.create_prompts()

# Save to prompts.json (will skip if already exists)
generator_prompt.save_prompts(prompts)

# STEP 2: call LLM API to get generated code (needs the API keys described in utils.api_key_configuration)
GENERATE = False
# STEP 3: run every generated script and score execution, NMSE and convergence
EVALUATE = True

# List of LLM models to evaluate
llm_models = [
    "gpt-4o",
    "o3-mini",
    "gemini",
    "sonnet-35",
    "haiku",
]

prompt_json = json_file  # the file under ./prompt/
summaries = []
# Loop over all models
for llm_model in llm_models:
    print(f"\n=== Running for model: {llm_model} ===\n")
    print(f"\n=== Running for prompt: {json_file} ===")
    if GENERATE:
        # Call the API to generate code for each task
        LLMCodeGenerator(llm_model, prompt_json).call_api()
    if EVALUATE:
        # Execution (exit code 0 within the time limit), NMSE against results/solution, and a convergence test
        # (re-run with the declared resolution variables refined). Per-task results: eval/<model>/<prompt>/.
        _, summary = Evaluator(llm_model, prompt_json).run()
        summaries.append(summary)

if summaries:
    os.makedirs(os.path.join(root_dir, "eval"), exist_ok=True)
    pd.DataFrame(summaries).to_csv(os.path.join(root_dir, "eval", "summary_all_models.csv"), index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
