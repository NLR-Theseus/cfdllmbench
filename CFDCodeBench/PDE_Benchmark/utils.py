import json
import re
import subprocess
import boto3
from openai import OpenAI
import logging
from datetime import datetime
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import matplotlib.pyplot as plt
import ast
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
# from google import genai
# from google.genai import types


def generate_prompt(data):
    parts = [
        "You are given the following partial differential equation (PDE) problem:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Boundary Conditions:**\n" + data.get("boundary conditions", "") + "\n",
        "**Initial Conditions:**\n" + data.get("initial conditions", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n",
        "**Numerical Method:**\n" + data.get("numerical method", "") + "\n"
    ]

    # Variables to save. v1 task files used the key "save values" (with a space), which this function never
    # read, so no prompt listed them; accept both spellings.
    save_values = data.get("save_values", data.get("save values", []))
    if isinstance(save_values, str):
        save_values = [save_values]
    files_str = ", ".join(f"`{v}.npy`" for v in save_values)
    res_vars = list(data.get("resolution_variables", {}))
    res_str = ", ".join(f"`{v}`" for v in res_vars)
    time_step_rule = (" Compute the time step as dt = T_final / nt so that the final time is reached exactly."
                      if "nt" in res_vars else "")
    # Always end with task specification for the code
    parts.append(
        "### Task:\n"
        "- Write Python code to numerically solve the given CFD problem. Choose an appropriate numerical method based "
        "on the problem characteristics.\n"
        "- If the problem is **unsteady**, only compute and save the **solution at the final time**.\n"
        f"- **Save the following variables**, each in its own file in the current working directory: {files_str}.\n"
        f"- {data.get('output', '')}\n"
        + (f"- Define the resolution as integer variables {res_str}, each assigned a literal value at the top level of "
           if len(res_vars) > 1 else
           f"- Define the resolution as an integer variable {res_str} assigned a literal value at the top level of ")
        + f"the script (for example `{res_vars[0] if res_vars else 'nx'} = 101`).{time_step_rule} The benchmark re-runs "
        "the script with these values refined to test convergence.\n"
        "- **Return only the complete, runnable Python code** that implements the above tasks, "
        "ensuring no extra explanations or information is included."
    )

    return "\n".join(parts)


def generate_mms_prompt(data):
    parts = [
        "You are given the following partial differential equation (PDE) to test using the **Method of Manufactured "
        "Solutions (MMS)**:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n",
        "**Numerical Method:**\n" + data.get("numerical method", "") + "\n"
    ]

    # Check for 'save_values' and add to task description
    save_values = data.get("save_values", [])
    save_values_str = ", ".join(save_values) if save_values else "the relevant solution variables"

    parts.append(
        "### Task:\n"
        "- Choose a smooth manufactured solution \( u(x, t) \) that fits the domain.\n"
        "- Derive a source term \( f(x, t) \) such that the modified PDE:\n"
        "  \n    Original_PDE + f(x, t) = 0\n"
        "  \n  holds true for the chosen manufactured solution.\n"
        "- Implement a numerical solver using the specified method to solve the PDE with the source term.\n"
        "- At the final time step, compare the numerical result to the exact (manufactured) solution.\n"
        "- For each variable in the list below, save a `.npy` file named after the variable (e.g., `u.npy`, `p.npy`).\n"
        f"  - Variables to save: {save_values_str}\n"
        "- Each `.npy` file should contain a Python dictionary with two entries:\n"
        "  \n    `{'numerical': ..., 'MMS': ...}`\n"
        "  \n  where `'numerical'` is the computed solution and `'MMS'` is the exact solution.\n"
        "- Example: for variable `u`, save using `np.save('u.npy', {'numerical': u_numeric, 'MMS': u_exact})`\n"
        "- **Return only the complete, runnable Python code** that performs all steps: MMS setup, solver, comparison, "
        "and saving."
    )

    return "\n".join(parts)


class PromptGenerator:
    def __init__(self, root_dir, json_file):
        self.problems = None
        self.root_dir = root_dir
        self.input_json = os.path.join(root_dir, 'prompt/PDE_TASK_QUESTION_ONLY.json')
        self.output_json = os.path.join(root_dir, f'prompt/{json_file}')
        self.json_file = json_file

    def load_problem_data(self):
        with open(self.input_json, "r") as f:
            self.problems = json.load(f)

    def create_prompts(self):
        output_data = {}
        for name, data in self.problems.items():
            if self.json_file == "prompts.json":
                output_data[name] = generate_prompt(data)
            elif self.json_file == "mms_prompts.json":
                output_data[name] = generate_mms_prompt(data)
            else:
                raise ValueError(f"Unsupported prompt type: {self.json_file}")
        return {"prompts": output_data}  # <-- wrap in top-level "prompts" key

    def save_prompts(self, prompts):
        if os.path.exists(self.output_json):
            print(f"Skipped: {self.output_json} already exists.")
        else:
            with open(self.output_json, "w") as f:
                json.dump(prompts, f, indent=2)
            print(f"Created: {self.output_json}")


# === Function to Execute Python Script and Capture Errors and Warnings ===
def execute_python_script(filepath, timeout=60):
    try:
        result = subprocess.run(["python3", filepath], capture_output=True, text=True, timeout=timeout,
                                cwd=os.path.dirname(os.path.abspath(filepath)))
        stderr_output = result.stderr.strip()

        if result.returncode == 0:
            if "warning" in stderr_output.lower():
                logging.warning(f"Execution completed with warnings:\n{stderr_output}")
                return f"⚠️ Execution completed with warnings:\n{stderr_output}", result
            else:
                logging.info("Execution successful, no errors detected.")
                return "Execution successful, no errors detected.", result

        logging.error(f"Execution failed with errors:\n{stderr_output}")
        return stderr_output, result

    except Exception as e:
        logging.error(f"❌ Unexpected error while running script {filepath}: {e}")
        return f"❌ Unexpected error: {e}", None


def build_system_prompt():
    # Shared instruction - system instruction
    instruction = ("You are a highly skilled assistant capable of generating Python code to solve CFD problems "
                   "using appropriate numerical methods."
                   "Given the problem description, you should reason through the problem and determine the best "
                   "approach for discretizing and solving it,"
                   "while respecting the specified boundary conditions, initial conditions, and domain.\n"
                   "For unsteady problems, save only the solution at the final time step. For 1D problems, "
                   "save a 1D array; for 2D problems, save a 2D array.\n"
                   "Ensure the code follows the user's specifications and saves the requested variables exactly "
                   "as named in `save_values`.\n"
                   "Your task is to generate the correct, fully runnable Python code for solving the problem "
                   "without additional explanations.")
    return instruction


def build_conversation(original_prompt, llm_model):
    # Shared instruction - system instruction
    instruction = build_system_prompt()

    # Prompt augmentation
    user_prompt = (original_prompt +
                   "\nIf it is an unsteady problem, only save the solution at the final time step. "
                   "If the problem is 1D, the saved array should be 1D. "
                   "If the problem is 2D, the saved array should be 2D.")
    # Determine system or user role for the instruction
    if llm_model in ["o3-mini", "sonnet-35", "haiku"]:
        conversation_history = [
            {"role": "user", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    elif llm_model == "gpt-4o":
        conversation_history = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    elif llm_model == "gemini":
        conversation_history = user_prompt
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return conversation_history


def extract_model_response(llm_model, response):
    # Extract model response
    if llm_model in ["gpt-4o", "o3-mini"]:
        model_response = response.choices[0].message.content.strip()
    elif llm_model in ["sonnet-35", "haiku"]:
        # Parse the response
        response_body = json.loads(response["body"].read().decode("utf-8"))

        # Extract model response
        if "content" in response_body:
            model_response = response_body["content"][0]["text"]
        else:
            model_response = "Error: No response received from model."
    elif llm_model == "gemini":
        model_response = response.text
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return model_response


def update_token_usage(llm_model, response, tokens_counts):
    if llm_model == "gpt-4o":
        usage = response.usage
        tokens_counts['total_input_tokens'] += usage.prompt_tokens
        tokens_counts['total_output_tokens'] += usage.completion_tokens
        cost = (usage.prompt_tokens / 1000) * 0.005 + (usage.completion_tokens / 1000) * 0.015
        tokens_counts['total_cost'] += cost

    elif llm_model == "o3-mini":
        # OpenAI's o3-mini (same cost as gpt-3.5-turbo)
        usage = response.usage
        tokens_counts['total_input_tokens'] += usage.prompt_tokens
        tokens_counts['total_output_tokens'] += usage.completion_tokens
        cost = (usage.prompt_tokens + usage.completion_tokens) / 1000 * 0.0005
        tokens_counts['total_cost'] += cost

    elif llm_model == "sonnet-35":
        # Anthropic Sonnet-3.5 (via Bedrock response metadata)
        metadata = response.get('usage', {})
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.015
        tokens_counts['total_cost'] += cost

    elif llm_model == "haiku":
        # Anthropic Haiku
        metadata = response.get('usage', {})
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.00025 + (output_tokens / 1000) * 0.00125
        tokens_counts['total_cost'] += cost

    elif llm_model == "gemini":
        # Gemini response has model_usage field
        metadata = response.usage_metadata
        input_tokens = metadata.prompt_token_count
        output_tokens = metadata.candidates_token_count
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.00025 + (output_tokens / 1000) * 0.0005
        tokens_counts['total_cost'] += cost

    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    # Log summary
    logging.info(f"[{llm_model}] Input Tokens: {tokens_counts['total_input_tokens']}, "
                 f"Output Tokens: {tokens_counts['total_output_tokens']}, "
                 f"Estimated Cost: ${tokens_counts['total_cost']:.4f}")


def extract_code(model_response):
    # Match ```python ...``` or ```...``` code blocks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", model_response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try parsing the whole thing as Python code
    try:
        ast.parse(model_response)
        return model_response.strip()
    except SyntaxError:
        pass

    # Extract "Python-like" lines
    lines = model_response.splitlines()
    python_lines = []
    for line in lines:
        if re.match(r'^\s*(import|from|def|class|for|if|while|print|#|@|[a-zA-Z_]+\s*=)', line):
            python_lines.append(line)

    if python_lines:
        return "\n".join(python_lines).strip()

    return "# No valid Python code extracted"


def save_model_outputs(task_name, output_folder, model_response):
    # Extract Python code:
    # start as ```python / ``` / or pure python code
    extracted_code = extract_code(model_response)

    # Save the full model response
    response_file = os.path.join(output_folder, f"{task_name}.txt")
    with open(response_file, "w") as txt_file:
        txt_file.write(model_response)

    # Save the extracted Python code
    script_path = os.path.join(output_folder, f"{task_name}.py")
    with open(script_path, "w") as py_file:
        py_file.write(extracted_code)

    print(f"✅ Code saved: {script_path}")

    return script_path


def execute_check_errors(llm_model, script_path, task_name, conversation_history):
    # Execute and check for errors
    execution_feedback, _ = execute_python_script(script_path)

    if "no errors detected" in execution_feedback:
        print(f"🎯 {task_name} executed successfully without syntax errors.")
        logging.info(f"🎯 {task_name} executed successfully without syntax errors.")
        return True  # Exit function if no errors

    else:
        print(f"❌ Error detected in {task_name}, refining prompt...")
        logging.info(f"❌ Error detected in {task_name}, refining prompt...")
        logging.info(
            f"\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it.")
        updated_prompt = f"[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it."

        # Add the refine prompt feedback to the conversation as input
        if llm_model == "gemini":
            conversation_history += updated_prompt
        else:
            conversation_history.append({"role": "user", "content": updated_prompt})


def call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime, inference_profile_arn):
    if llm_model == "o3-mini":
        # Call OpenAI o3-mini API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history
        )
    elif llm_model == "gpt-4o":
        # Call OpenAI GPT-4o API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history,
            temperature=temperature
        )
    elif llm_model in ["sonnet-35", "haiku"]:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": temperature,
            "messages": conversation_history
        }

        # Invoke AWS Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId=inference_profile_arn,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
    elif llm_model == "gemini":
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_history,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=build_system_prompt()
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")
    return response


def generate_code(llm_model, task_name, prompt, client, temperature, bedrock_runtime, inference_profile_arn,
                  output_folder, tokens_counts, max_retries=5):
    retries = 0
    original_prompt = prompt  # Keep the original prompt unchanged
    # Initialize an empty list to store the conversation history
    conversation_history = build_conversation(original_prompt, llm_model)

    while retries < max_retries:
        print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        try:
            response = call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime,
                                    inference_profile_arn)
            # log the input message
            logging.info(
                "---------------------------------INPUT TO LLM FIRST-----------------------------------------")
            logging.info(conversation_history)
            # log the LLM response
            logging.info(
                "------------------------------------LLM RESPONSE--------------------------------------------")
            logging.info(response)

            # Extract model response
            model_response = extract_model_response(llm_model, response)
            # Add the response to the conversation as input
            if llm_model == "gemini":
                conversation_history += model_response
            else:
                conversation_history.append({"role": "assistant", "content": model_response})
            logging.info(
                "---------------------------------INPUT TO LLM UPDATE----------------------------------------")
            logging.info(conversation_history)

            # tracking the token usage will return input and output tokens each round
            update_token_usage(llm_model, response, tokens_counts)

            # Extract Python code using regex, save the full model response and extracted python code
            script_path = save_model_outputs(task_name, output_folder, model_response)

            if retries + 1 >= max_retries:
                return  # one shot (or last attempt): no execution feedback; the Evaluator runs the code

            # Execute and check for errors
            if execute_check_errors(llm_model, script_path, task_name, conversation_history):
                return  # Exit function if no errors
            retries += 1

        except Exception as e:
            print(f"❌ API Call Error for {task_name}: {str(e)}")
            logging.info(f"❌ API Call Error for {task_name}: {str(e)}")
            return  # Stop retrying if API call fails

    print(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")
    logging.info(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")


def api_key_configuration(llm_model):
    # === OpenAI API Configuration ===
    if llm_model in ["gpt-4o", "o3-mini"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=api_key)
    elif llm_model == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")
        client = genai.Client(api_key=api_key)
    elif llm_model in ["sonnet-35", "haiku"]:
        api_key, client = None, None
    else:
        raise ValueError(f"Invalid {llm_model} api key")

    if llm_model == "sonnet-35":
        # Define Sonnet-3.5 profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w"
    elif llm_model == "haiku":
        # Define Haiku profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/56i8iq1vib3e"
    elif llm_model in ["o3-mini", "gpt-4o", "gemini"]:
        inference_profile_arn = None
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return api_key, client, inference_profile_arn


class LLMCodeGenerator:
    def __init__(self, llm_model, prompt_json, temperature=0.0, reviewer=False):
        # reviewer=False (default): one shot per task, as in CFDLLMBench V2. reviewer=True: up to 5 attempts, each
        # with the execution error of the previous one as feedback.
        self.llm_model = llm_model
        self.prompt_json = prompt_json
        self.temperature = temperature
        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")
        # Get current time with microseconds
        self.timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
        # === Paths ===
        # PDE_Benchmark root
        self.ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PDE_Benchmark')
        self.PROMPTS_FILE = os.path.join(self.ROOT_DIR, "prompt", prompt_json)
        self.OUTPUT_FOLDER = os.path.join(self.ROOT_DIR, f"solver/{llm_model}/{os.path.splitext(prompt_json)[0]}")
        self.REPORT_FOLDER = os.path.join(self.ROOT_DIR, 'report')
        self.LOG_FILE = os.path.join(self.REPORT_FOLDER, f"{llm_model}_{os.path.splitext(prompt_json)[0]}_"
                                                         f"{self.timestamp}.log")
        # Ensure the output directory exists
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.REPORT_FOLDER, exist_ok=True)

        # Tracking the total input and output tokens
        self.tokens_counts = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0}
        if reviewer:
            self.max_retries = 5
        else:
            self.max_retries = 1

        logging.basicConfig(
            filename=self.LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def call_api(self):
        logging.info(
            "####################################################################################################")
        logging.info(f"Using the {self.llm_model}, change temperature to {self.temperature}, "
                     f"use the prompt {self.prompt_json}")
        # === Get API credentials, client, and profile ===
        api_key, client, inference_profile_arn = api_key_configuration(self.llm_model)
        # Load prompts
        with open(self.PROMPTS_FILE, "r") as file:
            pde_prompts = json.load(file)

        # Loop through prompts and generate code
        for task_name, prompt in pde_prompts["prompts"].items():
            # if task_name not in ["Fully_Developed_Turbulent_Channel_Flow"]:
            #     continue
            generate_code(
                self.llm_model,
                task_name,
                prompt,
                client,
                self.temperature,
                self.bedrock_runtime,
                inference_profile_arn,
                self.OUTPUT_FOLDER,
                self.tokens_counts,
                max_retries=self.max_retries
            )

        print("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info(f"Total Input Tokens: {self.tokens_counts['total_input_tokens']}")
        logging.info(f"Total Output Tokens: {self.tokens_counts['total_output_tokens']}")
        logging.info(f"Total Estimated Cost: ${self.tokens_counts['total_cost']:.6f}")


def replacer_factory(base_name, save_dir):
    def replacer(match):
        var_name = match.group(3)  # u, v, p, etc.
        new_filename = f"{var_name}_{base_name}.npy"
        new_path = os.path.join(save_dir, new_filename).replace("\\", "/")  # Use Unix-style paths
        return f"np.save('{new_path}', {var_name})"

    return replacer


def call_post_process(generated_solvers_dir, save_dir):
    # === Paths ===
    folder_path = generated_solvers_dir
    os.makedirs(save_dir, exist_ok=True)

    # pattern = r"np\.save\((['\"])(.+?\.npy)\1\s*,\s*(\w+)\s*\)"
    # Match np.save("something", variable) — with or without .npy
    pattern = r"np\.save\((['\"])(.+?)\1\s*,\s*(\w+)\s*\)"

    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            py_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # e.g., "burgers_solver"

            with open(py_path, "r") as f:
                code = f.read()

            # Create a replacer for this specific file
            replacer = replacer_factory(base_name, save_dir)

            # Apply the replacement
            new_code, count = re.subn(pattern, replacer, code)

            if count > 0:
                with open(py_path, "w") as f:
                    f.write(new_code)
                print(f"✅ Updated {filename}: replaced {count} save path(s)")
            else:
                print(f"ℹ️ No np.save calls updated in {filename}")


def write_execute_results_to_log(log, script, result, status_counts):
    try:
        # Write execution results to log file
        log.write(f"--- Running: {script} ---\n")
        log.write(f"Exit Code: {result.returncode}\n")
        log.write("Output:\n")
        log.write(result.stdout + "\n")
        log.write("Errors:\n")
        log.write(result.stderr + "\n")
        log.write("-" * 50 + "\n\n")

        # Normalize case for easier checking
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()

        # Define pass condition: no "error" or "warning" in either output
        has_error_or_warning = (
            "error" in stdout_lower or
            "error" in stderr_lower or
            "warning" in stdout_lower or
            "warning" in stderr_lower
        )

        if result.returncode == 0 and not has_error_or_warning:
            print(f"✅ {script} executed successfully with clean output.")
            status_counts['pass'] += 1
        else:
            print(f"❌ {script} has warnings or errors (check logs).")
            status_counts['fail'] += 1

    except Exception as e:
        log.write(f"Exception occurred while logging execution: {e}\n")
        print(f"❌ Exception while logging {script}: {e}")
        status_counts['fail'] += 1


def write_execute_error_to_log(log, script, status_counts):
    try:
        log.write(f"--- Running: {script} ---\n")
        log.write("⚠️ Timeout Error: Script took too long to execute.\n")
        log.write("-" * 50 + "\n\n")
        print(f"⚠️ Timeout: {script} took too long and was skipped.")
        status_counts['fail'] += 1  # Increment fail count for timeout
    except Exception as e:
        log.write(f" {e}\n")


def write_execute_summary_to_log(log, status_counts):
    try:
        # Log the summary of pass and fail counts
        log.write("\n\n====== Execution Summary ======\n")
        log.write(f"Total Scripts Passed: {status_counts['pass']}\n")
        log.write(f"Total Scripts Failed: {status_counts['fail']}\n")
    except Exception as e:
        log.write(f" {e}\n")


def open_log_save_execution_results(log_file, python_files, generate_solvers_dir, status_counts):
    try:
        with open(log_file, "w") as log:
            log.write("====== Execution Results for Generated Solvers ======\n\n")

            for script in python_files:
                script_path = os.path.join(generate_solvers_dir, script)
                print(f"🔹 Running: {script} ...")

                # Run the script and capture the output
                try:
                    _, result = execute_python_script(script_path)

                    # Write execution results to log file
                    write_execute_results_to_log(log, script, result, status_counts)

                except subprocess.TimeoutExpired:
                    write_execute_error_to_log(log, script, status_counts)

                # Log the summary of pass and fail counts
                write_execute_summary_to_log(log, status_counts)

        print(f"\n🎯 Execution completed. Results saved in: {log_file}")
    except Exception as e:
        log.write(f" {e}\n")


def call_execute_solver(generated_solvers_dir, log_file, status_counts):
    try:
        # Define the directory where generated solver scripts are stored
        GENERATED_SOLVERS_DIR = generated_solvers_dir
        # Define the log file for execution results
        LOG_FILE = log_file

        # Ensure the output directory exists
        os.makedirs(GENERATED_SOLVERS_DIR, exist_ok=True)

        # Get all Python files in the solvers directory
        python_files = [f for f in os.listdir(GENERATED_SOLVERS_DIR) if f.endswith(".py")]

        # Ensure there are solver scripts to run
        if not python_files:
            print("No Python solver scripts found in the directory.")
            exit()
        # Initialize counters for pass and fail
        # Open a log file to save execution results
        open_log_save_execution_results(LOG_FILE, python_files, GENERATED_SOLVERS_DIR, status_counts)
    except Exception as e:
        print(f" {e}\n")


def interpolate_to_match(gt, pred):
    if gt.shape == pred.shape:
        return pred
    try:
        factors = np.array(gt.shape) / np.array(pred.shape)
        pred_resized = zoom(pred, factors, order=1)
        return pred_resized
    except Exception as e:
        print(f"RuntimeError: Interpolation failed: {e}")


def compute_losses(gt, pred):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    mse = mean_squared_error(gt_flat, pred_flat)
    mae = mean_absolute_error(gt_flat, pred_flat)
    rmse = np.sqrt(mse)
    cosine_sim = cosine_similarity(gt_flat.reshape(1, -1), pred_flat.reshape(1, -1))[0][0]
    r2 = r2_score(gt_flat, pred_flat)

    # NMSE: Normalized Mean Squared Error
    nmse = mse / np.mean(gt_flat**2)

    return mse, mae, rmse, cosine_sim, r2, nmse


def print_summary(results):
    print("\n=== Summary ===")
    for fname, res in results.items():
        print(f"📄 {fname}")
        for k, v in res.items():
            print(f"   {k}: {v}")
        print("-" * 40)


def compute_errors_gt_pred(common_files, ground_truth_dir, prediction_dir, results):
    for fname in common_files:
        try:
            gt_path = os.path.join(ground_truth_dir, fname)
            pred_path = os.path.join(prediction_dir, fname)
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            if gt.ndim == 1:
                gt = gt[:, np.newaxis]
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]

            pred = interpolate_to_match(gt, pred)

            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch after interpolation: {gt.shape} vs {pred.shape}")

            mse, mae, rmse, cosine_sim, r2, nmse = compute_losses(gt, pred)

            results[fname] = {
                "MSE": f"{mse:.3e}",
                "MAE": f"{mae:.3e}",
                "RMSE": f"{rmse:.3e}",
                "CosineSimilarity": f"{cosine_sim:.3f}",
                "R2": f"{r2:.3f}",
                "NMSE": f"{nmse:.3f}"
            }

            logging.info(
                f"{fname}: MSE={mse:.3e}, MAE={mae:.3e}, RMSE={rmse:.3e}, Cosine={cosine_sim:.3f}, R2={r2:.3f}, "
                f"NMSE={nmse:.3f}")

        except Exception as e:
            results[fname] = {"Error": str(e)}
            logging.error(f"❌ {fname} failed: {str(e)}")


def get_common_files(ground_truth_dir, prediction_dir):
    files_gt = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
    files_pred = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}
    common_files = sorted(files_gt & files_pred)

    print(f"Found {len(common_files)} common files to compare.")
    logging.info(f"Found {len(common_files)} common files.")

    return common_files


def call_compare_output_mismatch(ground_truth_dir, prediction_dir, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear old logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup new log file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("====== Starting Comparison ======")

    # === Get common .npy files ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    results = {}

    compute_errors_gt_pred(common_files, ground_truth_dir, prediction_dir, results)

    # === Print Summary ===
    print_summary(results)

    print(f"\n🎯 Log saved to: {log_file}")


def get_problem_name_pred(filename):
    return filename.rsplit('_', 1)[0]


def get_problem_name_gt(filename):
    return filename.rsplit('_', 2)[0]


def call_compare_image_mismatch(save_dir_gt, save_dir_pred, save_csv_path):
    # === Configuration ===
    os.makedirs(save_csv_path, exist_ok=True)

    # === Find common files ===
    gt_files = {get_problem_name_gt(f): f for f in os.listdir(save_dir_gt) if f.endswith('.png')}
    pred_files = {get_problem_name_pred(f): f for f in os.listdir(save_dir_pred) if f.endswith('.png')}

    common_files = gt_files.keys() & pred_files.keys()

    results = []

    for filename in common_files:
        gt_path = os.path.join(save_dir_gt, gt_files[filename])
        pred_path = os.path.join(save_dir_pred, pred_files[filename])

        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        img_pred_resized = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        abs_error = np.abs(img_gt - img_pred_resized)
        mse_val = np.mean((img_gt - img_pred_resized) ** 2)
        mae_val = np.mean(abs_error)
        ssim_val = ssim(img_gt, img_pred_resized, data_range=1.0)
        psnr_val = psnr(img_gt, img_pred_resized, data_range=1.0)

        results.append({
            "filename": filename,
            "MSE": mse_val,
            "MAE": mae_val,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    csv_file = os.path.join(save_csv_path, 'image_similarity_scores.csv')
    df.to_csv(csv_file, index=False, float_format="%.3e")


def call_create_table(compare_results_log_file, save_table_path):
    # === Config ===
    log_file_path = compare_results_log_file
    output_csv_path = save_table_path

    # === Regex pattern for results line ===
    pattern = re.compile(
        r"INFO - ([\w_.\-]+): MSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"MAE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"RMSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"Cosine=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"R2=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    # === Extract Data ===
    results = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                results.append({
                    'Filename': match.group(1),
                    'MSE': float(match.group(2)),
                    'MAE': float(match.group(3)),
                    'RMSE': float(match.group(4)),
                    'Cosine Similarity': float(match.group(5)),
                    'R-squared': float(match.group(6)),
                })

    # === Create DataFrame ===
    df = pd.DataFrame(results)

    # Format all float columns to scientific notation with 3 significant digits
    for col in ['MSE', 'MAE', 'RMSE', 'Cosine Similarity', 'R-squared']:
        df[col] = df[col].apply(lambda x: f"{x:.3e}")

    # === Save to CSV ===
    df.to_csv(output_csv_path, index=False)

    # === Display all rows/columns ===
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("✅ Table saved to:", output_csv_path)
        print(df)


def plot_1d(gt, pred, file_name, save_dir):
    x_gt = np.arange(len(gt))
    x_pred = np.arange(len(pred))

    # Get shared y-axis range
    ymin = min(gt.min(), pred.min())
    ymax = max(gt.max(), pred.max())

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"{file_name}")
    plt.subplot(2, 1, 1)
    plt.plot(x_gt, gt, label='Ground Truth', color='blue')
    plt.ylim([ymin, ymax])
    plt.title('Ground Truth')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_pred, pred, label='Prediction', color='green')
    plt.ylim([ymin, ymax])
    plt.title('Prediction')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


def plot_2d(gt, pred, file_name, save_dir):
    # Get shared colorbar range
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.suptitle(f"{file_name}")
    im0 = axes[0].imshow(gt, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


def plot_1d_diff(gt, pred, file_name, save_dir_gt, save_dir_pred):
    x_gt = np.arange(len(gt))
    x_pred = np.arange(len(pred))

    # Get shared y-axis range
    ymin = min(gt.min(), pred.min())
    ymax = max(gt.max(), pred.max())

    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_gt, gt, label='Ground Truth', color='blue')
    plt.ylim([ymin, ymax])
    plt.title(f'Ground Truth - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_pred, pred, label='Prediction', color='green')
    plt.ylim([ymin, ymax])
    plt.title(f'Prediction - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


def plot_2d_diff(gt, pred, file_name, save_dir_gt, save_dir_pred):
    # Get shared colorbar range
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    im0 = plt.imshow(gt, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Ground Truth - {file_name}')
    plt.colorbar(im0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    im1 = plt.imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Prediction - {file_name}')
    plt.colorbar(im1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


def call_save_image_same_dir(save_dir, ground_truth_dir, prediction_dir):
    # === Configuration ===
    os.makedirs(save_dir, exist_ok=True)
    # === Common files (files that exist in both directories) ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    # === Iterate and Plot for Common Files ===
    for file in common_files:
        gt_path = os.path.join(ground_truth_dir, file)
        pred_path = os.path.join(prediction_dir, file)

        try:
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            # Plot
            if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
                plot_1d(gt.flatten(), pred.flatten(), file.replace(".npy", ""), save_dir)
            elif gt.ndim == 2:
                plot_2d(gt, pred, file.replace(".npy", ""), save_dir)
            else:
                print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
        except Exception as e:
            print(f"❌ Error plotting {file}: {str(e)}")

    print(f"\n🎯 Plotting complete. Images saved to: {save_dir}")


def call_save_image_different_dir(ground_truth_dir, prediction_dir, save_dir_gt, save_dir_pred):
    # === Configuration ===
    os.makedirs(save_dir_gt, exist_ok=True)
    os.makedirs(save_dir_pred, exist_ok=True)

    # === Common files (files that exist in both directories) ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    # === Iterate and Plot for Common Files ===
    for file in common_files:
        gt_path = os.path.join(ground_truth_dir, file)
        pred_path = os.path.join(prediction_dir, file)

        try:
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            # Plot
            if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
                plot_1d_diff(gt.flatten(), pred.flatten(), file.replace(".npy", ""), save_dir_gt, save_dir_pred)
            elif gt.ndim == 2:
                plot_2d_diff(gt, pred, file.replace(".npy", ""), save_dir_gt, save_dir_pred)
            else:
                print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
        except Exception as e:
            print(f"❌ Error plotting {file}: {str(e)}")

    print(f"\n🎯 Plotting complete. Images saved to: {save_dir_gt} and {save_dir_pred}")


# ======================================================================================================
# Evaluation (v2): execution, NMSE against the reference, and a convergence test.
#
# Each generated script is run in its own empty working directory and must write one `<variable>.npy`
# per entry of `save_values` (see prompt/PDE_TASK_QUESTION_ONLY.json). Missing or misnamed variables
# count as failures instead of being skipped. For the convergence test the integer resolution
# variables named in `resolution_variables` are multiplied by the listed factors and the script is run
# again; it passes when the refined run executes and still matches the reference (NMSE < 0.1).
# ======================================================================================================
import sys

NMSE_PASS = 0.1          # NMSE below this scores 1
NMSE_PARTIAL = 0.3       # NMSE below this scores 0.5 (same thresholds as FoamBench)


def load_tasks(root_dir):
    with open(os.path.join(root_dir, "prompt", "PDE_TASK_QUESTION_ONLY.json")) as f:
        tasks = json.load(f)
    for t in tasks.values():
        sv = t.get("save_values", t.get("save values", []))
        t["save_values"] = [sv] if isinstance(sv, str) else list(sv)
        t.setdefault("resolution_variables", {})
        t.setdefault("gauge_free", [])
    return tasks


def nmse_to_score(value):
    if value < NMSE_PASS:
        return 1.0
    if value < NMSE_PARTIAL:
        return 0.5
    return 0.0


def run_in_dir(code, workdir, script_name, timeout):
    """Write `code` into an empty `workdir`, run it there, and report what happened."""
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    path = os.path.join(workdir, script_name)
    with open(path, "w") as f:
        f.write(code)
    start = datetime.now()
    try:
        r = subprocess.run([sys.executable, script_name], cwd=workdir, capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, "MPLBACKEND": "Agg"})
        status = {"exit_code": r.returncode, "timed_out": False, "stderr_tail": r.stderr.strip()[-500:]}
    except subprocess.TimeoutExpired:
        status = {"exit_code": None, "timed_out": True, "stderr_tail": f"timeout after {timeout} s"}
    status["seconds"] = round((datetime.now() - start).total_seconds(), 2)
    status["executed"] = status["exit_code"] == 0
    return status


def collect_outputs(workdir, save_values):
    """Find `<var>.npy` for each expected variable (top level first, then any sub-folder)."""
    found = {}
    for var in save_values:
        hits = [os.path.join(workdir, f"{var}.npy")]
        hits += [os.path.join(r, f"{var}.npy") for r, _, files in os.walk(workdir) if f"{var}.npy" in files]
        path = next((h for h in hits if os.path.isfile(h)), None)
        try:
            found[var] = np.load(path, allow_pickle=False) if path else None
        except Exception:
            found[var] = None
    return found


def compare_variable(gt, pred, gauge_free=False):
    """Return (nmse, note). NMSE = sum((gt - pred)^2) / sum(gt^2), after aligning pred to the gt grid."""
    if pred is None:
        return np.inf, "missing"
    note = ""
    gt = np.asarray(gt, dtype=float).squeeze()
    pred = np.asarray(pred)
    if np.iscomplexobj(pred):
        pred = pred.real
    pred = pred.astype(float).squeeze()
    if not np.all(np.isfinite(pred)):
        return np.inf, "non-finite values"
    if gt.ndim == 0:
        if pred.size != 1:
            return np.inf, f"expected a scalar, got shape {pred.shape}"
        pred = pred.reshape(())
    elif pred.ndim != gt.ndim:
        return np.inf, f"expected {gt.ndim}D, got shape {pred.shape}"
    elif pred.shape != gt.shape:
        # both are uniform grids that include the end points; zoom maps end point to end point
        pred = zoom(pred, np.array(gt.shape) / np.array(pred.shape), order=1)
        note = "interpolated"
    if gauge_free:
        gt, pred = gt - gt.mean(), pred - pred.mean()
    denom = float(np.sum(gt ** 2))
    if denom == 0.0:
        return np.inf, "reference is identically zero"
    return float(np.sum((gt - pred) ** 2)) / denom, note


def score_outputs(gt_dir, task_name, meta, outputs):
    per_var = {}
    for var in meta["save_values"]:
        gt = np.load(os.path.join(gt_dir, f"{var}_{task_name}.npy"))
        per_var[var] = compare_variable(gt, outputs.get(var), var in meta["gauge_free"])
    values = [v for v, _ in per_var.values()]
    task_nmse = float(np.mean(values)) if values else np.inf
    return per_var, task_nmse


def refine_resolution(code, factors):
    """Multiply top-level integer assignments such as `nx = 101` by the given factors."""
    changed = {}
    for name, factor in factors.items():
        pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(\d+)(\s*(#.*)?)$", re.M)
        m = pattern.search(code)
        if m:
            new_value = int(m.group(2)) * factor
            code = code[:m.start()] + f"{m.group(1)}{new_value}{m.group(3)}" + code[m.end():]
            changed[name] = (int(m.group(2)), new_value)
    return code, changed


class Evaluator:
    """Execution, NMSE and convergence for every task generated by one model."""

    def __init__(self, llm_model, prompt_json, timeout=600, refined_timeout=3600, root_dir=None, parallel=1,
                 resume=True):
        self.parallel = max(1, int(parallel))  # tasks evaluated at the same time (each in its own run folder)
        self.resume = resume  # keep tasks already scored in eval/<model>/<prompt>/task_rows.jsonl
        self.llm_model = llm_model
        self.prompt_name = os.path.splitext(prompt_json)[0]
        self.root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
        self.tasks = load_tasks(self.root_dir)
        self.solver_dir = os.path.join(self.root_dir, "solver", llm_model, self.prompt_name)
        self.gt_dir = os.path.join(self.root_dir, "results", "solution")
        self.run_dir = os.path.join(self.root_dir, "runs", llm_model, self.prompt_name)
        self.out_dir = os.path.join(self.root_dir, "eval", llm_model, self.prompt_name)
        self.timeout, self.refined_timeout = timeout, refined_timeout

    def evaluate_task(self, task_name, meta):
        row = {"task": task_name}
        script = os.path.join(self.solver_dir, f"{task_name}.py")
        if not os.path.isfile(script):
            row.update(executed=False, note="no generated script", nmse=np.inf, nmse_score=0.0,
                       converged=False, refined_nmse=np.inf)
            return row
        code = open(script).read()

        base_dir = os.path.join(self.run_dir, task_name, "base")
        status = run_in_dir(code, base_dir, f"{task_name}.py", self.timeout)
        outputs = collect_outputs(base_dir, meta["save_values"]) if status["executed"] else {}
        per_var, task_nmse = score_outputs(self.gt_dir, task_name, meta, outputs)
        row.update(executed=status["executed"], seconds=status["seconds"], timed_out=status["timed_out"],
                   nmse=task_nmse, nmse_score=nmse_to_score(task_nmse))
        for var, (value, note) in per_var.items():
            row[f"nmse_{var}"] = value
            if note:
                row[f"note_{var}"] = note
        if not status["executed"]:
            row["error"] = status["stderr_tail"]

        refined_code, changed = refine_resolution(code, meta["resolution_variables"])
        row["refined_vars"] = ", ".join(f"{k} {a}->{b}" for k, (a, b) in changed.items())
        if not changed:
            row.update(converged=False, refined_nmse=np.inf, refined_note="no resolution variables found")
            return row
        ref_dir = os.path.join(self.run_dir, task_name, "refined")
        ref_status = run_in_dir(refined_code, ref_dir, f"{task_name}.py", self.refined_timeout)
        ref_outputs = collect_outputs(ref_dir, meta["save_values"]) if ref_status["executed"] else {}
        _, ref_nmse = score_outputs(self.gt_dir, task_name, meta, ref_outputs)
        row.update(refined_executed=ref_status["executed"], refined_seconds=ref_status["seconds"],
                   refined_nmse=ref_nmse, converged=bool(ref_status["executed"] and ref_nmse < NMSE_PASS))
        return row

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        # each task's row is appended to task_rows.jsonl when it is done, so an interrupted evaluation resumes
        rows_path = os.path.join(self.out_dir, "task_rows.jsonl")
        done = {}
        if self.resume and os.path.isfile(rows_path):
            for line in open(rows_path):
                if line.strip():
                    row = json.loads(line)
                    done[row["task"]] = row
        elif os.path.isfile(rows_path):
            os.remove(rows_path)
        todo = [(t, m) for t, m in self.tasks.items() if t not in done]
        lock = threading.Lock()

        def one(item):
            task_name, meta = item
            print(f"🔹 [{self.llm_model}] {task_name}", flush=True)
            row = self.evaluate_task(task_name, meta)
            with lock:
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
                done[task_name] = row

        with ThreadPoolExecutor(self.parallel) as pool:
            list(pool.map(one, todo))
        df = pd.DataFrame([done[t] for t in self.tasks if t in done])
        # Success: executes, NMSE score 1 and convergent
        df["success"] = df["executed"].astype(bool) & (df["nmse_score"] == 1.0) & df["converged"].astype(bool)
        df.to_csv(os.path.join(self.out_dir, "task_results.csv"), index=False)
        finite = df["nmse"].replace(np.inf, np.nan)
        summary = {
            "model": self.llm_model, "prompt": self.prompt_name, "tasks": len(df),
            "execution_rate": df["executed"].mean(),
            "nmse_score": df["nmse_score"].mean(),
            "median_nmse": finite.median(),
            "convergence_rate": df["converged"].mean(),
            "success_rate": df["success"].mean(),
        }
        pd.DataFrame([summary]).to_csv(os.path.join(self.out_dir, "summary.csv"), index=False)
        print(pd.DataFrame([summary]).to_string(index=False))
        return df, summary
