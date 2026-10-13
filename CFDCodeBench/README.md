# Code Generation Benchmark Framework
This repository provides a benchmark suite to evaluate the performance of large language models (LLMs) on
**code generation tasks for solving partial differential equations (PDEs)**. It includes prompt templates, 
auto-evaluation utilities, and solution validation for physics-based problems.

---

## 📁 Repository Structure

```
├── __pycache__
│   ├── utils.cpython-310.pyc
│   └── utils.cpython-38.pyc
├── clean.py
├── folder_structure.md
├── prompt
│   ├── PDE_TASK_QUESTION_ONLY.json
│   └── prompts.json
├── results
│   └── solution
├── solution
│   ├── 1D_Burgers_Equation.py
│   ├── 1D_Diffusion.py
│   ├── 1D_Euler_Shock_Tube.py
│   ├── 1D_KdV_Burgers_Equation.py
│   ├── 1D_Linear_Convection_adams.py
│   ├── 1D_Linear_Convection_explicit_euler.py
│   ├── 1D_Linear_Convection_pred_corr.py
│   ├── 1D_Linear_Convection_rk.py
│   ├── 1D_Nonlinear_Convection_LW.py
│   ├── 1D_Nonlinear_Convection_Lax.py
│   ├── 1D_Nonlinear_Convection_Mk.py
│   ├── 2D_Burgers_Equation.py
│   ├── 2D_Convection.py
│   ├── 2D_Diffusion.py
│   ├── 2D_Diffusion_FVM.py
│   ├── 2D_Inviscid_Burgers_FOU.py
│   ├── 2D_Inviscid_Burgers_MK.py
│   ├── 2D_Laplace_Equation.py
│   ├── 2D_Linear_Convection.py
│   ├── 2D_Navier_Stokes_Cavity.py
│   ├── 2D_Navier_Stokes_Channel.py
│   ├── 2D_Possion.py
│   ├── 2D_Rayleigh_Benard_Convection.py
│   ├── 2D_Shear_Flow_With_Tracer.py
│   ├── 2D_Steady_Heat_Equation_Gauss.py
│   ├── 2D_Steady_Heat_Equation_Jac.py
│   ├── 2D_Steady_Heat_Equation_SOR.py
│   ├── 2D_Unsteady_Heat_Equation_ADI.py
│   ├── 2D_Unsteady_Heat_Equation_DF.py
│   ├── 2D_Unsteady_Heat_Equation_SE.py
│   ├── Flow_Past_Circular_Cylinder.py
│   ├── Fully_Developed_Turbulent_Channel_Flow.py
│   ├── Lane_Emden_Equation.py
│   ├── Lid_Driven_Cavity.py
│   ├── Pipe_Flow_Disk_EVP.py
│   └── Vortex_Roll_Up.py
├── train_plain_prompts.py
├── tree.py
└── utils.py
```
## 📁 Repository Structure Explained
This repository is designed to **benchmark LLM-generated PDE solvers** against ground-truth implementations. 
Here's an overview of each folder and file:
### 📂 `prompt/`
* `PDE_TASK_QUESTION_ONLY.json`: Main prompt configuration for generation
### 📂 `solution/`
* Holds ground-truth reference implementations of various PDE solvers (e.g., finite difference/volume/time methods).
* Each file solves a specific PDE (e.g., `1D_Burgers_Equation.py`, `2D_Laplace_Equation.py`, etc.).
* Used for comparison against LLM-generated solutions.
### 📂 `results/solution/`
* Stores outputs from the ground truth for comparison (e.g., `.npy` files).
### 📄 `train_plain_prompts.py`
* Main entry point for running the benchmark.
* This script:
  * Generate prompts for LLM using `PDE_TASK_QUESTION_ONLY.json`
  * Calls the selected LLM to generate solver code
  * Executes generated code and save the `.npy`
  * Compares results with reference solutions
  * Post-processing, create and save images / tables / log files
---
## 🚀 Quickstart

### 1. 🛠️ Installation

We recommend using **conda** to manage dependencies:

```bash
conda create -n pde_benchmark python=3.10
conda activate pde_benchmark

# Install Python dependencies
pip install -r requirements.txt
```
Make sure to install extra dependencies manually if needed:
```bash
pip install openai boto3 matplotlib scikit-learn opencv-python scikit-image
```

### 2. 🔑 API Keys
Before running, make sure your ```OpenAI```, ```Gemini```, or ```Bedrock``` API credentials are set.

You can either export them in environment variables:
```bash
export OPENAI_API_KEY=your-key
export GOOGLE_API_KEY=your-key
```
Or directly edit the ```utils.py``` to insert your keys.

---

## Datasets
The latest (V2) task definitions and prompts are included in this repository: [`PDE_Benchmark/prompt/PDE_TASK_QUESTION_ONLY.json`](PDE_Benchmark/prompt/PDE_TASK_QUESTION_ONLY.json) and [`PDE_Benchmark/prompt/prompts.json`](PDE_Benchmark/prompt/prompts.json).

The datasets used in the paper can be downloaded from `https://www.kaggle.com/datasets/nithinsekhar/cfdcodebench/data`. 
From the folder named CFDCodeBench in the dataset, copy the folders solution and prompt to the PDE_Benchmark folder.

## 💻 Run the Benchmark
To run the benchmark pipeline for all prompts and models:
```bash
python train_plain_prompts.py
```
This script performs the following steps:

* **Load Prompts**  
  Loads each task prompt from the `prompt/` directory.

* **Call LLM to Generate Code**  
  Uses the selected Large Language Model (LLM) (e.g., GPT-4o, Gemini, Claude, etc.) to generate solver code for the given PDE problem.

* **Save Generated Code**  
  Saves the generated Python solver scripts into the `solver/` directory.

* **Execute and Evaluate**  
  Runs each solver, compares the results with the ground-truth reference in the `solution/` directory.

* **Log Results**  
  * Performance metrics and runtime are logged into the `report/` folder.  
  * Visual outputs are saved in the `image/` folder.  
  * Final comparison results are stored in the `results/` directory.
  * Final comparison results tables are stored in the `table/` directory.
---
## 📊 Output and Evaluation
* Execution results (error logs, pass/fail): saved in `report/`

* Generated solvers: `solver/`

* Reference solutions: `solution/`

* Performance images: `image/`

* Comparison tools: `compare/`, `compare_images/`

* Summary tables: `table/`
---

## 🧹 clean.py — Reset Environment Script
This script is designed to reset the working environment before running a new experiment.
It acts like a "make clean" command in traditional build systems.
```bash
python clean.py
```
---
## 🧪 Add Your Own Task
* Add a new `prompt_name.json` under `prompt/`

* Add the corresponding ground truth solution under `solution/`
