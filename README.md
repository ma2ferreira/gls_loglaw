# gls_loglaw

Open-source Python implementation of the generalised least squares log-law regression model, detailed in:

> Aguiar Ferreira, M. A. and Ganapathisubramani, B. (2026) *Generalised least-squares applied to the log-law velocity profile.* Submitted for review to *Journal of Fluid Mechanics.*

This work was financially supported by the Engineering and Physical Sciences Research Council (EPSRC) through grant EP/0000000/0.

---

## Installation

This guide covers setup using **Anaconda Prompt** or **Git Bash** on Windows.

### 1. Create a project folder and navigate into it

```bash
mkdir "C:\Users\YourName\Projects"
cd "C:\Users\YourName\Projects"
```

### 2. Clone the repository

```bash
git clone https://github.com/ma2ferreira/gls_loglaw.git
cd gls_loglaw
```

### 3. Create and activate a Conda environment (Python ≥ 3.11)

```bash
conda create --name gls_env python=3.11
conda activate gls_env
```

**Using Spyder?** Also install compatible Spyder kernels:

```bash
conda install spyder-kernels==3.1.*
```

### 4. Install the package in editable mode

```bash
pip install -e .
```

Editable mode (`-e`) ensures any changes to the source are immediately reflected without reinstalling.

### 5. Verify the installation

```bash
python
>>> import gls_loglaw
>>> from gls_loglaw import bl_profile
>>> exit()
```

### 6. Run the example script

From the repository root:

```bash
python examples/example_script.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Confirm the `gls_env` environment is active and you are in the repo root |
| Spyder can't find the package | Set the interpreter in **Tools → Preferences → Python Interpreter** to the `gls_env` Python executable |
| `cd` fails on paths with spaces | Wrap the path in double quotes: `cd "C:\My Folder\gls_loglaw"` |