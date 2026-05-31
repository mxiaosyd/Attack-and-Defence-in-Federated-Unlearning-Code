# FU Reproduction Benchmark

This repository contains a modular reproduction benchmark for federated
unlearning (FU) under poisoning and backdoor settings. It has three experimental
layers:

1. independent reproduction of four FU attack papers;
2. independent reproduction of seven FU defense papers;
3. unified comprehensive evaluation of the seven reproduced defenses, plus a
   separate targeted validation for two copied improvement variants.

The original defense implementations, independent reproduction entries, attack
reproduction entries, and improvement variants are intentionally separated. This
keeps paper-level reproduction code stable while allowing broader evaluation and
targeted engineering validation.

## Directory Structure

```text
fu_repro_benchmark/
  common/                    shared FL, attack, metric, runner, and output code
  methods/                   original seven defense-method implementations
  improved_methods/          copied variants: FedSweep+ and FAST+
  independent_reproduction/  seven independent defense reproduction entries
  attack_reproduction/       four independent attack reproduction folders
  experiments/               unified comprehensive and improvement experiments
  data/                      downloaded datasets
  outputs/                   generated results and analysis artifacts
  tools/                     local helper scripts used for analysis/report work
```

## Main Methods

The seven reproduced defense methods are:

| Method | Role in this repository |
|---|---|
| `FedRecover` | Historical-information recovery from poisoning attacks. |
| `Crab` | Selective rollback and recovery with historical round/client information. |
| `FedUP` | Pruning-based FU for model poisoning attacks. |
| `FedSweep` | Backdoor unlearning with trigger inversion and suspicious-client filtering. |
| `UnlearningBackdoor` | Historical attacker-update subtraction with clean knowledge distillation. |
| `FAST` | Server-side malicious update subtraction and benchmark repair. |
| `MCC-Fed` | Malicious-client and contribution-aware FU with regularized unlearning. |

The two copied improvement variants are:

| Method | Purpose |
|---|---|
| `FedSweep+` | Adds historical-risk evidence and generated-trigger repair to reduce residual ASR. |
| `FAST+` | Adds localized trigger-invariance repair after FAST-style subtraction. |

The improvement variants live only under `improved_methods/`. They do not modify
the original `methods/` or `independent_reproduction/` code.

## Environment

Install the required Python packages in any Python environment that can run the
project:

```powershell
pip install -r requirements.txt
```

The current minimal requirements are:

```text
numpy
torch
torchvision
```

Activating `.venv` is optional if your default `python` already points to an
environment with these packages installed:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Dataset Preparation

The repository does not include downloaded datasets. The `data/` directory is
ignored by Git because dataset archives and extracted files are large. Each user
should prepare the datasets locally before running experiments.

The current experiments use the following TorchVision datasets:

| Dataset | Used by |
|---|---|
| `MNIST` | independent attack reproductions, independent defense reproductions, and comprehensive experiments |
| `FashionMNIST` | comprehensive experiments |
| `CIFAR-10` | comprehensive experiments |

By default, experiments use:

```text
data/
```

For the defense reproductions and comprehensive experiments, automatic download
is already enabled in the default commands. PyTorch/TorchVision will download the
required datasets into `data/` on the first run if the machine has internet
access.

For the independent attack reproductions, the default configuration keeps
`download = False` in each attack folder's `config.py`. Either place the required
dataset under `data/` before running the attack, or set `download = True` in the
corresponding `config.py`.

After automatic or manual preparation, the expected local layout is:

```text
data/MNIST/
data/FashionMNIST/
data/cifar-10-batches-py/
```

For manual CIFAR-10 preparation, extract `cifar-10-python.tar.gz` so that
`data/cifar-10-batches-py/` exists. Do not commit dataset archives such as
`cifar-10-python.tar.gz` or extracted dataset files to GitHub.

## Independent Attack Reproduction

Each attack paper has its own folder, parameter file, implementation, runner,
and output path.

```text
attack_reproduction/
  common_attack.py
  gat_la_reconstruction/
    config.py
    attack.py
    run.py
  fuia_model_inversion/
    config.py
    attack.py
    run.py
  ulia_label_inference/
    config.py
    attack.py
    run.py
  camouflaged_poisoning/
    config.py
    attack.py
    run.py
```

Run the four attack reproductions:

```powershell
python attack_reproduction/gat_la_reconstruction/run.py
python attack_reproduction/fuia_model_inversion/run.py
python attack_reproduction/ulia_label_inference/run.py
python attack_reproduction/camouflaged_poisoning/run.py
```

Paper-specific parameters are stored inside each attack folder's `config.py`.
Outputs are written under:

```text
outputs/attack_reproduction/<attack_folder>/
```

## Independent Defense Reproduction

The seven independent defense reproduction entries are:

```powershell
python independent_reproduction/fedrecover.py
python independent_reproduction/crab.py
python independent_reproduction/fedup.py
python independent_reproduction/fedsweep.py
python independent_reproduction/unlearning_backdoor.py
python independent_reproduction/fast.py
python independent_reproduction/mcc_fed.py
```

Shared independent-reproduction parameters are defined in:

```text
independent_reproduction/reproduction_config.py
```

The default independent setting is:

```text
dataset       = MNIST
model         = SmallCNN
train_size    = 15000
test_size     = 3000
clients       = 20 total, 4 malicious
partition     = Dirichlet non-IID, alpha=0.5
rounds        = 60 global rounds
local_epochs  = 2
batch_size    = 64
learning_rate = 0.08
attack        = square-trigger backdoor, target label 0
poison_frac   = 0.7
public_data   = held-out test split, public_fraction=0.5
```

`MCC-Fed` uses its paper-aligned independent setting:

```text
clients       = 10 total, 4 malicious
partition     = IID
poison_frac   = 0.9
```

Each independent run writes a compact summary file:

```text
outputs/independent_reproduction/<method>/result.json
```

The same folder may also contain the raw `run-YYYYMMDD-HHMMSS/` directory with
CSV, JSON, config, command, and log files.

## Unified Comprehensive Evaluation

The original comprehensive experiment evaluates only the seven reproduced
defense methods in `methods/`. The preferred entry points are:

```powershell
python experiments/original_7/data_distribution.py
python experiments/original_7/attack_robustness.py
python experiments/original_7/malicious_ratio.py
python experiments/original_7/efficiency.py
```

The scenario definitions and shared parameters are in:

```text
experiments/comprehensive_config.py
```

The four experiment groups are:

| Group | Scenarios | What changes | Main purpose |
|---|---:|---|---|
| `data_distribution` | 8 | dataset and IID/non-IID distribution | Test stability under data heterogeneity. |
| `attack_robustness` | 11 | trigger pattern, trigger position, label flip, model replacement | Test robustness to attack variation. |
| `malicious_ratio` | 8 | malicious-client fraction | Test sensitivity to adversary scale. |
| `efficiency` | 4 | representative datasets | Compare runtime together with recovery quality. |

Comprehensive result tables in the report use group-level summaries. For a group
with multiple scenarios, the reported value is the arithmetic mean of the
scenario-level metric for that method. The raw per-scenario outputs remain in
the output directory.

Comprehensive outputs are written under:

```text
outputs/comprehensive_reproduction/
  manifest.json
  <experiment>/
    summary.csv
    summary.json
    detailed_results.csv
    detailed_results.json
    manifest.json
    <scenario>/
      run-YYYYMMDD-HHMMSS/
        results.csv
        results.json
        config.json
        command.json
        comprehensive.log
```

## Improved Validation

The improvement validation evaluates only `FedSweep+` and `FAST+`. It should be
compared with the corresponding original `FedSweep` and `FAST` rows from:

```text
outputs/comprehensive_reproduction/
```

Run the improved validation entries:

```powershell
python experiments/improved_validation/data_distribution.py
python experiments/improved_validation/attack_robustness.py
python experiments/improved_validation/malicious_ratio.py
python experiments/improved_validation/efficiency.py
```

Outputs are written under:

```text
outputs/improved_validation/
```

This experiment is not a second seven-method benchmark. It is a paired
original-versus-plus validation for the two copied improvement variants.

## Metrics

Common output fields include:

| Field | Meaning |
|---|---|
| `clean_acc` | Clean test accuracy after poisoning, unlearning, or retraining. |
| `clean_loss` | Clean test cross-entropy loss. |
| `asr` | Attack success rate for the configured attack objective. |
| `runtime_sec` | Recorded recovery or method runtime. |
| `history_storage_mb` | Approximate history storage footprint when recorded by the runner. |
| `clean_acc_delta_vs_poisoned` | Clean accuracy difference from the poisoned model. |
| `asr_drop_vs_poisoned` | ASR reduction from the poisoned model. |
| `clean_acc_gap_to_retrain` | Clean accuracy gap to the clean-retraining reference. |
| `asr_gap_to_retrain` | ASR gap to the clean-retraining reference. |

Method-specific fields may include detected clients, true malicious clients,
selected rounds, rollback traces, pruning details, trigger-inversion statistics,
removed update norms, contribution scores, and recovery traces.

## Output Interpretation

The code does not decide whether a reproduction is successful. It writes
experiment outputs, metrics, configurations, commands, and logs. Reproduction
quality should be judged from the generated values and the corresponding paper
requirements.

For defense experiments:

```text
Poisoned      = model trained with malicious clients before unlearning
RetrainClean  = clean reference model trained without the removed malicious influence
Recovered     = model produced by the evaluated unlearning/recovery method
```

Low ASR alone is not sufficient if clean accuracy collapses. High clean accuracy
alone is not sufficient if residual ASR remains high. The main defense
evaluation therefore considers clean utility, attack removal, retraining gap, and
runtime together.

## Important Files

```text
common/experiment_runner.py
```

Shared runner used by independent and comprehensive defense experiments.

```text
common/context.py
common/recovery_base.py
```

Shared method interface and method context passed to defense implementations.

```text
methods/<method>/method.py
```

Original reproduced defense implementation for one paper.

```text
independent_reproduction/<method>.py
```

Direct Python entry point for one independent defense reproduction.

```text
attack_reproduction/<paper>/run.py
```

Direct Python entry point for one independent attack reproduction.

```text
experiments/original_7/*.py
```

Direct entries for the four original seven-method comprehensive experiment
groups.

```text
experiments/improved_validation/*.py
```

Direct entries for the paired `FedSweep+` and `FAST+` validation groups.

## Analysis Artifacts

Generated reports, LaTeX files, figures, Word documents, and PDFs may be stored
under:

```text
outputs/analysis/
```

These files are analysis artifacts. They are not required for running the
reproduction code.
