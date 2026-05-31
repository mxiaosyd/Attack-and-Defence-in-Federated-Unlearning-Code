# Attack Reproduction

This directory contains four independent attack-paper reproductions. The attack
code is not connected to the seven defense-method reproductions or to the
comprehensive defense benchmark.

## Structure

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

`common_attack.py` contains only shared data, model, gradient, metric, and FL
utilities. Paper-specific parameters are kept inside each paper folder's
`config.py`.

## Commands

```powershell
python attack_reproduction/gat_la_reconstruction/run.py
python attack_reproduction/fuia_model_inversion/run.py
python attack_reproduction/ulia_label_inference/run.py
python attack_reproduction/camouflaged_poisoning/run.py
```

Outputs are written under `outputs/attack_reproduction/<paper_folder>/`.

