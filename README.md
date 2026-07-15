# Invariant Tree

Code accompanying Chapter 4 ("Invariant Decision Trees") of Amit Schechter's MIT SM thesis, [*Methods for Enhancing Robustness and Generalization in Machine Learning*](https://dspace.mit.edu/entities/publication/651ae76c-49f5-4053-9983-a3dfe4841652) (MIT EECS, 2024).

This repository implements **invariant decision trees**: a tree-based learning method designed to be robust to distribution shift by encouraging splits and predictions that remain stable ("invariant") across different environments or subgroups in the training data, rather than relying on spurious, environment-specific correlations.

## Background

Standard empirical risk minimization can lead models — including tree-based ones — to exploit correlations that hold on average but break down for certain subgroups or under distribution shift (e.g., a background feature that happens to correlate with the label in training data but is not causally related to it). The thesis explores two complementary approaches to this problem:

1. A **Group DRO formulation with soft, probabilistic group assignment**, useful when subgroup/environment labels are noisy, partially observed, or unavailable.
2. **Invariant decision trees**, which adapt ideas from invariant risk minimization to tree-based models, encouraging split decisions and leaf predictions to generalize across environments instead of overfitting to environment-specific shortcuts.

This repository focuses on the second contribution. It is also related to, and benchmarks against, the tree-based baselines studied in [*Subgroup Robustness Grows on Trees: An Empirical Baseline Investigation*](https://arxiv.org/abs/2211.12703) (Gardner et al., NeurIPS 2022), whose experimental code is referenced under `subgroup-robustness-grows-on-trees/`.

## Repository structure

```
invariant_tree/
├── inv_tree_model.py   # Core implementation of the invariant decision tree model
├── tree_model.py        # Baseline / standard decision tree model used for comparison
├── inv_tree_test.py      # Tests / sanity checks for the invariant tree implementation
├── train.py              # Training entry point for fitting models
├── main.py                # Main script for running experiments end-to-end
├── datasets.py            # Dataset loading and preprocessing (incl. group/environment labels)
├── convert_data_to_csv.py  # Utility for converting raw data sources into CSV format
├── utils.py                 # Shared helper functions
├── lib/                     # Supporting library code
├── experiment_code/          # Scripts/configs for reproducing thesis experiments
└── subgroup-robustness-grows-on-trees/scripts/  # Reference scripts from the related baseline paper
```

## Installation

```bash
git clone https://github.com/amitschechter/invariant_tree.git
cd invariant_tree
pip install -r requirements.txt  # if present; otherwise install dependencies as needed
```

The codebase is written in Python and builds on standard scientific Python tooling (e.g., NumPy, pandas, scikit-learn). Check the import statements at the top of each script for the exact dependencies required in your environment.

## Usage

At a high level, the workflow is:

1. **Prepare data.** Use `datasets.py` (and `convert_data_to_csv.py` where applicable) to load and format a dataset, including any group/environment labels used to measure or enforce subgroup robustness.
2. **Train a model.** Use `train.py` or `main.py` to fit either the baseline tree model (`tree_model.py`) or the invariant tree model (`inv_tree_model.py`) on the prepared data.
3. **Evaluate.** Compare performance and subgroup robustness metrics between the standard and invariant tree models, following the experiments described in Chapter 4 of the thesis (synthetic data and the Heart Failure dataset).

Example (adjust flags/arguments to match the actual script interface):

```bash
python main.py --dataset <dataset_name> --model invariant_tree
```

Refer to the argument parser in `main.py`/`train.py` for the full list of supported options (dataset choice, model type, hyperparameters, etc.).

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{schechter2024methods,
  title  = {Methods for Enhancing Robustness and Generalization in Machine Learning},
  author = {Schechter, Amit},
  school = {Massachusetts Institute of Technology},
  year   = {2024},
  note   = {Advisor: Tommi S. Jaakkola}
}
```

You may also wish to cite the related baseline paper if you compare against tree-based methods:

```bibtex
@inproceedings{gardner2022subgroup,
  title     = {Subgroup Robustness Grows on Trees: An Empirical Baseline Investigation},
  author    = {Gardner, Josh and Popovic, Zoran and Schmidt, Ludwig},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```

## License

Released under the [MIT License](LICENSE).
