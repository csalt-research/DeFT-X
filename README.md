## Installation

```
conda create -n deftx python=3.10.14
conda activate deftx
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

Download and install DeFT-X:
```
git clone https://github.com/sonaelzasimon/DeFT-X_fork.git
cd DeFT-X
pip install -e .
```

## Training SFTs

`DeFTX_SparseFineTuner` is function is used to modify the `transformers Trainer` class and its subclasses to perform DeFT-X Sparse Fine-Tuning.

The `DeFTX_SparseFineTuner` takes the following arguments in addition to those of `Trainer`:
* `sft_args`: an `DeFTArguments` object which holds hyperparameters relating to DeFT training (c.f. `transformers TrainingArguments`).
* `maskable_params`: a list of model parameter tensors which are eligible for DeFT sparse fine-tuning. Parameters of the classification head should be excluded from this list because these should typically be fully fine-tuned. E.g.
```
maskable_params = [
    n for n, p in model.named_parameters()
    if n.startswith(model.base_model_prefix) and p.requires_grad
]
```

The following command-line params processed by `DeFTArguments` may be useful:
* `rank_variant` - choice of rank variant to use: `uniform` or `variance`.
* `rank` - controls the rank value when `rank_variant = uniform`.
* `variance_threshold` - variance threshold to decides the rank value when `rank_variant = variance`.
* `keep_raio` - controls the percentage of parameters to be kept while pruning the high-frequency components.
* `ft_params_num`/`ft_params_proportion` - controls the number/proportion of the maskable params that will be fine-tuned.
* `full_ft_max_steps_per_iteration`/`full_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the first phase of LT-SFT. Both can be set.
* `sparse_ft_max_steps_per_iteration`/`sparse_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the second phase of LT-SFT. Both can be set.
* `full_ft_min_steps_per_iteration`/`sparse_ft_min_steps_per_iteration` - controls the minimum number of steps in the first/second phase of LT-SFT. Takes effect if a max number of epochs is set which amounts to a lesser number of steps.


## Example Scripts
Examples of SFT training and evaluation are provided in [`examples/`](examples).


## Citation
