## Installation

```
conda create -n deftx python=3.10
conda activate deftx
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Download and install DeFT-X:
```
git clone https://github.com/csalt-research/DeFT-X.git
cd DeFT-X
pip install -e .
```

## Training SFTs
`DeFTArguments` that may be useful:
* `rank_variant` - choice of rank variant to use: `uniform` or `variance`.
* `rank` - controls the rank value when `rank_variant = uniform`.
* `variance_threshold` - variance threshold to decides the rank value when `rank_variant = variance`.
* `keep_raio` - controls the percentage of parameters to be kept while pruning the high-frequency components.
* `ft_params_num`/`ft_params_proportion` - controls the number/proportion of the maskable params that will be fine-tuned.
* `full_ft_max_steps_per_iteration`/`full_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the first phase of LT-SFT. Both can be set.
* `sparse_ft_max_steps_per_iteration`/`sparse_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the second phase of LT-SFT. Both can be set.
* `full_ft_min_steps_per_iteration`/`sparse_ft_min_steps_per_iteration` - controls the minimum number of steps in the first/second phase of LT-SFT. Takes effect if a max number of epochs is set which amounts to a lesser number of steps.

## Example Scripts
Examples of DeFT-X training and evaluation are provided in [`examples/`](examples).

## Citation
