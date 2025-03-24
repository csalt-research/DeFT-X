import logging
import numpy as np
import torch
from tqdm import tqdm

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)
from deftx.deft import DeFT

def DeFTX_SparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _DeFTX_SparseFineTuner(_SparseFineTuner):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            logger.setLevel(self.args.get_process_log_level())
            if self.deft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.deft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.deft_args.ft_params_num

        def magnitude_pruning(self, tensor, keep_ratio = 0.05):
            k = int(keep_ratio * tensor.numel())
            abs_tensor = torch.abs(tensor)
            topk_values, _ = torch.topk(abs_tensor.flatten(), k)  
            threshold = topk_values[-1]  
            return torch.where(abs_tensor >= threshold, tensor, torch.zeros_like(tensor))
        
        def parallel_computation(self, rank_variant, rank=100, variance_threshold=0.90, keep_ratio=0.05):
            # Create CUDA streams for parallel execution
            streams = {n: torch.cuda.Stream() for n, p in self.model.named_parameters() if n in self.maskable_params and 'bias' not in n}
            results = {}
            for n, p in self.model.named_parameters():
                if n in self.maskable_params:
                    p.grad = None  # Save memory
                    W = p - self._original_params[n].to(p.device)
                    if('bias' in n):
                        results[n] = W
                    else:
                        # Use CUDA stream
                        with torch.cuda.stream(streams[n]):
                            # Compute SVD
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                            if(rank_variant == 'variance'):
                                # Determine rank for 90% variance
                                total_variance = torch.sum(S ** 2)
                                explained_variance = torch.cumsum(S ** 2, dim=0)
                                r = torch.searchsorted(explained_variance / total_variance, torch.tensor(variance_threshold, device=S.device)) + 1
                                print("Variance threshold = ", variance_threshold)
                            elif(rank_variant == 'uniform'):
                                r = min(rank, S.shape[0])
                            print(n, r)
                            # Select top-r singular values and vectors
                            U_r = U[:, :r]
                            S_r = torch.diag(S[:r])
                            Vh_r = Vh[:r, :]
                            # Compute new W
                            B = U_r @ S_r @ Vh_r
                            T = self.magnitude_pruning(W - B, keep_ratio)
                            W_ = B + T
                            results[n] = W_
            # Synchronize all streams
            torch.cuda.synchronize()
            delta = {}
            abs_delta = torch.tensor([], device=p.device)
            for n, W_ in results.items():
                delta[n] = W_
                abs_delta = torch.cat([abs_delta, torch.abs(W_).flatten()], dim=0)
            return delta, abs_delta
        
        def unfreeze_k_most_changed_params_using_svd_denoising(self, k):
            with torch.no_grad():
                delta = {}
                abs_delta = []
                delta, abs_delta = self.parallel_computation(self.deft_args.rank_variant, self.deft_args.rank, self.deft_args.variance_threshold, self.deft_args.keep_ratio)
                top_k_values, _ = torch.topk(abs_delta, k) 
                thresh = top_k_values[-1].item()
                print(f'Masking threshold = {thresh}')
                print(f'k value = {k}')
                n_masked = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        if n in delta.keys():
                            abs_delta = delta[n].abs()
                            to_mask = (abs_delta >= thresh) & (~self._mask[n])
                            self._mask[n] = to_mask | self._mask[n]
                            n_masked += to_mask.sum()
                print(f'n_masked = {n_masked}')
                           
        def train(self, **kwargs):
            self.freeze() # self._mask[n] is set to zero
            result = None
            
            for it in range(self.deft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it+1}')
                with torch.no_grad():
                    previous_params = {
                        n: torch.zeros_like(p, device='cpu').copy_(p)
                        for n, p in self.model.named_parameters()
                    }

                self.disable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.deft_args.full_ft_min_steps_per_iteration,
                    self.deft_args.full_ft_max_steps_per_iteration,
                    self.deft_args.full_ft_max_epochs_per_iteration,
                )
                super().train(**kwargs)

                self.unfreeze_k_most_changed_params_using_svd_denoising(
                    self.n_tunable_params // self.deft_args.n_ft_iterations
                )
                
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.copy_(previous_params[n])

                self.enable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.deft_args.sparse_ft_min_steps_per_iteration,
                    self.deft_args.sparse_ft_max_steps_per_iteration,
                    self.deft_args.sparse_ft_max_epochs_per_iteration,
                )
                
                result = super().train(**kwargs)
            
            return result
    return _DeFTX_SparseFineTuner
