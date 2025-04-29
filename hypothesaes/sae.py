"""Top-K sparse autoencoder implementation and training.

Implementation details:
1. Matryoshka loss: average the loss computed with increasing prefixes of neurons.
2. Auxiliary-K reconstruction: revive dead neurons.
3. Multi-K reconstruction: use a slightly less sparse reconstruction with lower weight.

Our implementation draws from:
- Bussmann et al. (2025) https://github.com/bartbussmann/matryoshka_sae/blob/main/sae.py (for Matryoshka loss)
- O'Neill et al. (2024) https://github.com/Christine8888/saerch/blob/main/saerch/topk_sae.py (for top-K with dead neuron revival)
"""

import os
import pickle
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Sparse Autoencoder with optional Matryoshka loss
# ----------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: Optional[int] = None,
        multi_k: Optional[int] = None,
        dead_neuron_threshold_steps: int = 256,
        prefix_lengths: Optional[List[int]] = None,
    ) -> None:
        """Create a top-K sparse autoencoder.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input representations.
        m_total_neurons : int
            Total number of neurons (SAE features).
        k_active_neurons : int
            Number of active neurons selected per example.
        aux_k : int | None, optional
            Upper bound on the number of dead neurons to try for auxiliary residual prediction.
            The default value, set during initialization, is `2 * k_active_neurons`.
            The default coefficient on the auxiliary loss is 1/32 (see `compute_loss`).
        multi_k : int | None, optional
            How many neurons to use for the less‑sparse multi‑K reconstruction
            term.  The default weight on this loss is 0, and the default value
            is None (i.e. no multi‑K reconstruction).  If using, a recommend starting 
            value is `4 * k_active_neurons`.
        dead_neuron_threshold_steps : int, optional
            Steps of non‑activation after which a neuron counts as *dead*.
        prefix_lengths : list[int] | None, optional
            If given (e.g. `[16, 64]`), activates *Matryoshka* loss: the first
            prefix has 16 neurons, the second 64, etc.  If *None*, all
            M neurons are treated equally.
        """

        super().__init__()
        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons

        # Fallback defaults ---------------------------------------------------
        self.aux_k = (
            min(2 * k_active_neurons, m_total_neurons) if aux_k is None else aux_k
        )
        self.multi_k = multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        # Matryoshka prefixes as full lengths --------------------------------
        self.prefix_lengths = prefix_lengths
        if self.prefix_lengths is not None:
            assert (
                self.prefix_lengths[-1] == m_total_neurons
            ), "Last prefix length must equal m_total_neurons"
            assert all(
                x > y for x, y in zip(self.prefix_lengths[1:], self.prefix_lengths[:-1])
            ), "Each prefix length must be greater than the previous one"

        # weight initialization --------------------------------------------------------------
        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        # dead‑neuron bookkeeping --------------------------------------------
        self.steps_since_activation = torch.zeros(
            m_total_neurons, dtype=torch.long, device=device
        )

        self.to(device)

    # ---------------------------------------------------------------------
    # Forward pass (agnostic to Matryoshka configuration)
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # W_enc(x - b_pre) + b_enc
        x = x - self.input_bias
        pre_act = self.encoder(x) + self.neuron_bias

        # main Top‑K ---------------------------------------------------------
        topk_vals, topk_idx = torch.topk(pre_act, self.k_active_neurons, dim=-1)
        topk_vals = F.relu(topk_vals)
        activ = torch.zeros_like(pre_act)
        activ.scatter_(-1, topk_idx, topk_vals)

        # multi‑K --------------------------------------------------
        if self.multi_k is not None:
            multik_vals, multik_idx = torch.topk(pre_act, self.multi_k, dim=-1)
            multik_vals = F.relu(multik_vals)
            multik_activ = torch.zeros_like(pre_act)
            multik_activ.scatter_(-1, multik_idx, multik_vals)
            multik_recon = self.decoder(multik_activ) + self.input_bias
        else:
            multik_recon = None

        # dead‑neuron tracking
        self.steps_since_activation += 1
        self.steps_since_activation.scatter_(0, topk_idx.unique(), 0)

        # reconstructions ----------------------------------------------------
        recon = self.decoder(activ) + self.input_bias

        # aux‑K --------------------------------------------------------------
        aux_idx = aux_vals = None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_pre_act = pre_act * dead_mask
            aux_vals, aux_idx = torch.topk(dead_pre_act, self.aux_k, dim=-1)
            aux_vals = F.relu(aux_vals)

        info = {
            "activations": activ,  # needed for Matryoshka slices
            "topk_indices": topk_idx,
            "topk_values": topk_vals,
            "multik_reconstruction": multik_recon,
            "aux_indices": aux_idx,
            "aux_values": aux_vals,
        }
        return recon, info

    # ------------------------------------------------------------------
    # Loss with optional Matryoshka terms
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        baseline_mse = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
        return mse / baseline_mse
    
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float,
        multi_coef: float,
    ) -> torch.Tensor:
        """Return total loss (Matryoshka L2 + optional multi‑K + aux).

        If `len(prefix_lengths)==1` there is no Matryoshka nesting.
        Otherwise we average the L2 of every prefix reconstruction as in
        Bussmann et al. (2025).

        multiK / auxK implemented as in O'Neill et al. (2024).
        """

        activ = info["activations"]
        # main L2 -----------------------------------------------------------
        if self.prefix_lengths is None or len(self.prefix_lengths) == 1:
            main_l2 = self._normalized_mse(recon, x)
        else:
            l2_terms = []
            dec_weight = self.decoder.weight  # (input_dim, m_total_neurons)
            for end in self.prefix_lengths:
                # activ[:, :end] is (batchsize, end);  dec_weight[:, :end] is (input_dim, end)
                prefix_recon = activ[:, :end] @ dec_weight[:, :end].t() + self.input_bias
                l2_terms.append(self._normalized_mse(prefix_recon, x))
            main_l2 = torch.stack(l2_terms).mean()

        # multi‑K term ------------------------------------------------------
        if multi_coef != 0 and info["multik_reconstruction"] is not None:
            main_l2 = main_l2 + multi_coef * self._normalized_mse(
                info["multik_reconstruction"], x
            )

        # aux‑K term --------------------------------------------------------
        if self.aux_k is not None and info["aux_indices"] is not None:
            err = x - recon.detach()
            aux_act = torch.zeros_like(activ)
            aux_act.scatter_(-1, info["aux_indices"], info["aux_values"])
            err_recon = self.decoder(aux_act)
            aux_loss = self._normalized_mse(err_recon, err)
            return main_l2 + aux_coef * aux_loss
        else:
            return main_l2

    # ------------------------------------------------------------------
    # Utility helpers for training
    # ------------------------------------------------------------------
    def normalize_decoder_(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def adjust_decoder_gradient_(self):
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = (self.decoder.weight * self.decoder.weight.grad).sum(dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor):
        self.input_bias.data = torch.median(data_sample, dim=0).values
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)
        
    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        config = {
            "input_dim": self.input_dim,
            "m_total_neurons": self.m_total_neurons,
            "k_active_neurons": self.k_active_neurons,
            "aux_k": self.aux_k,
            "multi_k": self.multi_k,
            "dead_neuron_threshold_steps": self.dead_neuron_threshold_steps,
            "prefix_lengths": self.prefix_lengths,
        }
        torch.save({"config": config, "state_dict": self.state_dict()}, save_path, pickle_module=pickle)
        print(f"Saved model to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float = 1.0
    ) -> Dict:
        """Train the sparse autoencoder on input data."""
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size) if X_val is not None else None
        
        # Initialize from batch of data
        self.initialize_weights_(X_train.to(device))
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop setup
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'dead_neuron_ratio': []}
        
        # Training loop
        iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
        for epoch in iterator:
            self.train()
            train_losses = []
            
            for batch_x, in train_loader:
                batch_x = batch_x.to(device)
                recon, info = self(batch_x)
                loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                
                optimizer.zero_grad()
                loss.backward()
                self.adjust_decoder_gradient_()
                
                # Apply gradient clipping
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                
                optimizer.step()
                self.normalize_decoder_()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Track dead neurons
            dead_ratio = (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
            history['dead_neuron_ratio'].append(dead_ratio)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, in val_loader:
                        batch_x = batch_x.to(device)
                        recon, info = self(batch_x)
                        val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Update progress bar
            if show_progress:
                iterator.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}' if val_loader else 'N/A',
                    'dead_ratio': f'{dead_ratio:.3f}'
                })
        
        # Save final model
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = get_sae_checkpoint_name(self.m_total_neurons, self.k_active_neurons, self.prefix_lengths)
            self.save(os.path.join(save_dir, filename))
            
        return history
    
    # ------------------------------------------------------------------
    # Compute activations with batched SAE inference
    # ------------------------------------------------------------------
    def get_activations(self, inputs, batch_size=16384, show_progress=True):
        """Get sparse activations for input data with batching to prevent CUDA OOM.
        
        Args:
            inputs: Input data as numpy array or torch tensor
            batch_size: Number of samples per batch (default: 16384)
        
        Returns:
            Numpy array of activations
        """
        self.eval()

        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float)
        elif isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a list, numpy array, or torch tensor")
        if not inputs.dtype == torch.float:
            inputs = inputs.float()
        
        num_samples = inputs.shape[0]
        all_activations = []
        with torch.no_grad():
            if show_progress:
                iterator = tqdm(range(0, num_samples, batch_size), desc=f"Computing activations (batchsize={batch_size})")
            else:
                iterator = range(0, num_samples, batch_size)
                
            for i in iterator:
                batch = inputs[i:i+batch_size]
                batch = batch.to(device)
                _, info = self(batch)
                batch_activations = info['activations']
                all_activations.append(batch_activations.cpu())
        
        return torch.cat(all_activations, dim=0).numpy()

# -----------------------------------------------------------------------------
# Additional utils
# -----------------------------------------------------------------------------
def get_sae_checkpoint_name(m_total_neurons, k_active_neurons, prefix_lengths=None):
    if prefix_lengths is None:
        return f'SAE_M={m_total_neurons}_K={k_active_neurons}.pt'
    else:
        prefix_str = "-".join(str(g) for g in prefix_lengths)
        return f'SAE_matryoshka_M={m_total_neurons}_K={k_active_neurons}_prefixes={prefix_str}.pt'

def load_model(path: str) -> SparseAutoencoder:
    ckpt = torch.load(path, pickle_module=pickle)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from {path} onto device {device}")
    return model

def get_multiple_sae_activations(sae_list, X, return_neuron_source_info=False, **kwargs):
    if not isinstance(sae_list, list):
        sae_list = [sae_list]
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float)

    activations_list = []
    neuron_source_sae_info = []
    for s in sae_list:
        activations_list.append(s.get_activations(X, **kwargs))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    
    if return_neuron_source_info:
        return activations, neuron_source_sae_info
    else:
        return activations
