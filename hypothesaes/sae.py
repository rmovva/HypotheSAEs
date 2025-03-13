"""Sparse autoencoder implementation and training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm.auto import tqdm
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseAutoencoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        aux_k: Optional[int] = None, # Number of neurons to consider for dead neuron revival
        multi_k: Optional[int] = None, # Number of neurons for secondary reconstruction
        dead_neuron_threshold_steps: int = 256 # Number of non-firing steps after which a neuron is considered dead
    ):
        super().__init__()
        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons
        self.aux_k = 2 * k_active_neurons if aux_k is None else aux_k
        self.multi_k = 4 * k_active_neurons if multi_k is None else multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        # Core layers
        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)
        
        # Biases as separate parameters
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))
        
        # Tracking dead neurons
        self.steps_since_activation = torch.zeros(m_total_neurons, dtype=torch.long, device=device)

        # Put model on correct device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Center input
        x = x - self.input_bias
        pre_act = self.encoder(x) + self.neuron_bias
        
        # Main top-k activation
        topk_values, topk_indices = torch.topk(pre_act, k=self.k_active_neurons, dim=-1)
        topk_values = F.relu(topk_values)
        
        # Multi-k activation for loss computation
        multik_values, multik_indices = torch.topk(pre_act, k=self.multi_k, dim=-1)
        multik_values = F.relu(multik_values)
        
        # Create sparse activation tensors
        activations = torch.zeros_like(pre_act)
        activations.scatter_(-1, topk_indices, topk_values)
        
        multik_activations = torch.zeros_like(pre_act)
        multik_activations.scatter_(-1, multik_indices, multik_values)
        
        # Update dead neuron tracking
        self.steps_since_activation += 1
        self.steps_since_activation.scatter_(0, topk_indices.unique(), 0)
        
        # Compute reconstructions
        reconstruction = self.decoder(activations) + self.input_bias
        multik_reconstruction = self.decoder(multik_activations) + self.input_bias
        
        # Handle auxiliary dead neuron revival
        aux_values, aux_indices = None, None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_neuron_pre_act = pre_act * dead_mask
            aux_values, aux_indices = torch.topk(dead_neuron_pre_act, k=self.aux_k, dim=-1)
            aux_values = F.relu(aux_values)
            
        return reconstruction, {
            "activations": activations,
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_reconstruction": multik_reconstruction,
            "aux_indices": aux_indices,
            "aux_values": aux_values,
        }

    def reconstruct_input_from_latents(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the sparse latent representation.
        Inputs:
            indices: Tensor of indices of active neurons
            values: Tensor of values of active neurons
        Returns:
            Reconstructed input tensor
        """
        activations = torch.zeros(self.m_total_neurons, device=indices.device)
        activations.scatter_(-1, indices, values)
        return self.decoder(activations) + self.input_bias

    def normalize_decoder_(self):
        """Normalize decoder weights to unit norm in-place."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def adjust_decoder_gradient_(self):
        """Adjust decoder gradient to maintain unit norm constraint."""
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = torch.sum(self.decoder.weight * self.decoder.weight.grad, dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor):
        """Initialize parameters from data statistics."""
        # Initialize bias to median of data -- See O'Neill et al. (2024) Appendix A.1
        self.input_bias.data = torch.median(data_sample, dim=0).values 
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)

    def save(self, save_path: str):
        """Save model state dict and config to specified directory."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config = {
            'input_dim': self.input_dim,
            'm_total_neurons': self.m_total_neurons,
            'k_active_neurons': self.k_active_neurons,
            'aux_k': self.aux_k,
            'multi_k': self.multi_k,
            'dead_neuron_threshold_steps': self.dead_neuron_threshold_steps
        }
        
        torch.save({
            'config': config,
            'state_dict': self.state_dict()
        }, save_path, pickle_module=pickle)
        print(f"Saved model to {save_path}")

        return save_path

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float = 1/32,
        multi_coef: float = 0.0
    ) -> torch.Tensor:
        """Compute SAE loss with auxiliary terms."""
        def normalized_mse(pred, target):
            mse = F.mse_loss(pred, target)
            baseline_mse = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
            return mse / baseline_mse
        
        recon_loss = normalized_mse(recon, x)
        recon_loss += multi_coef * normalized_mse(info["multik_reconstruction"], x)
        
        if self.aux_k is not None:
            error = x - recon.detach()
            aux_activations = torch.zeros_like(info["activations"])
            aux_activations.scatter_(-1, info["aux_indices"], info["aux_values"])
            error_reconstruction = self.decoder(aux_activations)
            aux_loss = normalized_mse(error_reconstruction, error)
            total_loss = recon_loss + aux_coef * aux_loss
        else:
            total_loss = recon_loss
            
        return total_loss

    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1/32,
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
            self.save(os.path.join(save_dir, f'SAE_M={self.m_total_neurons}_K={self.k_active_neurons}.pt'))
            
        return history

    def get_activations(self, inputs):
        """Get sparse activations for input data."""
        self.eval()
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        
        inputs = inputs.to(device)
        
        with torch.no_grad():
            _, info = self(inputs)
            activations = info['activations']
        
        return activations.cpu().numpy()

def load_model(path: str) -> SparseAutoencoder:
    """Load a saved model from path."""
    checkpoint = torch.load(path, pickle_module=pickle)
    model = SparseAutoencoder(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from {path}")
    return model

def get_multiple_sae_activations(sae_list, X, return_neuron_source_info=False):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float().to(device)
    activations_list = []
    neuron_source_sae_info = []
    for s in sae_list:
        activations_list.append(s.get_activations(X))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    if return_neuron_source_info:
        return activations, neuron_source_sae_info
    else:
        return activations
