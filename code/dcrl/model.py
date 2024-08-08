import tqdm
import torch
import numpy as np
import torch.nn as nn
from norms import frobenius_norm, l21_norm, spectral_norm 
from common import get_default_device, apply_model_to_batch

# Network definition
class Net(nn.Module):
    def __init__(self, in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
        super().__init__()
        
        # Store configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.L = L

        # Store device
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device
        self.device_type = self.device.type

        # Convert model to own device
        self.to(self.device)
        
        # Create layers
        self.fc_hidden_layers = []
        for _ in range(1, self.L):
            self.fc_hidden_layers.append(
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            )
            self.fc_hidden_layers.append(
                nn.ReLU()    
            )
        self.v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(), 
            *self.fc_hidden_layers
        )
        self.U = nn.Linear(hidden_dim, out_dim)

        # Store reference matrices
        self.references = []
        for l in range(1, self.L + 1):
            self.references.append(self._get_v_layer_weights(layer=l))
        
    def _tensor_to_numpy(self, x):
        if self.device_type == 'cuda':
            return x.cpu().detach().numpy()            
        else:
            return x.detach().numpy()
        
    def _get_v_layer_linear(self, layer=1):
        return list(self.v.modules())[0][layer*2-2]
    
    def _get_v_layer_activation(self, layer=1):
        return list(self.v.modules())[0][layer*2-1]
    
    def _get_v_layer_weights(self, layer=1):
        v_layer = self._get_v_layer_linear(layer=layer)
        return self._tensor_to_numpy(v_layer.weight)
    
    def _get_output_from_layer(self, x, last_layer=1, preactivation=False):
        for l in range(1, last_layer + 1):
            # Get the preactivation of current layer
            x = self._get_v_layer_linear(layer=l)(x)

            # Get the activation function
            activation = self._get_v_layer_activation(layer=l)

            # If this is the last layer
            if l == last_layer:
                # Do not activate if take only preactivation
                # i.e preactivation == True
                if not preactivation: 
                    x = activation(x)
            
            # If not last layer - activate and go to next layer
            else:
                x = activation(x)
        return x 
    
    def forward(self, x):
        return self.U(self.v(x))

def get_model(in_dim=784, out_dim=64, hidden_dim=128, L=10, device=None):
    return Net(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, L=L, device=device)

# Define loss functions
def logistic_loss(y, y_positive, y_negatives):
    N, d = y.shape
    h_exp_sum = 0.0
    
    for y_negative in y_negatives:
        h_exp_sum += torch.exp(
            -torch.matmul(
                y.reshape(N, 1, d), 
                (y_positive - y_negative).reshape(N, d, 1)
            ).squeeze(1)
        )
    loss = torch.log(1 + h_exp_sum)
    return loss

# Compute Yunwen's complexity measure
def compute_complexity_FRO(dataloader, network: Net):
    # Report
    print('[INFO] Computing Yunwen et. al. complexity measure...')

    # Get necessary constants
    L = network.L 
    n = len(dataloader)
    d = network.out_dim
    
    # Compute complexity
    complexity = np.sqrt(L * d)
    for l in range(1, L+1):
        A_l = network._get_v_layer_weights(layer=l)
        complexity *= frobenius_norm(A_l)
        complexity *= spectral_norm(A_l)
    complexity = complexity / np.sqrt(n)

    # Find B_x
    B_x = 0.0
    with tqdm.tqdm(total=N) as pbar:
        for i, batch in enumerate(dataloader):
            # Calculate loss
            y1, y2, y3 = apply_model_to_batch(network, batch, device=network.device)
            loss = logistic_loss(y1, y2, y3)

            # Update progress
            pbar.update(1)

    return complexity