import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Network settings
d = 64
L_min = 2
L_max = 3
hidden_dim = 128

# Data settings
k = 3
batch_size = 32
data_ratio = 0.1

# Training settings
epochs = 10

# Get GPU if applicable
device = (
    "cuda"
    if torch.cuda.is_available() else "mps"
    if torch.backends.mps.is_available() else "cpu"
)
    
# Norm calculation functions
def _l21_norm(A):
    norm = 0.0
    for j in range(A.shape[1]):
        aj_l2 = np.linalg.norm(A[:, j])
        norm += aj_l2
    return norm

def _frobenius_norm(A):
    return np.linalg.norm(A, ord='fro')

def _spectral_norm(A):
    sv = np.linalg.svd(A).S
    return max(sv)

# Network definition
class Net(nn.Module):
    def __init__(self, in_dim=784, out_dim=64, hidden_dim=128, L=10, device=device):
        super().__init__()
        
        # Store configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.L = L
        
        # Create layers
        self.fc_hidden_layers = []
        for i in range(1, self.L):
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
        
    def _tensor_to_numpy(self, x):
        return x.cpu().detach().numpy()            
        
    def _get_v_layer_linear(self, layer=1):
        return list(self.v.modules())[0][layer*2-2]
    
    def _get_v_layer_activation(self, layer=1):
        return list(self.v.modules())[0][layer*2-1]
    
    def _get_v_layer_weights(self, layer=1):
        v_layer = self._get_v_layer_linear(layer=layer)
        return self._tensor_to_numpy(v_layer.weight)
    
    def _get_hidden_output(self, x, last_layer=1, last_activation=True):
        for l in range(1, last_layer + 1):
            x = self._get_v_layer_linear(layer=l)(x)
            activation = self._get_v_layer_activation(layer=l)
            if(l == last_layer):
                if(last_activation): x = activation(x)
            else:
                x = activation(x)
        return np.linalg.norm(self._tensor_to_numpy(x), ord=2)
    
    def _calculate_network_complexitites(self, dataloader):
        max_rhos = [0 for _ in range(self.L)]
        max_Bl = [0 for _ in range(self.L)]
        max_x_l2 = 0.0
        
        # Calculations
        print('[INFO] Calculating spectral complexity...')
        for l in range(1, self.L+1):
            print(f'[INFO] Calculating statistics for layer #{l}:')
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for j, batch in enumerate(train_dataloader):
                    X = torch.cat(
                        [batch[0].to(device), batch[1].to(device), *[x.to(device) for x in batch[2]]],
                        dim=0
                    )
                    for x in X:
                        # Calculations
                        x_l2      = np.linalg.norm(self._tensor_to_numpy(x), ord=2)
                        Bl_act    = self._get_hidden_output(x, last_layer=l)
                        Bl_no_act = self._get_hidden_output(x, last_layer=l, last_activation=False)
                        
                        max_rho   = 0.0
                        for U in range(l+1, self.L+1):
                            BU    = self._get_hidden_output(x, last_layer=U)
                            prod_spec_norm = np.prod([
                                _spectral_norm(self._get_v_layer_weights(layer=u)) for u in range(l+1, U+1)
                            ])
                            BU_prod_spec_norm = prod_spec_norm / BU
                            if(max_rho < BU_prod_spec_norm):
                                max_rho = BU_prod_spec_norm
                            
                        
                        # Update max values
                        if(max_x_l2 < x_l2): max_x_l2 = x_l2
                        if(max_Bl[l-1] < Bl_act): max_Bl[l-1] = Bl_act
                        if(max_rhos[l-1] < max_rho): max_rhos[l-1] = max_rho
                    pbar.update(1)
                    if(j==0): break
        max_Bl.insert(0, max_x_l2)
        
        # Last layer
        RA = 0.0
        for l in range(1, self.L + 1):
            RA += ((
                _l21_norm(self._get_v_layer_weights(layer=l).T) \
                * max_Bl[l-1] \
                * max_rhos[l-1]
            ) ** (2/3))
            print('l', RA**(3/2))
            
        RA += ((
            np.sqrt(self.out_dim) \
            * _frobenius_norm(self._tensor_to_numpy(self.U.weight)) \
            * max_Bl[self.L]
        ) ** (2/3))
        RA = RA ** (3/2)
        print('L', RA)
        
        # Compare with Yun wei
        fro_prod = 1.0
        for l in range(1, self.L+1):
            weight = self._get_v_layer_linear(layer=l).weight
            fro_prod *= _frobenius_norm(self._tensor_to_numpy(weight))
        fro_prod *= np.sqrt(self.out_dim * self.L) * max_x_l2
        return RA, fro_prod
    
    def forward(self, x):
        return self.U(self.v(x))
    
    
# Data loader definition
class MNISTTripletDataset(Dataset):
    def __init__(self, mnist_dataset, k):
        self.mnist_dataset = mnist_dataset
        self.k = k
        self.label_to_indices = defaultdict(list)
        
        for idx, (_, label) in enumerate(self.mnist_dataset):
            self.label_to_indices[label].append(idx)
        
        self.indices_by_label = {label: np.array(indices) for label, indices in self.label_to_indices.items()}
        self.all_labels = list(self.label_to_indices.keys())

    def __getitem__(self, index):
        x, label = self.mnist_dataset[index]
        x = x.view(-1)
        
        # Select x_positive with the same label
        positive_idx = np.random.choice(self.indices_by_label[label])
        while positive_idx == index:
            positive_idx = np.random.choice(self.indices_by_label[label])
        x_positive, _ = self.mnist_dataset[positive_idx]
        x_positive = x_positive.view(-1)

        # Select k negative samples with different labels
        negative_samples = []
        negative_labels = np.random.choice([l for l in self.all_labels if l != label], self.k, replace=False)
        for neg_label in negative_labels:
            negative_idx = np.random.choice(self.indices_by_label[neg_label])
            x_negative, _ = self.mnist_dataset[negative_idx]
            x_negative = x_negative.view(-1)
            negative_samples.append(x_negative)
        
        return (x, x_positive, negative_samples)

    def __len__(self):
        return len(self.mnist_dataset)

def _get_mnist_dataloaders(k=3, batch_size=32):
    # Set up transformations and load the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
    # Create the custom datasets
    train_dataset = MNISTTripletDataset(mnist_train, k)
    test_dataset = MNISTTripletDataset(mnist_test, k)
    
    # Create custom dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_dataloader, test_dataloader

def _apply_model_to_batch(model, batch, device=None):
    # Unpack the batch
    x1, x2, x3 = batch[0].to(device), batch[1].to(device), [x.to(device) for x in batch[2]]
    
    # Apply model to batch
    y1, y2, y3 = model(x1), model(x2), [model(x) for x in x3]
    return y1, y2, y3

# Define loss functions
def _logistic_loss(y, y_positive, y_negatives):
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

# Running
if __name__ == '__main__':
    # Initialize model
    train_dataloader, test_dataloader = _get_mnist_dataloaders(k=k, batch_size=batch_size)
    model = Net(in_dim=784, out_dim=d, hidden_dim=hidden_dim, L=L_max).to(device)
    RA, FRO = model._calculate_network_complexitites(train_dataloader)
    print(RA, FRO)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    
    # # Load data
    # train_dataloader, test_dataloader = _get_mnist_dataloaders(k=k, batch_size=batch_size)
    
    
    # Start training
    # print(model)
    # model.train()
    # for epoch in range(epochs):
    #     print(f'[*] Epoch #[{epoch+1}/{epochs}]:')
    #     with tqdm.tqdm(total=len(train_dataloader)) as pbar:
    #         for i, batch in enumerate(train_dataloader):
    #             # Calculate loss
    #             y1, y2, y3 = _apply_model_to_batch(model, batch, device=device)
    #             loss = torch.sum(_logistic_loss(y1, y2, y3))            
                   
    #             # Back propagation
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
                
    #             # Update progress bar
    #             pbar.set_postfix({
    #                 'train_loss' : f'{loss.item():.5f}'
    #             })
    #             pbar.update(1)
    