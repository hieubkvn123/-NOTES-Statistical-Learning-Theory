import time
import tqdm
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataset import get_dataloader
from common import apply_model_to_batch
from model import (
    get_model, 
    logistic_loss,
    compute_complexity_YW,
    compute_complexity_THM1,
    compute_complexity_THM2,
    compute_complexity_THM3
)

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'weight' : 'bold',
    'size' : 18
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
MAX_EPOCHS = 10 
BATCH_SIZE = 64

# Constants for ablation study
MIN_WIDTH = 1
MAX_WIDTH = 8
MIN_DEPTH = 2
MAX_DEPTH = 10
DATASET_TO_INDIM = {'mnist' : 784}
RESULT_KEYS = {'ar': 'Arora et al.', 'yw' : 'Lei et al.', 'thm1' : 'Ours (Thm. 1)', 'thm2' : 'Ours (Thm. 2)', 'thm3' : 'Ours (Thm. 3)'}
COLOR_KEYS = {'ar' : 'tab:orange', 'yw' : 'tab:red', 'thm1' : 'tab:blue', 'thm2' : 'tab:purple', 'thm3' : 'tab:green'}

def train(epochs, dataset='mnist', d_dim=64, hidden_dim=128, k=3, L=2, batch_size=64, num_batches=1000):
    # Get dataset 
    train_dataloader, test_dataloader = get_dataloader(name=dataset, k=k, batch_size=batch_size, num_batches=num_batches)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    
    # Load model
    model = get_model(in_dim=DATASET_TO_INDIM[dataset], out_dim=d_dim, hidden_dim=hidden_dim, L=L)
    model = model.to(model.device)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001, 
        amsgrad=True)

    # Train model
    model.train()
    for epoch in range(epochs):
        print(f'[*] Epoch #[{epoch+1}/{epochs}]:')
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):
                # Calculate loss
                y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                loss = logistic_loss(y1, y2, y3)
                total_loss_batchwise = torch.sum(loss) 
                    
                # Back propagation
                total_loss_batchwise.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Update loss for this epoch
                total_loss += total_loss_batchwise.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss' : f'{total_loss_batchwise.item():.5f}',
                    'batch' : f'#[{i+1}/{num_train_batches}]'
                })
                pbar.update(1)
            time.sleep(0.1)
            print(f'\nAverage train loss : {total_loss/(num_train_batches*batch_size):.4f}\n------\n')

    # Evaluate the model
    model.eval()
    print('------\nEvaluation:')
    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        total_loss = 0.0
        for i, batch in enumerate(test_dataloader):
            # Calculate loss
            y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
            loss = logistic_loss(y1, y2, y3)
            total_loss_batchwise = torch.sum(loss) 
            
            # Update loss
            total_loss += total_loss_batchwise.item()

            # Update progress bar
            pbar.set_postfix({
                'test_loss' : f'{total_loss_batchwise.item():.5f}',
                'batch' : f'#[{i+1}/{num_test_batches}]'
            })
            pbar.update(1)
        time.sleep(0.1)
        print(f'Average test loss : {total_loss/(num_test_batches*batch_size)}')

    # Evaluate complexity measures
    print('------\nComplexity measures computation:')
    complexity_YW = np.log(compute_complexity_YW(train_dataloader, model))
    complexity_AR = np.sqrt(k) * complexity_YW
    complexity_THM1 = np.log(compute_complexity_THM1(train_dataloader, model))
    complexity_THM2 = np.log(compute_complexity_THM2(train_dataloader, model))
    complexity_THM3 = np.log(compute_complexity_THM3(train_dataloader, model))
    return complexity_AR, complexity_YW, complexity_THM1, complexity_THM2, complexity_THM3

def results_visualization_utils(results, xaxis_data, title, xlabel, ylabel, save_path='file.png'):
    # Initialize plot
    _, ax = plt.subplots(figsize=(10, 7))
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Visualize
    for key, result in results.items():
        ax.plot(xaxis_data, result, label=RESULT_KEYS[key], color=COLOR_KEYS[key], marker='o')
    ax.set_xlabel(xlabel, fontdict=fontconfig)
    ax.set_ylabel(ylabel, fontdict=fontconfig)
    plt.grid()
    plt.legend(loc='upper left', fontsize="15")
    plt.title(title, fontdict=fontconfig)
    plt.tight_layout()
    plt.savefig(save_path)

def ablation_study_varying_depths(args, min_depth, max_depth):
    # Initialize results
    depths = list(range(min_depth, max_depth + 1))
    results_depth = { 'ar' : [], 'yw': [], 'thm1': [], 'thm2': [], 'thm3': []}

    # Conduct training
    for i, L in enumerate(depths):
        print(f'[INFO] Experiment #[{i+1}/{len(depths)}], L = {L}')
        ar, yw, thm1, thm2, thm3 = train(
            epochs=MAX_EPOCHS, 
            batch_size=BATCH_SIZE,
            L=L,
            dataset=args['dataset'],
            hidden_dim=args['hidden_dim'],
            d_dim=args['output_dim'],
            k=args['k'],
            num_batches=args['n'], 
        )
        results_depth['ar'].append(ar)
        results_depth['yw'].append(yw)
        results_depth['thm1'].append(thm1)
        results_depth['thm2'].append(thm2)
        results_depth['thm3'].append(thm3)
    return depths, results_depth

def ablation_study_varying_widths(args, min_width, max_width):
    # Initialize results
    widths = list(range(min_width, max_width + 1))
    results_width = { 'ar' : [], 'yw': [], 'thm1': [], 'thm2': [], 'thm3': []}

    # Conduct training
    for i, W in enumerate(widths):
        print(f'[INFO] Experiment #[{i+1}/{len(widths)}], W = {W*32}')
        ar, yw, thm1, thm2, thm3 = train(
            epochs=MAX_EPOCHS, 
            batch_size=BATCH_SIZE,
            hidden_dim=W * 32,
            L=args['L'],
            dataset=args['dataset'],
            d_dim=args['output_dim'],
            k=args['k'],
            num_batches=args['n'], 
        )
        results_width['ar'].append(ar)
        results_width['yw'].append(yw)
        results_width['thm1'].append(thm1)
        results_width['thm2'].append(thm2)
        results_width['thm3'].append(thm3)
    return widths, results_width

if __name__ == '__main__':
    # Ablation study with depth
    args = {'dataset' : 'mnist', 'hidden_dim' : 64, 'output_dim' : 64, 'k' : 3, 'n' : 100}
    depths, results = ablation_study_varying_depths(args, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
    results_visualization_utils(
        results,
        xaxis_data=depths,
        title='Generalization bounds at varying depths',
        xlabel='Depths ($L$)',
        ylabel='Generalization bounds (log-scaled)',
        save_path='ablation_study_depth.png'
    )

    # Ablation study with width
    args = {'dataset' : 'mnist', 'L' : 3, 'output_dim' : 64, 'k' : 3, 'n' : 100}
    widths, results = ablation_study_varying_widths(args, min_width=MIN_WIDTH, max_width=MAX_WIDTH)
    results_visualization_utils(
        results,
        xaxis_data=widths,
        title='Generalization bounds at varying widths',
        xlabel='Widths ($W$ - in multiples of $32$)',
        ylabel='Generalization bounds (log-scaled)',
        save_path='ablation_study_width.png'
    )
