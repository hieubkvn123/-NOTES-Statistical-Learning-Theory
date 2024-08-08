import torch
import tqdm
from model import Net, logistic_loss
from dataset import UnsupervisedDataset, get_dataloader

# Configurations
BATCH_SIZE = 128
MAX_EPOCHS = 100
MAX_LR = 0.001
WEIGHT_DECAY = 0.001

# Utils
def apply_model_to_batch(model, batch, device=None):
    # Unpack the batch
    x1, x2, x3 = batch[0].to(device), batch[1].to(device), [x.to(device) for x in batch[2]]
    
    # Apply model to batch
    y1, y2, y3 = model(x1), model(x2), [model(x) for x in x3]
    return y1, y2, y3 

def train(epochs, d_dim=64, hidden_dim=128, k=3, batch_size=64, num_batches=1000):
    # Get dataset 
    train_dataloader, test_dataloader = get_dataloader(name='mnist', k=k, batch_size=batch_size, num_batches=num_batches)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    
    # Load model
    model = Net(in_dim=784, out_dim=d_dim, hidden_dim=hidden_dim, L=2)
    model = model.to(model.device)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=MAX_LR, 
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
        print(f'Average test loss : {total_loss/(num_test_batches*batch_size)}')

if __name__ == '__main__':
    train(epochs=1)

