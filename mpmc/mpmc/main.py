
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .utils import L2discrepancy, hickernell_all_emphasized
from .models import MPMC_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rmpmc(nsamples, dim, 
          show_progress=True,
          **kwargs):
    """
        Train MPMC model with given number of samples and dimensionality.

        Args:

            nsamples: number of samples

            dim: dimensionality

            show_progress: whether to show progress bar or not

            **kwargs: additional arguments
        
    """    

    if 'epochs' not in kwargs: 
        epochs = 200000
    
    if 'loss_fn' not in kwargs:
        loss_fn = 'L2'
    
    if 'n_hid' not in kwargs:
        n_hid = 128
    
    if 'n_layers' not in kwargs:
        n_layers = 2
    
    if 'n_batch' not in kwargs:
        n_batch = 16
    
    if 'radius' not in kwargs:
        radius = 0.35

    if 'dim_emphasize' not in kwargs:
        dim_emphasize = [1]
    
    if 'n_projections' not in kwargs:
        n_projections = 15

    model = MPMC_net(dim=dim, nsamples=nsamples, 
                     nhid=n_hid, nlayers=n_layers, 
                     nbatch=n_batch, radius=radius, 
                     loss_fn=loss_fn, dim_emphasize=dim_emphasize, 
                     n_projections=n_projections).to(device)

    if ('lr' in kwargs) and ('weight_decay' in kwargs):
        optimizer = optim.Adam(model.parameters(), 
                               lr=lr, weight_decay=weight_decay)
    else: 
        lr = 0.001
        weight_decay=1e-6
        optimizer = optim.Adam(model.parameters(), 
                               lr=lr, weight_decay=weight_decay)

    best_loss = 10000.
    patience = 0

    ## could be tuned for better performance
    start_reduce = 100000
    reduce_point = 10

    if show_progress:
        iterator = tqdm(range(epochs))
    else:
        iterator = range(epochs)
    
    mpmc_points = []

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        loss, X = model()
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            y = X.clone()
            if loss_fn == 'L2':
                batched_discrepancies = L2discrepancy(y.detach())
            elif loss_fn == 'approx_hickernell':
                ## compute sum over all projections with emphasized dimensionality:
                batched_discrepancies = hickernell_all_emphasized(y.detach(),dim_emphasize)
            else:
                NotImplementedError('Loss function not implemented')
            min_discrepancy, mean_discrepancy = torch.min(batched_discrepancies).item(), torch.mean(batched_discrepancies).item

            if min_discrepancy < best_loss:
                best_loss = min_discrepancy

                ## save MPMC points:
                PATH = 'outputs/dim_'+str(dim)+'/nsamples_'+str(nsamples)+'.npy'
                y = y.detach().cpu().numpy()
                mpmc_points.append(y)

            if (min_discrepancy > best_loss and (epoch + 1) >= start_reduce):
                patience += 1

            if (epoch + 1) >= start_reduce and patience == reduce_point:
                patience = 0
                lr /= 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if (lr < 1e-6):
                break

    return np.asarray(mpmc_points)