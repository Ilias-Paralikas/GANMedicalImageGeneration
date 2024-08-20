import torch 

def generate_fixed_random_tensor(seed, size,device='cpu'):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        random_tensor = torch.rand(1,*size,device=device)
    return random_tensor
