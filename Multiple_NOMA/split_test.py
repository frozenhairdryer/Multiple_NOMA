from training_routine_additive import *

M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.08,0.08], dtype=float)
Add_NOMA(M,sigma_n,train_params=cp.array([100,300,0.01]),canc_method='none', modradius=torch.tensor([2/3,1/3],device=device), plotting=True)
