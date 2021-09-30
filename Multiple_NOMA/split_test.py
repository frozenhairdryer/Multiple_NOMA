from training_routine_additive import *

M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.18,0.18], dtype=float)
Add_NOMA(M,sigma_n,train_params=cp.array([50,300,0.01]),canc_method='diff', modradius=torch.tensor([2/3,1/3],device=device), plotting=True)


# from training_routine import *
# M=torch.tensor([4,4], dtype=int)
# sigma_n=torch.tensor([0.18,0.18], dtype=float)
# Multipl_NOMA(M,sigma_n,train_params=cp.array([60,600,0.005]),canc_method='nn', modradius=torch.tensor([1,1],device=device), plotting=True)
 