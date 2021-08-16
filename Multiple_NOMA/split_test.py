# from training_routine_additive import *

# M=torch.tensor([4,4], dtype=int)
# sigma_n=torch.tensor([0.08,0.08], dtype=float)
# Add_NOMA(M,sigma_n,train_params=cp.array([50,300,0.01]),canc_method='diff', modradius=torch.tensor([2/3,1/3],device=device), plotting=True)


from training_routine import *
M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.09,0.09], dtype=float)
Multipl_NOMA(M,sigma_n,train_params=cp.array([60,300,0.002]),canc_method='div', modradius=torch.tensor([1,1],device=device), plotting=True)
 