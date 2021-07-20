from training_routine import *

M=torch.tensor([8,8], dtype=int)
sigma_n=torch.tensor([0.06,0.06], dtype=float)
Multipl_NOMA(M,sigma_n,train_params=cp.array([100,300,0.002]),canc_method='nn', modradius=torch.tensor([1,1/3*np.sqrt(2)]), plotting=True)
