import datetime
""" ## choice controls the kind of simulation ##
choice = 1    : additive NOMA
choice = 2    : multiplicative NOMA
choice = 3    : theoretical multiplicative 16-QAM constellation
choice = 4    : multiplicative NOMA including rectangular pulseshaping with the option to simulate chromatic dispersion
"""

choice = 2

begin_time = datetime.datetime.now()
if choice == 1:
    from training_routine_additive import *
    M=torch.tensor([4,4], dtype=int)
    sigma_n=torch.tensor([0.18,0.18], dtype=float)
    Add_NOMA(M,sigma_n,train_params=cp.array([50,300,0.01]),canc_method='diff', modradius=torch.tensor([2/3,1/3],device=device), plotting=True)
elif choice == 2:
    from training_routine import *
    M=torch.tensor([4,4], dtype=int)
    sigma_n=torch.tensor([0.18,0.18], dtype=float)
    #Multipl_NOMA(M,sigma_n,train_params=cp.array([60,600,0.003]),canc_method='nn', modradius=torch.tensor([1,1],device=device), plotting=True)
    Multipl_NOMA(M,sigma_n,train_params=cp.array([10,600,0.005]),canc_method='div', modradius=torch.tensor([1,1],device=device), plotting=True)
elif choice == 3:
    from theoreticalMNOMA import *
    M=torch.tensor([4,4], dtype=int)
    sigma_n=torch.tensor([0.18,0.18], dtype=float)
    t_Multipl_NOMA(M,sigma_n,train_params=cp.array([60,300,0.002]),canc_method='nn', modradius=torch.tensor([1,1/3*np.sqrt(2)],device=device), plotting=True)
elif choice == 4:
    from waveform_training_routine import *
    M=torch.tensor([4,4], dtype=int)
    sigma_n=torch.tensor([0.18,0.18], dtype=float)
    waveform_Multipl_NOMA(M,sigma_n,train_params=cp.array([60,600,0.01]),canc_method='div', modradius=torch.tensor([1,0.58],device=device), plotting=True, chrom_disp=True)
else:
    print("Choose valid simulation option for choice in split_test.py !")

print("Skript runtime: "+ str(datetime.datetime.now() - begin_time))