from functions_serialform import *
import pickle

M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.08,0.08], dtype=float)
begin_time = datetime.datetime.now()
Multipl_NOMA(M,sigma_n,train_params=cp.array([60,300,0.002]),canc_method='nn', modradius=cp.array([1,1]), plotting=True)
print(datetime.datetime.now() - begin_time)                                     


#sigma_n=torch.tensor([0.03,0.03,0.03])
#M=torch.tensor([4,4,4])
#alph=[np.sqrt(2)/9,1/3*np.sqrt(2),1]

#_,enc, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[200,1200,0.002],canc_method='nn', modradius=alph, plotting=False)

#with open('3user_modraduis.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([ enc, dec, canc, gmi, ser, gmi_exact], f)
