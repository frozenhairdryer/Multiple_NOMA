# from training_routine_additive import *

# M=torch.tensor([4,4], dtype=int)
# sigma_n=torch.tensor([0.08,0.08], dtype=float)
# Add_NOMA(M,sigma_n,train_params=cp.array([50,300,0.01]),canc_method='diff', modradius=torch.tensor([2/3,1/3],device=device), plotting=True)


from training_routine import *
M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.09,0.09], dtype=float)
mradius = torch.tensor([1, 1/3*np.sqrt(2)],device=device)

encs=nn.ModuleList([])
GMI_serial=[]
for m in range(len(M)):
    if m==0:
        M=torch.tensor([4,1], dtype=int)
        sigma_n=torch.tensor([0.09,0], dtype=float)
        mradius = torch.tensor([1, 1],device=device)
        canc_method,encoder_best,dec_best, gmi, validation_SERs,gmi_exact=Multipl_NOMA(M,sigma_n,train_params=cp.array([30,300,0.002]),canc_method='div', modradius=mradius, plotting=False)
    elif m==len(M):
        M=torch.tensor([4,4], dtype=int)
        sigma_n=torch.tensor([0.09,0.09], dtype=float)
        mradius = torch.tensor([1, 1/3*np.sqrt(2)],device=device)
        canc_method,enc_best,dec_best, gmi, validation_SERs,gmi_exact=Multipl_NOMA(M,sigma_n,train_params=cp.array([30,300,0.002]),canc_method='div', modradius=mradius, plotting=True, encoder=encs)
    else:
         canc_method,enc_best,dec_best, gmi, validation_SERs,gmi_exact=Multipl_NOMA(M,sigma_n,train_params=cp.array([30,300,0.002]),canc_method='div', modradius=mradius, plotting=False, encoder=encs)
    encs.append(enc_best)
    g = gmi_exact.detach().cpu().numpy()
    GMI_serial.append(g)
 
matplotlib.rcParams.update({
"pgf.texsystem": "pdflatex",
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'font.size' : 10,
})

plt.figure("GMIs",figsize=(3,2.5))
    #plt.plot(GMIs_appr.cpu().detach().numpy(),linestyle='--',label='Appr.')
    #plt.plot(gmi_hd,linestyle='--',label='GMI Hard decision')
    #plt.plot(max_GMI,GMIs_appr[max_GMI],c='red')
for lnum in range(len(M)):
    offset=lnum*30
    gmi_exact = GMI_serial[lnum]
    for num in range(len(gmi_exact[0,:])):
        if num==0:
            t=gmi_exact[:,num]
            plt.fill_between(np.arange(len(t))+offset,t, alpha=0.4)
        else:
            plt.fill_between(np.arange(len(t))+offset,t,(t+gmi_exact[:,num]),alpha=0.4)
            t+=gmi_exact[:,num]
    plt.plot(t, label='GMI')
    plt.plot(argmax(t),max(t),marker='o',c='red')
    plt.annotate('Max', (0.95*argmax(t),0.9*max(t)),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('GMI')
    plt.ylim(0,4)
    plt.legend(loc=3)
    plt.grid(which='both')
    plt.title('GMI on Validation Dataset')
    plt.tight_layout()
    plt.savefig("Multiple_NOMA/figures/gmis.pdf")