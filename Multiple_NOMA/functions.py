from imports import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def SER(predictions, labels):
    s2 = torch.argmax(predictions, 1)
    return torch.sum( s2!= labels)/predictions.shape[0]

def BER(predictions, labels,m):
    # Bit representation of symbols
    binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,m,dtype='uint8')), (-1,8)), device=device)
    binaries = binaries[:,int(8-torch.log2(m)):]
    y_valid_binary = binaries[labels]
    pred_binary = binaries[torch.argmax(predictions, axis=1),:]
    ber=torch.zeros(int(torch.log2(m)), device=device, requires_grad=True)
    ber = 1-torch.mean(torch.isclose(pred_binary, y_valid_binary,rtol=0.5),axis=0, dtype=float)
    return ber

def GMI_est(SERs, M, ber=None):
    # gmi estimate 1 or both estimates, if bers are given
    gmi_est=0
    if M.any()==1:
        length=1
    else:
        length=len(M)
    for mod in range(length):
        Pe = torch.min(SERs[mod],torch.tensor(0.5)) # all bit are simultaneously wrong
        gmi_est += torch.log2(M[mod])*(1+Pe*torch.log2(Pe+1e-12)+(1-Pe)*torch.log2((1-Pe)+1e-12))
    if ber!=None:
        gmi=torch.zeros(len(M),int(torch.log2(max(M))))
        for num in range(len(M)):
            for x in range(int(torch.log2(M[num]))):
                gmi[num][x]=((1+(ber[num][x]*torch.log2(ber[num][x]+1e-12)+(1-ber[num][x])*torch.log2((1-ber[num][x])+1e-12))))    
        return gmi_est, gmi.flatten()
    else:
        return gmi_est

def GMI(M, my_outputs, mylabels):
    # gmi estimate or calculation, if bers are given
    if mylabels!=None and my_outputs!=None:
        gmi=torch.zeros(int(torch.log2(M)), device=device)
        binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,M,dtype='uint8')), (-1,8)),dtype=torch.float32, device=device)
        binaries = binaries[:,int(8-torch.log2(M)):]
        b_labels = binaries[mylabels].int()
        # calculate bitwise estimates
        bitnum = int(torch.log2(M))
        b_estimates = torch.zeros(len(my_outputs),bitnum, device=device)
        P_sym = torch.bincount(mylabels)/len(mylabels)
        for bit in range(bitnum):
            pos_0 = torch.where(binaries[:,bit]==0)[0]
            pos_1 = torch.where(binaries[:,bit]==1)[0]
            est_0 = torch.sum(torch.index_select(my_outputs,1,pos_0), axis=1) +1e-12
            est_1 = torch.sum(torch.index_select(my_outputs,1,pos_1), axis=1) +1e-12 # increase stability
            #print(my_outputs)
            llr = torch.log(est_0/est_1+1e-12)
            gmi[bit]=1-1/(len(llr))*torch.sum(torch.log2(torch.exp((2*b_labels[:,bit]-1)*llr)+1+1e-12), axis=0)
        return gmi.flatten()

def chromatic_dispersion(sigIN,fa):
    #applies chromatic dispersion to the signal
    c0 = 299792458                                              # in m/s
    alpha = 0                                                   # 10**(0/10) # Attenuation approx 0.2 [1/km] 
    lam = 1550e-9                                               # wavelength, [nm]
    D = 20                                                      # [ps/nm/km]
    beta2 = - (D * np.square(lam)) / (2 * np.pi * 3e8) * 1e-3   # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
    L = 50                                                      # in km
    sigINf = np.fft.fftshift(np.fft.fft(sigIN))
    f=np.linspace(-1/2,1/2,(len(sigIN)))*fa
    
    HCD = np.exp(( - 1j * beta2 / 2 * np.square(2*np.pi*f)) * L - alpha / 2 * L )
    sigOUTf = sigINf*HCD
    sigOUT = np.fft.ifft(np.fft.ifftshift(sigOUTf))
    return sigOUT

def channel(signal, fa, sigma, cd=False):
    """Channel Simulation
    Upsampled signal is processed according to noise, Chromatic dispersion parameters
    Args:
    signal (float): input signal to channel
    sigma (float): Noise parameter (sigma^2 is noise variance).
    cd (bool): toggles the simulation of chromatic dispersion for a fiber of 50km, D=17 ps/nm*km    

    Returns:
        r_signal: signal after channel, still upsampled

    """
    if cd == True:
        signal = chromatic_dispersion(signal, fa)
    r_signal = torch.add(signal, (0.5*sigma*torch.randn(len(signal)).to(device)+ 1j*0.5*sigma*torch.randn(len(signal)).to(device)))
    return r_signal

def pulseshape(samples, n_up, shape='rect'):
    """Pulse Shaping
    Upsample signal and apply pulseshape 'shape'
    Args:
    samples (complex): signal samples
    n_up (int): Upsampling factor
    shape (str): pulse shape, supported: 'rect', 'sinc'    

    Returns:
        r_signal: signal after pulseshaping
        fa : sampling frequency for upsampled signal

    """
    t_symb = 3.2*1e-10     # Symbol duration for optical fiber
    syms_per_filt = 6      # symbols per filter (plus minus in both directions)
    K_filt = 2 * syms_per_filt * n_up + 1         # length of the fir filter
    fa = 1/t_symb*n_up
    if shape!='rect' and shape!='sinc':
        raise ValueError("Variable shape can only take 'rect', 'rc' or 'sinc'!")
    
    if shape=='sinc':
        pulse = np.sinc(np.linspace(-syms_per_filt,syms_per_filt, K_filt))
        pulse /= np.max(pulse)

    if shape=='rect':
        pulse = np.append( np.ones( n_up ), np.zeros( K_filt - n_up ) )
        pulse /= np.max(pulse)
        pulse = np.roll(pulse,int((K_filt-n_up)/2))
    
    s_up = np.zeros( len(samples) * n_up , dtype=complex)      
    s_up[ : : n_up ] = samples
    r_signal = np.convolve(samples, pulse)
    return r_signal, fa




def plot_training(SERs,valid_r,cvalid,M, const, GMIs_appr, decision_region_evolution, meshgrid, constellation_base, gmi_exact, gmi_hd):
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 10,
    })
    
    cmap = matplotlib.cm.tab20
    base = plt.cm.get_cmap(cmap)
    color_list = base.colors
    new_color_list = np.array([[t/2 + 0.49 for t in color_list[k]] for k in range(len(color_list))])

    sum_SERs = np.sum(SERs, axis=0)/len(M)
    min_SER_iter = np.argmin(cp.sum(SERs,axis=0))
    max_GMI = np.argmax(np.sum(gmi_exact, axis=1))
    ext_max_plot = 1.2#*np.max(np.abs(valid_r[int(min_SER_iter)]))

    print('Minimum mean SER obtained: %1.5f (epoch %d out of %d)' % (sum_SERs[min_SER_iter], min_SER_iter, len(SERs[0])))
    print('Maximum obtained GMI: %1.5f (epoch %d out of %d)' % (np.sum(gmi_exact[max_GMI]),max_GMI,len(GMIs_appr)))
    print('The corresponding constellation symbols are:\n', const)

    plt.figure("SERs",figsize=(3.5,3.5))
    #plt.figure("SERs",figsize=(3.5,3.5))
    for num in range(len(M)):
        plt.plot(SERs[num],marker='.',linestyle='--',markersize=2, label="Enc"+str(num))
        plt.plot(min_SER_iter,SERs[num][min_SER_iter],marker='o',markersize=3,c='red')
        plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[num][min_SER_iter]),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('SER')
    plt.grid(which='both')
    plt.legend(loc=1)
    plt.title('SER on Validation Dataset')
    plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig("Multiple_NOMA/figures/Sers.pdf")
    #tikzplotlib.save("figures/SERs.tex", strict=True, externalize_tables=True, override_externals=True)

    plt.figure("GMIs",figsize=(3,2.5))
    #plt.plot(GMIs_appr.cpu().detach().numpy(),linestyle='--',label='Appr.')
    #plt.plot(gmi_hd,linestyle='--',label='GMI Hard decision')
    #plt.plot(max_GMI,GMIs_appr[max_GMI],c='red')
    for num in range(len(gmi_exact[0,:])):
        if num==0:
            t=gmi_exact[:,num]
            plt.fill_between(np.arange(len(t)),t, alpha=0.4)
        else:
            plt.fill_between(np.arange(len(t)),t,(t+gmi_exact[:,num]),alpha=0.4)
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
    #tikzplotlib.save("figures/gmis.tex", strict=True, externalize_tables=True, override_externals=True)


    constellations = np.array(const.get()).flatten()
    bitmapping=[]
    torch.prod(M)
    int(torch.prod(M))
    helper= np.arange((int(torch.prod(M))))
    for h in helper:
        bitmapping.append(format(h, '04b'))

    plt.figure("constellation", figsize=(3,3))
    #plt.subplot(121)
    plt.scatter(np.real(constellations),np.imag(constellations),c=range(np.product(M.cpu().detach().numpy())), cmap='tab20',s=50)
    for i in range(len(constellations)):
        plt.annotate(bitmapping[i], (np.real(constellations)[i], np.imag(constellations)[i]))
    
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    plt.grid(which='both')
    plt.title('Constellation')
    plt.tight_layout()
    plt.savefig("Multiple_NOMA/figures/constellation.pdf")
    #tikzplotlib.save("figures/constellation.tex", strict=True, externalize_tables=True, override_externals=True)

    val_cmplx=np.array((valid_r[min_SER_iter][:,0]+1j*valid_r[min_SER_iter][:,1]).get())

    plt.figure("Received signal",figsize=(2.7,2.7))
    #plt.subplot(122)
    plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    plt.grid()
    plt.title('Received')
    plt.tight_layout()
    plt.savefig("Multiple_NOMA/figures/received.pdf")
    #tikzplotlib.save("figures/received.tex", strict=True, externalize_tables=True, override_externals=True)

    
    
    
    plt.figure("Decision regions", figsize=(5,3))
    for num in range(len(M)):
        plt.subplot(1,len(M),num+1)
        decision_scatter = np.argmax(decision_region_evolution[num], axis=1)
        grid=meshgrid.get()
        if num==0:
            plt.scatter(grid[:,0], grid[:,1], c=decision_scatter,s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:int(M[num])]))
        else:
            plt.scatter(grid[:,0], grid[:,1], c=decision_scatter,s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[int(M[num-1]):int(M[num-1])+int(M[num])]))
        #plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
        plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
        plt.axis('scaled')
        plt.xlim((-ext_max_plot,ext_max_plot))
        plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.title('Decoder %d' % (num+1))
    plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig("Multiple_NOMA/figures/decision_regions.pdf")
    #tikzplotlib.save("figures/decision_regions.tex", strict=True, externalize_tables=True, override_externals=True)

    
    plt.figure("Base Constellations", figsize=(5,3))
    for num in range(len(M)):
        bitm=[]
        helper= np.arange(int(M[num]))
        for h in helper:
            bitm.append(format(h, '02b'))
        plt.subplot(1,len(M),num+1)
        plt.title("Decoder "+str(num+1))
        plt.scatter(np.real(constellation_base[num]),np.imag(constellation_base[num]), c=np.arange(int(M[num])))
        for bit in range(len(bitm)):
            plt.annotate(bitm[bit],(np.real(constellation_base[num][bit]),np.imag(constellation_base[num][bit])))
        plt.xlim((-ext_max_plot,ext_max_plot))
        plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.grid()
    plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig("Multiple_NOMA/figures/base_constellation.pdf")
    #tikzplotlib.save("figures/base_constellations.tex", strict=True, externalize_tables=True, override_externals=True)
    #tikzplotlib.save(f'{output_path}{output_fname}.tex', figure=fig1, wrap=False, add_axis_environment=False, externalize_tables=True, override_externals=True)
    #plt.show()

