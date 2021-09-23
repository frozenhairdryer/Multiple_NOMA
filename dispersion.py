# importing
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# plotting options 
# font = {'size'   : 10}
# plt.rc('font', **font)
# plt.rc('text', usetex=matplotlib.checkdep_usetex(True))

#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

####
# Function: Apply chromatic dispersion to pulse
####
def cd(sigIN,L,D,fa,lamb,alpha):
    #cd applies chromatic dispersion to the signal
    c0 = 299792458; #in m/s
    sigINf = np.fft.fftshift(np.fft.fft(sigIN))
    f=np.linspace(-1/2,1/2,(len(sigIN)))*fa
    #f=np.fft.fftfreq(len(sigIN),t_samp)
    #exponent = 1j*np.pi*(lamb)**2/c0*D*L*f**2
    beta2 = - (D * np.square(lamb)) / (2 * np.pi * c0) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
    #HCD = np.exp(exponent)
    HCD = np.exp(( - 1j * beta2 / 2 * np.square(2*np.pi*f)) * L - alpha / 2 * L )
    sigOUTf = sigINf*HCD
    sigOUT = np.fft.ifft(np.fft.ifftshift(sigOUTf))
    return sigOUT


# modulation scheme and constellation points
#M = [4,4]
M=[2,2]
#mradius=1/3*np.sqrt(2)
#c2 = (1+mradius*np.array([1,-1j,1j,-1]))/(1+mradius)
#constellation_points = [[ -1, 1, 1j,-1j ],[1.+0.j, 0.67962276-0.32037724j,0.67962276+0.32037724j, 0.35924552+0.j ]]
#constellation_points = [[ -1,1],[1j,-1j]] # addition
constellation_points = [[1+1j,1-1j], [1,-1]] # multiplication
#constellation_points = [[ 0.9999727 +0.00739252j,  0.0061883 +0.9961329j , 0.03276862-0.99794596j, -0.99756783+0.02455366j],[0.93617743-0.3515279j , 0.47354087-0.18245742j, 0.9338867 +0.35331026j, 0.48145026+0.19996208j]]
precompensate=True

# symbol time and number of symbols    
t_symb = 3.2*1e-10
n_symb = 100


# Fiber and dispersion parameters
alpha = 0                                                   # 10**(0/10) # Attenuation approx 0.2 [1/km] 
lam = 1550e-9                                               # wavelength, [nm]
D = 20                                                        # [ps/nm/km]
beta2 = - (D * np.square(lam)) / (2 * np.pi * 3e8) * 1e-3     # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
Ld = (t_symb)**2/np.abs(beta2)
L = np.array([0,10,50,100])                                      # propagation distance in km
 

n_up = 37         # samples per symbol
syms_per_filt = 6  # symbols per filter (plus minus in both directions)
t_samp = t_symb/n_up
fa = 1/(t_symb) * n_up                                            # 10GBit/s

K_filt = 2 * syms_per_filt * n_up + 1         # length of the fir filter


# parameters for frequency regime
N_fft = 512
Omega = np.linspace( -np.pi, np.pi, N_fft)
f_vec = Omega / ( 2 * np.pi * t_symb / n_up )

# get pulses
sinc = np.sinc(np.linspace(-syms_per_filt,syms_per_filt, K_filt))
sinc /= np.linalg.norm( sinc )
#sinc = sinc*np.max(sinc)  # get the same pulse amplitude than for multiplication  

rect = np.append( np.ones( n_up ), np.zeros( len( sinc ) - n_up ) )
rect /= np.linalg.norm( rect )
rect = np.roll(rect,int((len(rect)-n_up)/2))
#rect = rect*np.max(rect) # get the same pulse amplitude than for multiplication  

t = np.linspace(-syms_per_filt,syms_per_filt, K_filt)
gauss = np.exp(-2.5*(np.linspace(-syms_per_filt,syms_per_filt, K_filt)**2))
gauss /=np.linalg.norm( gauss)
#gauss = gauss*np.max(gauss) # get the same pulse amplitude than for multiplication  

# get pulse spectra
RECT_PSD = np.abs( np.fft.fftshift( np.fft.fft( rect, N_fft ) ) )**2
RECT_PSD /= n_up

SINC_PSD = np.abs( np.fft.fftshift( np.fft.fft( sinc, N_fft ) ) )**2
SINC_PSD /= n_up

GAUSS_PSD = np.abs( np.fft.fftshift( np.fft.fft( gauss, N_fft ) ) )**2
GAUSS_PSD /= n_up

# number of realizations along which to average the psd estimate
n_real = 20

# initialize two-dimensional field for collecting several realizations along which to average 
S_rect = np.zeros( (n_real, N_fft ), dtype=complex )
S_sinc = np.zeros( (n_real, N_fft ), dtype=complex )
S_gauss = np.zeros( (n_real, N_fft ), dtype=complex )

# loop for multiple realizations in order to improve spectral estimation
s_rect = np.zeros((len(L),len(rect)+n_symb * n_up-1), dtype=complex)
s_sinc = np.zeros((len(L),len(sinc)+n_symb * n_up-1), dtype=complex)
s_gauss = np.zeros((len(L),len(gauss)+n_symb * n_up-1), dtype=complex)
for k in range( n_real ):
    for num in range(len(M)):
        # generate random binary vector and 
        # modulate the specified modulation scheme
        data = np.random.randint( M[num], size = n_symb )
        const = constellation_points[num]
        s = [ const[ d ] for d in data ]
        
        # apply RECTANGULAR filtering/pulse-shaping
        s_up_rect = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_rect[ : : n_up ] = s

        # apply Sinc filtering/pulse-shaping
        s_up_sinc = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_sinc[ : : n_up ] = s

        # apply Gauss filtering/pulse-shaping
        s_up_gauss = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_gauss[ : : n_up ] = s


        for value in range(len(L)):
            if num==0:
                s_rect[value,:] = np.convolve( rect, s_up_rect)
                s_sinc[value,:] = np.convolve( sinc, s_up_sinc)
                s_gauss[value,:] = np.convolve( gauss, s_up_gauss)
                if precompensate==True:
                    s_rect[value,:] = cd(s_rect[value,:],2*L[value],-D,fa,lam,alpha)
                    s_sinc[value,:] = cd(s_sinc[value,:],2*L[value],-D,fa,lam,alpha)
                    s_gauss[value,:] = cd(s_gauss[value,:],2*L[value],-D,fa,lam,alpha)

            else:
                # precompensation
                if precompensate==True:
                    ps_rect = cd(np.convolve( rect, s_up_rect),L[value],-D,fa,lam,alpha)
                    ps_sinc = cd(np.convolve( sinc, s_up_rect),L[value],-D,fa,lam,alpha)
                    ps_gauss = cd(np.convolve( gauss, s_up_rect),L[value],-D,fa,lam,alpha)

                    s_rect[value,:] = s_rect[value,:] * ps_rect
                    s_sinc[value,:] = s_sinc[value,:] * ps_sinc
                    s_gauss[value,:] = s_gauss[value,:] * ps_gauss
                else:
                    s_rect[value,:] = s_rect[value,:] * np.convolve( rect, s_up_rect)
                    s_sinc[value,:] = s_sinc[value,:] * np.convolve( sinc, s_up_rect)
                    s_gauss[value,:] = s_gauss[value,:] * np.convolve( gauss, s_up_rect)

                
        
            s_rect[value,:] = cd(s_rect[value,:],L[value],D,fa,lam,alpha)
            s_sinc[value,:] = cd(s_sinc[value,:],L[value],D,fa,lam,alpha)
            s_gauss[value,:] = cd(s_gauss[value,:],L[value],D,fa,lam,alpha)



        
    # get spectrum using Bartlett method
    # S_rc[k, :] = np.fft.fftshift( np.fft.fft( s_rc, N_fft ) )
    # S_rect[k, :] = np.fft.fftshift( np.fft.fft( s_rect, N_fft ) )
    # S_sinc[k, :] = np.fft.fftshift( np.fft.fft(s_sinc, N_fft))
    # S_gauss[k, :] = np.fft.fftshift( np.fft.fft(s_gauss, N_fft))
        



# average along realizations
# RC_PSD_sim = np.average( np.abs( S_rc )**2, axis=0 )
# RC_PSD_sim /= np.max( RC_PSD_sim )

# RECT_PSD_sim = np.average( np.abs( S_rect )**2, axis=0 ) 
# RECT_PSD_sim /= np.max( RECT_PSD_sim )

# SINC_PSD_sim = np.average( np.abs( S_sinc )**2, axis=0 ) 
# SINC_PSD_sim /= np.max( SINC_PSD_sim )

# GAUSS_PSD_sim = np.average( np.abs( S_gauss )**2, axis=0 ) 
# GAUSS_PSD_sim /= np.max( GAUSS_PSD_sim )


fig1, ax1 = plt.subplots(len(L),2, figsize=(6,8))
t = np.arange( np.size( np.real(s_rect[0,10*n_up:30*n_up]))) * t_symb / n_up

for val in range(len(L)):
    ax1[val,0].plot(t, np.real(s_rect[val,10*n_up:30*n_up]), label=r"rect"+str(L[val]))
    ax1[val,1].plot(t, np.imag(s_rect[val,10*n_up:30*n_up]), label=r"rect"+str(L[val]))

    ax1[val,0].plot(t, np.real(s_sinc[val,10*n_up:30*n_up]), label=r"sinc"+str(L[val]))
    ax1[val,1].plot(t, np.imag(s_sinc[val,10*n_up:30*n_up]), label=r"sinc"+str(L[val]))

    ax1[val,0].plot(t, np.real(s_gauss[val,10*n_up:30*n_up]), label=r"gauss"+str(L[val]))
    ax1[val,1].plot(t, np.imag(s_gauss[val,10*n_up:30*n_up]), label=r"gauss"+str(L[val]))

    ax1[val,0].set_ylim(-0.05,0.05)
    ax1[val,1].set_ylim(-0.05,0.05)
    #plt.legend(loc='upper left')
    ax1[val,0].set_title('L = '+str(2*L[val])+' km')
plt.tight_layout()
plt.savefig("dispersion.pdf")

# Eye Diagram
t=np.arange(n_up)*t_symb/n_up

def plot_eye(eye, station, num,L,figure=None):
    heatmap = np.vstack([np.histogram(eye[:,j], bins = np.linspace(-2,2,100))[0]/eye.shape[0] for j in np.arange(eye.shape[1])]).T
    
    heatmap_cum = np.zeros_like(heatmap)
    ps = np.concatenate((np.logspace(-3,-1,3), [0.25,0.5,0.75]))
    for p in ps:
        levels = -np.sort(-heatmap, axis = 0)
        a = np.argmax(np.cumsum(levels, axis = 0) >= 1-p, axis = 0)
        cut = levels[a, np.arange(levels.shape[1])]
        heatmap_cum[heatmap >= cut] = p
    
    figure = plt.figure("Eyediagram",figsize = (6,8), facecolor = 'w')
    time = (np.arange(eye.shape[1])-eye.shape[1]//2)
    time = time/np.max(time)*(2*3.2*1e-10)

    ax1 = figure.add_subplot(len(L),2,2*num+1)
    #plt.subplot(121)
    ax1.plot(time, np.real(eye.T), color = 'C0', alpha = 0.4)
    plt.title(station)
    ax1.set_ylabel(r'$\Re\{s(t)\}$')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylim(-0.05,0.05)

    ax1 = figure.add_subplot(len(L),2,2*num+2)
    ax1.plot(time, np.imag(eye.T), color = 'C0', alpha = 0.4)
    #plt.title(station)
    ax1.set_ylabel(r'$\Im\{s(t)\}$')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylim(-0.05,0.05)
    return figure
    #plt.tight_layout()
    #plt.savefig(f'eye_{station}.pdf')
    
    # plt.figure(figsize = (10,6), facecolor = 'w')
    # plt.imshow(heatmap, extent = [time[0],time[1],-2,2], aspect = 'auto', interpolation = 'bicubic')
    
    # plt.figure(figsize = (12,6), facecolor = 'w')
    # plt.contourf(heatmap_cum, levels = ps, cmap = 'viridis')
    # plt.colorbar()



#plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'sinc')

for l in range(len(L)):
    eye_sig = s_sinc[l,n_up*(syms_per_filt+2):len(s_sinc[l,:])-n_up*(syms_per_filt+2)]
    if l==0:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L)
    else:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L,fig)
plt.tight_layout()
plt.savefig(f'eye_sinc.pdf')
plt.close(fig)

for l in range(len(L)):
    eye_sig = s_rect[l,n_up*(syms_per_filt-2):len(s_sinc[l,:])-n_up*(syms_per_filt+4)]
    if l==0:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L)
    else:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L,fig)
plt.tight_layout()
plt.savefig(f'eye_rect.pdf')
plt.close(fig)

for l in range(len(L)):
    eye_sig = s_gauss[l,n_up*(syms_per_filt):len(s_sinc[l,:])-n_up*(syms_per_filt+2)]
    if l==0:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L)
    else:
        fig = plot_eye(np.roll(eye_sig,0).reshape(int(len(eye_sig)/(n_up*2)),int(n_up*2)),'L = '+str(2*L[l])+' km',l,L,fig)
plt.tight_layout()
plt.savefig(f'eye_gauss.pdf')


cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
# plot received constellation
plt.figure("constellation",figsize=(3,3))
for x in range(n_symb):
    plt.scatter(np.real(s_rect[0,x*n_up]), np.imag(s_rect[0,x*n_up]),color=color_list[0], alpha=0.8)
    plt.scatter(np.real(s_rect[1,x*n_up]), np.imag(s_rect[2,x*n_up]),color=color_list[2], alpha=0.8)
    plt.scatter(np.real(s_rect[2,x*n_up]), np.imag(s_rect[2,x*n_up]),color=color_list[4], alpha=0.8)
plt.grid()
plt.xlabel(r'$\Re\{s(t)\}$')
plt.ylabel(r'$\Im\{s(t)\}$')
plt.legend(['L = 0 km','L = 20 km','L = 100 km'], loc='lower right')
plt.tight_layout()
plt.savefig(f'dispersion_const_prec.pdf')