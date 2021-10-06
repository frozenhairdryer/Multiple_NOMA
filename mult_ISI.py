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

########################
# find impulse response of an RC filter
########################
def get_rc_ir(K, n_up, t_symbol, beta):
    
    ''' 
    Determines coefficients of an RC filter 
    
    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used 
    
    NOTE: Length of the IR has to be an odd number
    
    IN: length of IR, upsampling factor, symbol time, roll-off factor
    OUT: filter coefficients
    '''

    # check that IR length is odd
    assert K % 2 == 1, 'Length of the impulse response should be an odd number'
    
    # map zero r to close-to-zero
    if beta == 0:
        beta = 1e-32


    # initialize output length and sample time
    rc = np.zeros( K )
    t_sample = t_symbol / n_up
    
    
    # time indices and sampled time
    k_steps = np.arange( -(K-1) / 2.0, (K-1) / 2.0 + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):
        
        if t_steps[k] == 0:
            rc[ k ] = 1. / t_symbol
            
        elif np.abs( t_steps[k] ) == t_symbol / ( 2.0 * beta ):
            rc[ k ] = beta / ( 2.0 * t_symbol ) * np.sin( np.pi / ( 2.0 * beta ) )
            
        else:
            rc[ k ] = np.sin( np.pi * t_steps[k] / t_symbol ) / np.pi / t_steps[k] \
                * np.cos( beta * np.pi * t_steps[k] / t_symbol ) \
                / ( 1.0 - ( 2.0 * beta * t_steps[k] / t_symbol )**2 )
 
    return rc

# modulation scheme and constellation points
#M = [4,4]
M=[2,2]
#mradius=1/3*np.sqrt(2)
#c2 = (1+mradius*np.array([1,-1j,1j,-1]))/(1+mradius)
#constellation_points = [[ -1, 1, 1j,-1j ],[1.+0.j, 0.67962276-0.32037724j,0.67962276+0.32037724j, 0.35924552+0.j ]]
#constellation_points = [[ -1, 1],[1j,1]]
#constellation_points = [[ -1,1],[1j,-1j]] # addition
constellation_points = [[1,-1], [1+1j,1-1j]] # multiplication


# symbol time and number of symbols    
t_symb = 1.0
n_symb = 100
 

# parameters of the RRC filter
beta = .33
n_up = 15         # samples per symbol
syms_per_filt = 6  # symbols per filter (plus minus in both directions)

K_filt = 2 * syms_per_filt * n_up + 1         # length of the fir filter


# parameters for frequency regime
N_fft = 512
Omega = np.linspace( -np.pi, np.pi, N_fft)
f_vec = Omega / ( 2 * np.pi * t_symb / n_up )

# get RC pulse and rectangular pulse,
# both being normalized to energy 1
rc = get_rc_ir( K_filt, n_up, t_symb, beta )
rc /= np.linalg.norm( rc ) 

rect = np.append( np.ones( n_up ), np.zeros( len( rc ) - n_up ) )
rect /= np.linalg.norm( rect )
rect = np.roll(rect,int((len(rect)-n_up)/2))
#rect = rect*np.max(rect)  # get the same pulse amplitude than for multiplication 

sinc = np.sinc(np.linspace(-syms_per_filt,syms_per_filt, K_filt))
sinc /= np.linalg.norm( sinc )
#sinc = sinc*np.max(sinc)  # get the same pulse amplitude than for multiplication 

t = np.linspace(-syms_per_filt,syms_per_filt, K_filt)
gauss = np.exp(-2.5*(np.linspace(-syms_per_filt,syms_per_filt, K_filt)**2))
gauss /=np.linalg.norm( gauss)
#gauss = gauss*np.max(gauss)  # get the same pulse amplitude than for multiplication 

# get pulse spectra
RC_PSD = np.abs( np.fft.fftshift( np.fft.fft( rc, N_fft ) ) )**2
RC_PSD /= n_up

RECT_PSD = np.abs( np.fft.fftshift( np.fft.fft( rect, N_fft ) ) )**2
RECT_PSD /= n_up

SINC_PSD = np.abs( np.fft.fftshift( np.fft.fft( sinc, N_fft ) ) )**2
SINC_PSD /= n_up

GAUSS_PSD = np.abs( np.fft.fftshift( np.fft.fft( gauss, N_fft ) ) )**2
GAUSS_PSD /= n_up

# number of realizations along which to average the psd estimate
n_real = 20

# initialize two-dimensional field for collecting several realizations along which to average 
S_rc = np.zeros( (n_real, N_fft ), dtype=complex ) 
S_rect = np.zeros( (n_real, N_fft ), dtype=complex )
S_sinc = np.zeros( (n_real, N_fft ), dtype=complex )
S_gauss = np.zeros( (n_real, N_fft ), dtype=complex )

# loop for multiple realizations in order to improve spectral estimation
for k in range( n_real ):
    for num in range(len(M)):
        # generate random binary vector and 
        # modulate the specified modulation scheme
        data = np.random.randint( M[num], size = n_symb )
        const = constellation_points[num]
        s = [ const[ d ] for d in data ]

        # apply RC filtering/pulse-shaping
        s_up_rc = np.zeros( n_symb * n_up , dtype=complex)        
        s_up_rc[ : : n_up ] = s
        
        # apply RECTANGULAR filtering/pulse-shaping
        s_up_rect = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_rect[ : : n_up ] = s

        # apply Sinc filtering/pulse-shaping
        s_up_sinc = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_sinc[ : : n_up ] = s

        # apply Gauss filtering/pulse-shaping
        s_up_gauss = np.zeros( n_symb * n_up , dtype=complex)      
        s_up_gauss[ : : n_up ] = s
                
        if num==0:
            s_rect = np.convolve( rect, s_up_rect)
            s_rc = np.convolve( rc, s_up_rc)
            s_sinc = np.convolve( sinc, s_up_sinc)
            s_gauss = np.convolve( gauss, s_up_gauss)
        else:
            s_rect = s_rect * np.convolve( rect, s_up_rect)
            s_rc = s_rc * np.convolve( rc, s_up_rc)
            s_sinc = s_sinc * np.convolve( sinc, s_up_sinc)
            s_gauss = s_gauss * np.convolve( gauss, s_up_gauss)


        
    # get spectrum using Bartlett method
    S_rc[k, :] = np.fft.fftshift( np.fft.fft( s_rc, N_fft ) )
    S_rect[k, :] = np.fft.fftshift( np.fft.fft( s_rect, N_fft ) )
    S_sinc[k, :] = np.fft.fftshift( np.fft.fft(s_sinc, N_fft))
    S_gauss[k, :] = np.fft.fftshift( np.fft.fft(s_gauss, N_fft))
        



# average along realizations
RC_PSD_sim = np.average( np.abs( S_rc )**2, axis=0 )
RC_PSD_sim /= np.max( RC_PSD_sim )

RECT_PSD_sim = np.average( np.abs( S_rect )**2, axis=0 ) 
RECT_PSD_sim /= np.max( RECT_PSD_sim )

SINC_PSD_sim = np.average( np.abs( S_sinc )**2, axis=0 ) 
SINC_PSD_sim /= np.max( SINC_PSD_sim )

GAUSS_PSD_sim = np.average( np.abs( S_gauss )**2, axis=0 ) 
GAUSS_PSD_sim /= np.max( GAUSS_PSD_sim )

plt.figure("mult_ISI", figsize=[6,8])
plt.subplot(321)

#plt.plot( np.arange( np.size( rc ) ) * t_symb / n_up, rc, linewidth=2.0, label='RC' )
plt.plot( np.arange( np.size( rect ) ) * t_symb / n_up, rect, linewidth=1.0, label=r'Rect' )
plt.plot( np.arange( np.size( sinc ) ) * t_symb / n_up, sinc, linewidth=1.0, label=r'Sinc' )
plt.plot( np.arange( np.size( gauss ) ) * t_symb / n_up, gauss, linewidth=1.0, label=r'Gauss' )

#plt.ylim( (-.1, 1.1 ) ) 
plt.grid( True )
plt.legend( loc='upper left' )    
#plt.title( '$g(t), s(t)$' )
plt.ylabel('$g(t)$')
#plt.xlim(-4,4)


# plt.subplot(322)
# np.seterr(divide='ignore') # ignore warning for logarithm of 0

# plt.plot( f_vec, 10*np.log10( RECT_PSD ), linewidth=2.0, label=r'Rect theory' )
# plt.plot( f_vec, 10*np.log10( SINC_PSD ), linewidth=2.0, label=r'SINC theory' ) 
# plt.plot( f_vec, 10*np.log10( GAUSS_PSD ), linewidth=2.0, label=r'GAUSS theory' )  
# np.seterr(divide='warn') # enable warning for logarithm of 0

# plt.grid( True )    
# plt.legend( loc='upper right' )    
# plt.ylabel( '$|G(f)|^2$' )   
# plt.ylim( (-60, 10 ) )
plt.subplot(322)
np.seterr(divide='ignore') # ignore warning for logarithm of 0
#plt.plot( f_vec, 10*np.log10( RC_PSD_sim ), linewidth=2.0, label='RC' )
plt.plot( f_vec, 10*np.log10( RECT_PSD_sim ), linewidth=1.0, label=r'Rect' ) 
plt.plot( f_vec, 10*np.log10( SINC_PSD_sim ), linewidth=1.0, label=r'Sinc' ) 
#plt.plot( f_vec, 10*np.log10( GAUSS_PSD_sim ), linewidth=1.0, label=r'Gauss' ) 

#plt.plot( f_vec, 10*np.log10( RECT_PSD ), linewidth=2.0, label=r'Rect theory' )
#plt.plot( f_vec, 10*np.log10( SINC_PSD ), linewidth=2.0, label=r'SINC theory' ) 
#plt.plot( f_vec, 10*np.log10( GAUSS_PSD ), linewidth=2.0, label=r'Gauss theory' )  
np.seterr(divide='warn') # enable warning for logarithm of 0

plt.grid(True); 
plt.xlabel('$fT$');  
plt.ylabel( r'$|S(f)|^2$ (dB)' )   
plt.legend(loc='upper left')
plt.ylim( (-60, 10 ) )
plt.xlim(-4,4)


plt.subplot(312)

#plt.plot( np.arange( np.size( np.real(s_rc[:20*n_up]))) * t_symb / n_up, np.real(s_rc[:20*n_up]), linewidth=2.0, label='RC' )
plt.plot( np.arange( np.size( np.real(s_rect[:20*n_up]))) * t_symb / n_up, np.real(s_rect[:20*n_up]), linewidth=1.0, label=r'Rect') 
plt.plot( np.arange( np.size( np.real(s_sinc[:20*n_up]))) * t_symb / n_up, np.real(s_sinc[:20*n_up]), linewidth=1.0, label=r'Sinc')  
#plt.plot( np.arange( np.size( np.real(s_gauss[:20*n_up]))) * t_symb / n_up, np.real(s_gauss[:20*n_up]), linewidth=1.0, label=r'Gauss')  
#plt.plot( np.arange( np.size( s_up_rc[:20*n_up])) * t_symb / n_up, s_up_rc[:20*n_up], 'o', linewidth=2.0, label='Syms' )

#plt.ylim( (-1, 1 ) )
plt.grid(True)    
plt.legend(loc='upper left')    
plt.xlabel('$t/T$')
plt.ylabel(r'$\Re\{s(t)\}$')




plt.subplot(313)

#plt.plot( np.arange( np.size( np.imag(s_rc[:20*n_up]))) * t_symb / n_up, np.imag(s_rc[:20*n_up]), linewidth=2.0, label='RC' )
plt.plot( np.arange( np.size( np.imag(s_rect[:20*n_up]))) * t_symb / n_up, np.imag(s_rect[:20*n_up]), linewidth=1.0, label=r'Rect' )  
plt.plot( np.arange( np.size( np.imag(s_sinc[:20*n_up]))) * t_symb / n_up, np.imag(s_sinc[:20*n_up]), linewidth=1.0, label=r'Sinc' )   
#plt.plot( np.arange( np.size( np.imag(s_gauss[:20*n_up]))) * t_symb / n_up, np.imag(s_gauss[:20*n_up]), linewidth=1.0, label=r'Gauss' )     
#plt.plot( np.arange( np.size( s_up_rc[:20*n_up])) * t_symb / n_up, s_up_rc[:20*n_up], 'o', linewidth=2.0, label='Syms' )

#plt.ylim( (-1, 1 ) )
plt.grid(True)    
plt.legend(loc='upper left')    
plt.xlabel('$t/T$')
plt.ylabel(r'$\Im \{s(t)\}$')

plt.tight_layout()
plt.savefig('mult_ISI.pdf',bbox_inches='tight')

plt.figure()
np.seterr(divide='ignore') # ignore warning for logarithm of 0

plt.plot( f_vec, 10*np.log10(RECT_PSD_sim) , linewidth=1.0, label=r'Rect' ) 
plt.plot( f_vec, 10*np.log10(SINC_PSD_sim ), linewidth=1.0, label=r'Sinc' ) 
#plt.plot( f_vec, 10*np.log10(GAUSS_PSD_sim), linewidth=1.0, label=r'Gauss' ) 

#plt.plot( f_vec, 10*np.log10( RECT_PSD ), linewidth=2.0, label=r'Rect theory' )
#plt.plot( f_vec, 10*np.log10( SINC_PSD ), linewidth=2.0, label=r'SINC theory' ) 
#plt.plot( f_vec, 10*np.log10( GAUSS_PSD ), linewidth=2.0, label=r'Gauss theory' )  
np.seterr(divide='warn') # enable warning for logarithm of 0

plt.grid(True); 
plt.xlabel('$fT$');  
plt.ylabel( r'$|S(f)|^2$ (dB)' )   
plt.legend(loc='upper left')
plt.ylim( (0, 1 ) )
plt.xlim(-4,4)
np.seterr(divide='warn') # enable warning for logarithm of 0


plt.savefig('original_pulses.pdf')

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
# plot received constellation
plt.figure("constellation",figsize=(3,3))
for x in range(n_symb):
    plt.scatter(np.real(s_gauss[(x+syms_per_filt)*n_up]), np.imag(s_gauss[(x+syms_per_filt)*n_up]),color=color_list[0], alpha=0.8)
plt.grid()
plt.xlabel(r'$\Re\{s(t)\}$')
plt.ylabel(r'$\Im\{s(t)\}$')
#plt.legend(['L = 0 km','L = 20 km','L = 100 km'], loc='lower right')
plt.tight_layout()
plt.savefig(f'ISI_const.pdf')