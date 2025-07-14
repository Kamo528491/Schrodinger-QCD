import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
import scipy.signal
import matplotlib
from scipy.integrate import simpson
from scipy.optimize import curve_fit

hbar=1                                                       
N =  3000                                                   # Discretization level
r_min = 0.0000001
r_max = 5
l_max = 2
nr_max = 5
at = 1/18                                                   # [GeV-1]
r = np.linspace(r_min, r_max, N)                            # [GeV-1]
h = r[1]-r[0]                                               # Step 
b = 0.212                                                   # [GeV²]
m = 172.4                                                   # [GeV] (top quark mass)
mu = m/2                                                    # [GeV] (reduced mass)
alpha = 0.189*np.ones(r.shape)                              # αs for Coulomb potential 
alpha_cor = 0.211                                           # αs for Cornell potential                                              
Ca = 3
Tf = 1/2
nf = 5
Cf = 4/3
a1 = 31/9*Ca -20/9*Tf*nf
gammaE = 0.577
Beta0 = 11/3*Ca - 4/3*Tf*nf
aperys_cte = 1.20305                                        
a2 = (4343/162 + 4*np.pi**2 - np.pi**4/4 + 22/3*aperys_cte)*Ca**2 - (1798/81 + 56/3*aperys_cte)*Ca*Tf*nf - (55/3-16*aperys_cte)*Cf*Tf*nf + (20/9*Tf*nf)**2
Beta1 = 34*Ca**2/3 - 20*Ca*Tf*nf/3 - 4*Cf*Tf*nf
r0 = 0.507                                                  # [GeV-1] (0.1fm) 
Lambda = 0.24                                               # [GeV]

def alpha_s(r):
    """Strong coupling constant"""
    mu = 1/r
    term0 = 4*np.pi/(Beta0*np.log(mu**2/Lambda**2))
    term1 = Beta1*np.log(np.log(mu**2/Lambda**2))/(Beta0**2*np.log(mu**2/Lambda**2))

    return term0*(1-term1)                                                              # NaN and infs values occur beyond r0 where we use fixed alpha value 
                                                                                        # (cornell potential) and don't cause any trouble 

def k0(alpha):
    """1st order potential term"""
    return alpha                                              

def k1(alpha):
    """2nd order potential term"""
    return alpha*(1 + alpha/(4*np.pi)*(a1 + gammaE*Beta0))    

def k2(alpha):
    """3rd order potential term"""
    return alpha*(1 + alpha/(4*np.pi)*(a1 + gammaE*Beta0) + (alpha/(4*np.pi))**2*(a2 + (np.pi**2/3 + 4*gammaE**2)*Beta0**2 + 2*gammaE*(2*a1*Beta0 + Beta1)))  

def levels(l):
    """Returns the energy level given the value of l"""
    if  l == 0 :
        return "S"
    elif l == 1:
        return "P"
    elif l == 2 :
        return "D"

# Defining the spin dependent potential for each state  

delta = scipy.signal.unit_impulse(r.shape, idx = 1)                             # Dirac Delta             

def V0(alpha, i):
    return 0*i

def Veta(alpha, i):
    return -6*alpha[i]/(9*r[i]**2*m**2)*delta[i]

def Vh(alpha, i):
    return -6*alpha[i]/(9*r[i]**2*m**2)*delta[i]

def Vpsi(alpha, i):
    return 2*alpha[i]/(9*r[i]**2*m**2)*delta[i]    

def Vchi0(alpha, i):
    if r[i] < 0.005:
        return 0                                                                                 # Avoiding divergence in r = 0
    else:
        return 2*alpha[i]/(9*r[i]**2*m**2)*delta[i] + 1/(m**2)*((b/r[i] - 8*alpha[i]/(r[i])**3))

def Vchi1(alpha, i):
    return 2*alpha[i]/(9*r[i]**2*m**2)*delta[i]  + 1/(m**2)*((-b/(2*r[i]) + 2*alpha[i]/(r[i])**3) + 2*alpha[i]/(r[i]**3*3))

def Vchi2(alpha, i):
    return 2*alpha[i]/(9*r[i]**2*m**2)*delta[i]  + 1/(m**2)*((-b/(2*r[i]) + 2*alpha[i]/(r[i])**3) - 4*alpha[i]/(r[i]**3*30))      

def solve(l,k,Vadd,alpha):
    """Returns the eigenvalues & eigenvectors of the hamiltonian given the value 
    of l, chosen potential approximation k and the spin dependent potential"""
    
    V = np.ones(r.shape)

    for i in range(len(r)):                                                                      
        if r[i] < r0 :
            V[i] = hbar**2/(2*mu)*l*(l+1)/(r[i]**2) - k(alpha)[i]*Cf/r[i] + Vadd(alpha, i)       # Coulomb + spin dependant potential
        else : 
            V[i] = hbar**2/(2*mu)*l*(l+1)/(r[i]**2) - alpha_cor*Cf/r[i]  + b*r[i]                # Cornell potential

    Mdd = 1/h**2*(np.diag(np.ones(N-1),-1) + np.diag(-2*np.ones(N),0) + np.diag(np.ones(N-1),1)) # Laplacian operator for finite element method
    H = - hbar**2/(2*mu)*Mdd + np.diag(V)                                                        
    E,psi = scl.eigh(H[1:-1, 1:-1])                                                              # Getting the eigenvalues and eigenvectors

    return E, psi

def print_results(E, l, S, J):
    """Prints the energy values given the quantum numbers"""
    for i in range(nr_max):
        print(f"{i+1}{levels(l)}, S = {S}, J = {J} : ", "E_bind =", round(E[i],3), "E_tot =", round(E[i] + 2*m,3), "GeV") 

    print('--------------------------------------------------')

def get_results(k,alpha):
    """Returns the energy values given the chosen potential approximation"""

    Eeta, u_eta = solve(0,k, Veta, alpha)
    Eh, u_h = solve(1,k, Vh,alpha)
    Epsi,psi_psi = solve(0,k, Vpsi,alpha)
    Echi0,psi_chi0 = solve(1,k, Vchi0,alpha)
    Echi1,psi_chi1 = solve(1,k, Vchi1,alpha)
    Echi2,psi_chi2 = solve(1,k, Vchi2,alpha)

    print_results(Eeta, 0, 0, 0)                                                # eta_t : L = 0 (S term), S = 0, J = 0
    print_results(Eh, 1, 0, 1)                                                  # h_t : L = 1 (P term), S = 0, J = 1
    print_results(Epsi, 0, 1, 1)                                                # psi_t : L = 0 (S term), S = 1, J = 1
    print_results(Echi0, 1, 1, 0)                                               # chi_t0 : L = 1 (P term), S = 1, J = 0
    print_results(Echi1, 1, 1, 1)                                               # chi_t1 : L = 1 (P term), S = 1, J = 1
    print_results(Echi2, 1, 1, 2)                                               # chi_t2 : L = 1 (P term), S = 1, J = 2

    print('Difference between 1s, 1p, S = 0 :', round((Eeta[1] - Eh[0])*1000,3), 'MeV')
    print('Difference between 1s states S = 0,1 :', round(Eeta[0] - Epsi[0],6)*1000, 'MeV')
    print('Difference between 1p states, h, chi0 :', round(Eh[0]- Echi0[0],3)*1000, 'MeV')
    print('Difference between 1p states, h, chi1 :', round(Eh[0]- Echi1[0],3)*1000, 'MeV')
    print('Difference between 1p states, h, chi2 :', round(Eh[0]- Echi2[0],3)*1000, 'MeV')

def plot_results(k, alpha):
    """Plots the energy levels given the chosen potential approximation"""

    Eeta, psi_eta = solve(0,k, Veta, alpha)
    Eh, u_h = solve(1,k, Vh, alpha)
    Epsi,psi_psi = solve(0,k, Vpsi, alpha)
    Echi0,psi_chi0 = solve(1,k, Vchi0, alpha)
    Echi1,psi_chi1 = solve(1,k, Vchi1, alpha)
    Echi2,psi_chi2 = solve(1,k, Vchi2, alpha)

    plt.figure(figsize = (9,7))

    for i in range(nr_max):                                                     
        plt.axhline(Eeta[i]+2*m, xmin =  0/6, xmax = 0.1 + 0/6, color = "indigo", alpha = 1/4 + np.arctan(i/nr_max), lw = 2, label = f"{i+1}{levels(0)}")

    for i in range(nr_max):
        plt.axhline(Epsi[i]+2*m, xmin =  1/6, xmax = 0.1 + 1/6, color = "indigo", alpha = 1/4 + np.arctan(i/nr_max), lw = 2)
        plt.axhline(Eh[i]+2*m, xmin =  2/6, xmax = 0.1 + 2/6, color = "darkred", alpha = 1/4 + np.arctan(i/nr_max), lw = 2,  label = f"{i+1}{levels(1)}")
        plt.axhline(Echi0[i]+2*m, xmin = 3/6, xmax = 0.1 + 3/6, color = "darkred", alpha = 1/4 + np.arctan(i/nr_max), lw = 2)
        plt.axhline(Echi1[i]+2*m, xmin =  4/6, xmax = 0.1 + 4/6, color = "darkred", alpha = 1/4 + np.arctan(i/nr_max), lw = 2)
        plt.axhline(Echi2[i]+2*m, xmin =  5/6, xmax = 0.1 + 5/6, color = "darkred", alpha = 1/4 + np.arctan(i/nr_max), lw = 2)
        
    
    plt.title("$\\Lambda$ = " + str(Lambda) + " GeV, $m_T$ = 172.4 GeV")
    plt.ylabel('Energy [GeV]')
    plt.ylim(341,345.5)
    plt.axhline(2*m, color = 'black', label = f"$2m_T$", linestyle = '--')
    plt.xticks(ticks = [0.05 + i/6 for i in range(6)], labels = ["$^1S_0, \\eta_t$", "$^3S_1, \\psi_t$ / $\\theta_t$", "$^1P_1, h_t$", "$^3P_0, \\chi_{t0}$", "$^3P_1, \\chi_{t1}$", "$^3P_2, \\chi_{t2}$"] )
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.grid()
    #plt.savefig("Plot.pdf", format="pdf", bbox_inches="tight")                          # (saves as a pdf)
    plt.show()

def get_wavefunctions(k, alpha):
    """Evaluates and normalizes the wave functions given the chosen potential approximation"""
    
    Eeta, u_eta = solve(0,k, V0, alpha)                                     # Solving for spin-independant potential
    Eh, u_h = solve(1,k, V0, alpha)
    ED, u_D = solve(2,k, V0, alpha)

    for i in range(nr_max):
        norm_eta = simpson(np.abs(u_eta[:,i])**2, x = r[1:-1], dx = h)      # Calculating ∫|Ψ|²dr3                               
        u_eta[:,i] = u_eta[:,i]/np.sqrt(norm_eta)                                     

        norm_h = simpson(np.abs(u_h[:,i])**2, x = r[1:-1], dx = h)     
        u_h[:,i] = u_h[:,i]/np.sqrt(norm_h)

        norm_D = simpson(np.abs(u_D[:,i])**2, x = r[1:-1], dx = h)    
        u_D[:,i] = u_D[:,i]/np.sqrt(norm_D)

    return u_eta, u_h, u_D

def R(u,i):
    return u[:,i]/(r[1:-1]) 

def model(x,a,b):
    return a*x + b
                                            
def R_extrapolation(u, r, degree = 2):
    """n degree extrapolation function"""
    fit, cov = np.polyfit(r[1:11], u[:10], degree, full = False, cov = True)               # First degree fitting polynomial y = ax + b
    return fit[degree], cov[degree:,degree][0]                                             # Returns R(0), the vertical intercept and variance (= diagonal coefficient from covariance matrix)

def adjust(u,r):
    """Linear fit function"""
    solutions, pcov = curve_fit(model, r[1:6], u[:5])                                      # Linear fit at the origin, returns R(0) and uncertainties 
    return solutions[1], pcov[1]

def diff(x,y):
    """Differentiation function (centered), Input : r, f(r)"""
    n = len(x)
    xoutput = np.zeros(n)
    youtput = np.zeros(n)
    
    xoutput[0] = x[0]
    youtput[0] = (y[1] - y[0]) / (x[1] - x[0])                          # First growth rate
    
    xoutput[n - 1] = x[n - 1]
    youtput[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])      # Last growth rate
    
    for i in range(n - 2):                                              
        xoutput[i + 1] = x[i + 1]
        youtput[i + 1] = (y[i + 2] - y[i]) / (x[i + 2] - x[i])          # Calculating centered growth rate successively
    return [xoutput, youtput]

def print_wavefunctions(u_eta, u_h, u_D):
    """Prints the values of the wave functions at the origin given the chosen potential approximation"""

    Rs2_extrapolated = np.ones(nr_max)                                  # List of |R(0)|² 
    Rs_extrapolated = np.ones(nr_max)                                   # List of R(0)
    Rs = np.ones(nr_max)                                                # List of R(0) without extrapolation
    std_Rs = np.ones(nr_max)                                            # List of uncertainties on extrapolated R(0) value

    for i in range(nr_max):                                                
        R0, unc = adjust(u_eta[:,i]/r[1:-1], r)                              
        Rs2_extrapolated[i] = round(R0**2*10**(-4),3)                                 
        Rs_extrapolated[i] = round(R0/at**(-3/2),3)
        Rs[i] = round(abs(R(u_eta,i)[0]/at**(-3/2)),3)                                
        std_Rs[i] = round(abs(unc[0])/at**(-3/2),6)                     # Saving standard deviation (square root from variance)            

    print("|Rns|^2 [x10^4 GeV3] : ", Rs2_extrapolated)                                      
    print("Rs(0)[at -3/2] : ", Rs_extrapolated)
    print('Uncertainties on Rs(0) :', std_Rs)
    #print(Rs)
    print('--------------------------------------------------------------')

    Rp2_extrapolated = np.ones(nr_max)
    Rp_extrapolated = np.ones(nr_max)
    Rp = np.ones(nr_max)
    std_Rp = np.ones(nr_max)
    
    for i in range(nr_max):
        r_diff, Rp_diff = diff(r[1:-1], R(u_h,i))                                    # Differentiating
        diff_R0, var = R_extrapolation(Rp_diff, r_diff,2)                            # Extrapolated Rp'(0) and variance                                          
        Rp2_extrapolated[i] = round(diff_R0**2*10**(-5),3)                          
        Rp_extrapolated[i] = round(diff_R0/at**(-5/2),3)
        Rp[i] = round(abs(Rp_diff[0]/at**(-5/2)),3)                                  
        std_Rp[i] = round(np.sqrt(var)/at**(-5/2),6)

    print("|Rnp'|^2 [x10^5 GeV5] : ", Rp2_extrapolated)                                      
    print("Rp'(0)[at -5/2] : ", Rp_extrapolated)
    print("Uncertainties on Rp'(0) :", std_Rp)
    #print(Rp)
    print('-------------------------------------------------------------')

    Rd2_extrapolated = np.ones(nr_max)
    Rd_extrapolated = np.ones(nr_max)
    Rd = np.ones(nr_max)
    std_Rd = np.ones(nr_max)
    
    for i in range(nr_max):
        r_diff, Rd_diff = diff(r[1:-1], R(u_D,i))                                    # Differentiating two times
        r_diff2, Rd_diff2 = diff(r_diff, Rd_diff)
        diff2_R0, var = R_extrapolation(Rd_diff2, r_diff2, 2)                        # Extrapolated Rd''(0) and variance                                         
        Rd2_extrapolated[i] = round(diff2_R0**2*10**(-5),3)                          
        Rd_extrapolated[i] = round(diff2_R0/at**(-7/2),3)
        Rd[i] = round(abs(Rd_diff[0]/at**(-7/2)),3)
        std_Rd[i] = round(np.sqrt(var)/at**(-7/2),6)

    print("|Rnd''|^2 [x10^6 GeV7] : ", Rd2_extrapolated)                                      
    print("Rd''(0)[at -7/2] : ", Rd_extrapolated)
    print("Uncertainties on Rd''(0) :", std_Rd)
    #print(Rd)
    print('-------------------------------------------------------------')

    return Rs_extrapolated, Rp_extrapolated, Rd_extrapolated

def plot_wavefunctions(u_eta, u_h, u_D, Rs_extrapolated, Rp_extrapolated, Rd_extrapolated):
    """Plotting the normalized wave functions given the chosen potential approximation"""

    plt.figure(figsize = (10,8))
    cmap = matplotlib.colormaps['viridis']

    plt.subplot(3,1,1)
    plt.title("$ ^1S_0, ^3S_1$, $\\alpha_s = 0.189$, $\\Lambda$ = 0.24 GeV, $m_T = 172.4$ GeV")
    for i in range(nr_max):
        Rs0 = np.array([Rs_extrapolated[i]*at**(-3/2)])
        func = np.concatenate((Rs0,  R(u_eta,i)))                                    # Adding R(r = 0) extrapolated term
        plt.plot(r[:-1], func, label = f"{i+1}S,P", color = cmap(i/(nr_max-1)))
    plt.xlabel('r [GeV$^{-1}$]')
    plt.ylabel("$R(r)$ [GeV$^{-3/2}$]")
    plt.xlim(0,0.75)
    plt.grid()
    plt.legend(loc = 'upper right')

    plt.subplot(3,1,2)
    plt.title("$ ^1P_1, ^3P_J$")
    for i in range(nr_max):
        plt.plot(r[1:-1], R(u_h,i), label = f"{i+1}P", color = cmap(i/(nr_max-1)))
    plt.xlabel('r [GeV$^{-1}$]')
    plt.ylabel("$R(r)$ [GeV$^{-3/2}$]")
    plt.xlim(0,2)
    plt.grid()

    plt.subplot(3,1,3)
    plt.title("$ ^1P_1', ^3P_J'$")
    for i in range(nr_max):
        r_diff, Rp_diff = diff(r[1:-1], R(u_h,i))
        Rp0 = np.array([Rp_extrapolated[i]*at**(-5/2)])
        func = np.concatenate((Rp0,  Rp_diff)) 
        plt.plot(r[:-1], func, label = f"{i+1}P", color = cmap(i/(nr_max-1)))
    plt.xlabel('r [GeV$^{-1}$]')
    plt.ylabel("$R'(r)$ [GeV$^{-5/2}$]")
    plt.xlim(0,0.75)
    plt.grid()

    plt.subplots_adjust(wspace=0, hspace=0.7)
    #plt.savefig("Wave_functions - constant α.pdf", format="pdf", bbox_inches="tight")                          # (saves as a pdf)
    plt.show()

get_results(k2, alpha_s(r))       
plot_results(k2, alpha_s(r))
u_eta, u_h, u_D = get_wavefunctions(k1, alpha)
Rs_extrapolated, Rp_extrapolated, Rd_extrapolated = print_wavefunctions(u_eta, u_h, u_D)
plot_wavefunctions(u_eta, u_h, u_D, Rs_extrapolated, Rp_extrapolated, Rd_extrapolated)