import numpy as np
import numba
@numba.jit(nopython=True)
def Response_Spectra(acc,  dt,  xi, period,  m,  gamma,  beta):
    Np = acc.size
    Nt = period.size

    w = 2.0*np.pi/period
    c = 2.0*xi*m*w
    k = m*w**2.0
    k1 = k + gamma*c/beta/dt + m/beta/dt/dt
    a = m/beta/dt + gamma*c/beta
    b = 0.5*m/beta + dt*(gamma*0.5/beta - 1.0)*c
    
    p = np.zeros(Np, dtype=numba.float64)
    dp = np.zeros(Np-1, dtype=numba.float64)
    #SD = np.zeros(Nt, dtype=numba.float64)
    #PSV = np.zeros(Nt, dtype=numba.float64)
    PSA = np.zeros(Nt, dtype=numba.float64)
    #SV = np.zeros(Nt, dtype=numba.float64)
    #SA = np.zeros(Nt, dtype=numba.float64)

    dp1 = np.zeros(Np-1, dtype=numba.float64)
    u = np.zeros((Nt, Np), dtype=numba.float64)
    du = np.zeros(Np-1, dtype=numba.float64)
    du1 = np.zeros(Np-1, dtype=numba.float64)
    du2 = np.zeros(Np-1, dtype=numba.float64)
    u1 = np.zeros(Np, dtype=numba.float64)
    u2 = np.zeros(Np, dtype=numba.float64)

    p=-m*acc        #always same so move outside loop

    for i_T in range(Nt):
        #p = -m*acc
        dp = np.diff(p)
        
        for i_s in range(Np-1):
            dp1[i_s] = dp[i_s] + a[i_T]*u1[i_s] + b[i_T]*u2[i_s]
            du[i_s] = dp1[i_s]/k1[i_T]
            du1[i_s] = gamma*du[i_s]/beta/dt - gamma*u1[i_s]/beta + dt*(1.0-0.5*gamma/beta)*u2[i_s]
            du2[i_s] = du[i_s]/beta/dt/dt - u1[i_s]/beta/dt - 0.5*u2[i_s]/beta
            u[i_T, i_s+1] = u[i_T, i_s] + du[i_s]
            u1[i_s+1] = u1[i_s] + du1[i_s]
            u2[i_s+1] = u2[i_s] + du2[i_s]
        
        #SD[i_T] = np.max(np.abs(u))
        #PSV[i_T] = SD[i_T]*w[i_T]
        #PSA[i_T] = PSV[i_T]*w[i_T]
        PSA[i_T] = np.max(np.abs(u[i_T, :])) * w[i_T] * w[i_T]
        #SV[i_T] = np.max(np.abs(u1))           #never used in Python code so removed
        #SA[i_T] = np.max(np.abs(u2+acc))       #never used in Python code so removed

    #return SD, PSV, PSA, SV, SA
    return PSA, u



