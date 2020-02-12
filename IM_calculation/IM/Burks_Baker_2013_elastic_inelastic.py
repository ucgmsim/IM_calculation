import numpy as np

# %
# % Compute time history response of bilinear SDOF oscillator to ground
# % motion excitation.
# %
# % Author: Nicolas Luco
# % Last Revised: September 11, 2003
# % Reference: "Dynamics of Structures" (1995) by A.K. Chopra
# %
# % INPUT
# % T     = periods                           ( 1 x num_oscillators, or scalar )
# % z     = damping ratios                    ( " )
# % dy    = yield displacements               ( " )
# % alpha = strain hardening ratios           ( " )
# % ag    = ground acceleration time history  ( length(ag) x 1 )
# % Dtg   = time step of ag                   ( scalar )
# % Dt    = analyis time step                 ( scalar )
# %
# % OUTPUT
# % S.d   = relative displacement spectrum               ( 1 x num_oscillators )
# %  .v   = pseudo relative velocity spectrum            ( " )
# %  .a   = pseudo acceleration spectrum                 ( " )
# % H.d   = relative displacement response time history  ( length(ag) x num_oscillators )
# %  .v   = relative velocity response time history      ( " )
# %  .a   = acceleration response time history           ( " )
# %  .fs  = force response time history                  ( " )
# %
# %--------------------------------------------------------------------------
# %
# % Provided as e-supplement to:
# % Burks, L.S. and J.W. Baker (2013), "Validation of ground motion
# %   simulations through simple proxies for the response of engineered
# %   systems," Bulletin of the Seismological Society of America.
# %
# %--------------------------------------------------------------------------
#

GAMMA = 1 / 2
BETA = 1 / 6  # linear acceleration (stable if Dt/T<=0.551)


def Bilinear_Newmark_withTH(period: np.ndarray, z: np.ndarray, dy: np.ndarray, alpha: np.ndarray,
                            ag: np.ndarray, dt: float):

    num_oscillators = max([period.size, z.size, dy.size, alpha.size])

    # m*a + c*v + fs(k,fy,kalpha) = p
    m = 1
    w = 2 * np.pi / period
    c = z * (2 * m * w)
    k = (w ** 2 * m) * np.ones(num_oscillators)
    fy = k * dy
    kalpha = k * alpha

    p = -ag * m

    lp = p.size
    d = np.zeros((lp, num_oscillators))
    v = np.zeros((lp, num_oscillators))
    a = np.zeros((lp, num_oscillators))
    fs = np.zeros((lp, num_oscillators))

    a[0] = (p[0] - c * v[0] - fs[0]) / m
    A = 1 / (BETA * dt) * m + GAMMA / BETA * c
    B = 1 / (2 * BETA) * m + dt * (GAMMA / (2 * BETA) - 1) * c

    for i in range(lp-1):

        DPi = p[i + 1] - p[i] + A * v[i] + B * a[i]

        ki = k
        jj = np.where(
            np.logical_or(
                np.logical_and(DPi > 0, fs[i] >= fy + kalpha * (d[i] - dy)),
                np.logical_and(DPi < 0, fs[i] <= -fy + kalpha * (d[i] + dy))
            )
        )
        ki[jj] = kalpha[jj]

        Ki = ki + A / dt

        Ddi = DPi / Ki
        fs[i + 1] = fs[i] + ki * Ddi
        d[i + 1] = d[i] + Ddi

        fsmax = fy + kalpha * (d[i + 1] - dy)
        fsmin = -fy + kalpha * (d[i + 1] + dy)
        jjabove = np.where(fs[i + 1] > fsmax)
        jjbelow = np.where(fs[i + 1] < fsmin)
        if (len(jjabove)+len(jjbelow)) > 0:
            fs[i + 1, jjabove] = fsmax[jjabove]
            fs[i + 1, jjbelow] = fsmin[jjbelow]
            Df = fs[i + 1] - fs[i] + (Ki - ki) * Ddi
            DR = DPi - Df
            Ddi = Ddi + DR / Ki
            d[i + 1] = d[i] + Ddi

        Dvi = (
            GAMMA / (BETA * dt) * Ddi
            - GAMMA / BETA * v[i]
            + dt * (1 - GAMMA / (2 * BETA)) * a[i]
        )
        v[i + 1] = v[i] + Dvi

        a[i + 1] = (p[i + 1] - c * v[i + 1] - fs[i + 1]) / m

    Sd = np.max(np.abs(d), axis=0)
    # Sv = Sd * (2 * np.pi / period)
    # Sa = Sd * (2 * np.pi / period) ** 2
    #

    # Hd = d
    # Hv = v
    # Ha = a
    # Hfs = fs
    # else:

    return Sd #, Sv, Sa, Hd, Hv, Ha, Hfs


#
# % m*a + c*v + fs(k,fy,kalpha) = p
# % Interpolate p=-ag*m (linearly)
# % Memory allocation & initial conditions
# % Initial calculations
# % Time stepping
# % Spectral values
# % Histories at tg
