from pylab import *
from .extra import m1m2_from_mtotq
from lal import G_SI, MSUN_SI, C_SI

def get_area(m, chi, si=False):
    """Compute BH area from its mass in solar masses (m)
    and dimensionless spin magnitude (chi)
    
    If `si` returns in meters, otherwise MSUN
    """
    m_dist = m.copy()
    if si:
        m_dist *= G_SI * MSUN_SI / C_SI**2
    rp = m_dist*(1 + np.sqrt(1 - chi**2))
    area = 8*np.pi*m_dist*rp
    return area

def frac_diff(x, y):
    return 2*(x - y)/(x + y)

def get_area_diff(area1, ref):
    n = min(len(area1), len(ref))
    x = area1[np.random.choice(range(len(area1)), n, replace=False)]
    y = ref[np.random.choice(range(len(ref)), n, replace=False)]
    return (x - y) / y

def m1m2_from_mtotq(mtot, q):
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    return m1, m2

def get_insp_area(samples):
    m1, m2 = m1m2_from_mtotq(samples['mtotal'], samples['q'])
    try:
        chi1, chi2 = abs(samples['chi1z']), abs(samples['chi2z'])
    except ValueError:
        chi1, chi2 = abs(samples['chi1']), abs(samples['chi2'])

    return get_area(m1, chi1) + get_area(m2, chi2)

