import numpy as np
import numpy.ma as ma
import numpy.random as nr
import scipy.stats as ss
from scipy.special import psi, gamma
import scipy.spatial as ssp

# personal packages -- need to be on path
#import stats


'''
ENTROPY ESTIMATORS
'''


def entropy_normaldist(sigma):
    '''
    Calculates the theoretical entropy in bits for a univariate
    Gaussian distribution
    '''
    H_theory = np.log2(sigma*np.sqrt(2*np.pi*np.e))
    return H_theory


def entropy_hist(X):
    '''
    Compute the entropy in bits of an N-dimensional dataset using a histogram to
    estimate frequencies, ignoring bins with no counts.
    (Not the best method in general)
    '''
    # ignore masked rows
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel, :]
    # compute the D-dimensional histogram with automatic bin selection
    c, be = np.histogramdd(X)
    # compute bin centers
    bc = [np.array([np.array(edges).mean() for edges in zip(bei[:-1], bei[1:])])
          for bei in be]
    # normalize to get frequencies
    P = c/float(np.sum(c))
    # compute bin volume (constant)
    dV = np.array([bci[1]-bci[0] for bci in bc]).prod()
    # ignore bins with no observations
    sel = np.nonzero(P)
    P = P[sel]
    # return the Riemann-integrated entropy
    return np.sum(-P*np.log2(P)*dV)


def entropy_kd(X, f=2):
    '''
    Compute the entropy in bits of an N-dimensional dataset X using kernel density
    estimation to approximate the probability distribution. The distribution is
    integrated by sampling f-fold from the empirical distribution ([reference]).
    '''
    # ignore masked rows
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel, :]
    # generate a Gaussian kernel estimate of the probability dist
    Xkd = ss.gaussian_kde(X.T)
    # estimate the relative entropy by sampling from the estimated distribution of X
    S = Xkd.resample(int(round(f*X.shape[0])))
    Hest = -np.log2(Xkd(S))
    return Hest.mean()


def entropy_knn(X, k=1):
    '''
    Compute the entropy in bits of an N-dimensional dataset X using k-nearest
    neighbor statistics.
    Reference: http://dx.doi.org/10.1103/PhysRevE.69.066138
    '''
    # ignore masked rows
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel, :]
    N, d = X.shape
    # break degeneracy by adding random noise
    intens = 1e-10
    X = X + intens*nr.rand(N, d)
    # Krasnov mentioned two possible distance metric choices
    tree = ssp.cKDTree(X, k+1)
    eps = np.array([tree.query(xi,k+1)[0][k] for xi in X])*2.
    Hest = (-psi(k) + psi(N) + np.log(np.pi**(float(d)/2.))
            -np.log(gamma(1.+float(d)/2.))-np.log(2.**d) +
            float(d)/float(N)*(np.log(eps).sum()))
    return Hest/np.log(2)


def _kdpart_recur(X, l, N, Ln, k):
    '''
    This function is used by entropy_kdpart to recursively
    approximate the entropy.
    '''
    n, D = X.shape
    d = l % D
    Xmed = np.median(X[:,d])
    Xmin = X[:,d].min()
    Xmax = X[:,d].max()
    Z = np.sqrt(n)*(2*Xmed-Xmin-Xmax)/(Xmax-Xmin)
    if (n<k) or ((l >= Ln) and (np.abs(Z)< 1.96)):
        mu = (X.max(axis=0)-X.min(axis=0)).prod()
        return float(n)/float(N)*np.log2(float(N)/float(n)*mu)
    else:
        return (_kdpart_recur(X[X[:,d]<Xmed,:], l+1, N, Ln, k) +
                _kdpart_recur(X[X[:,d]>=Xmed,:], l+1, N, Ln, k))


def entropy_kdpart(X, k=20):
    '''
    Compute the entropy in bits of an N-dimensional dataset X by
    adaptively partitioning the data, maintaining at least k samples
    in each partition. The data is recursively split along the median
    until a minimum number of splits have been done (covering all
    dimensions) and the density within the bin is approximately
    uniform. I added an additional criteria that the bin has to contain
    k points (might have been done in the paper, but it wasn't in their
    pseudocode).
    Reference: http://dx.doi.org/10.1109/LSP.2009.2017346
    '''
    # ignore masked rows
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel, :]
    # compute minimum number of partitions
    N = X.shape[0]  # total number of samples
    Ln = 0.5*np.log2(N)
    # recursively partition data to generate entropy estimate
    return _kdpart_recur(X, 0, N, Ln, k)


'''
KULLBACK-LEIBLER DIVERGENCE
'''


def KLdiv_multivarnormal(mu0, Sigma0, mu1, Sigma1):
    '''
    Computes the theoretical KL-divergence in bits of two multivariate
    normal distributions with mean $\mu$ and covariance $\Sigma$.
    The theoretical divergence for a multivariate Gaussian is:
    $D_\text{KL}(\mathcal{N}_0 \| \mathcal{N}_1) = { 1 \over 2 }
        \left( \mathrm{tr} \left( \Sigma_1^{-1} \Sigma_0 \right) +
        \left( \mu_1 - \mu_0\right)^\top \Sigma_1^{-1} ( \mu_1 - \mu_0 )
        - k + \ln \left( { \det \Sigma_1 \over \det \Sigma_0  } \right) 
        \right)$
    '''
    D_theory = 0.5*(np.trace(np.dot(np.linalg.inv(Sigma1), Sigma0)) +
                np.dot(np.dot((mu1-mu0).T, np.linalg.inv(Sigma1)), (mu1-mu0)) +
                -len(mu0) + np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma0)))/np.log(2)
    return D_theory


def KLdiv_kd(X, Y, f=2):
    '''
    Computes the KL-divergence in bits of two N-dimensional distributions
    X and Y using kernel density estimation of the probability density.
    The divergence is integrated by sampling from the estimated distribution
    of X f-fold.
    '''
    # ignore masked rows
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel, :]
    sel = ~ma.getmaskarray(Y).any(axis=1)
    Y = Y[sel, :]
    # generate a Gaussian kernel estimate of the probability dist
    Xkd = ss.gaussian_kde(X.T)
    Ykd = ss.gaussian_kde(Y.T)
    # estimate the relative entropy by sampling from the estimated distribution of X
    S = Xkd.resample(int(round(f*(X.shape[0]+Y.shape[0]))))
    Dest = np.log2(Xkd(S)/Ykd(S))
    return Dest.mean()


def KLdiv_knn(X, Y, k=1, method='original'):
    '''
    Computes the KL-divergence in bits of two N-dimensional distributions
    X and Y using k-nearest neighbor statistics. Two related methods are
    implemented:
    'original' (default): just uses the distance to the k-NN
    'adaptive-k': corrects for the number of neighbors in each dimension

    Reference: http://dx.doi.org/10.1109/TIT.2009.2016060
    '''
    # remove missing data
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel,:]
    sel = ~ma.getmaskarray(Y).any(axis=1)
    Y = Y[sel,:]
    # shapes
    n, d = X.shape
    m = Y.shape[0]
    # add tiny noise to prevent degeneracy
    intens = 1e-10
    X = X + intens*np.random.randn(n, d)
    Y = Y + intens*np.random.randn(m, d)
    # calculation
    Xtree = ssp.cKDTree(X, k+1)
    Ytree = ssp.cKDTree(Y, k+1)
    rho = np.array([np.atleast_1d(Xtree.query(xi,k+1)[0])[-1] for xi in X])
    nu = np.array([np.atleast_1d(Ytree.query(xi,k)[0])[-1] for xi in X])
    if method is 'original':
        return (float(d)/float(n)*(np.log(nu/rho).sum()) +
                np.log(float(m)/float(n-1)))/np.log(2)
    elif method is 'adaptive-k':
        eps = np.vstack((rho,nu)).T.max(axis=1)
        ln = np.array([len(Xtree.query_ball_point(xi, epsi+1e-10))-1 for (xi, epsi) in zip(X,eps)])
        kn = np.array([len(Ytree.query_ball_point(xi, epsi+1e-10)) for (xi, epsi) in zip(X,eps)])
        rhol = np.array([np.atleast_1d(Xtree.query(xi,lni+1)[0])[-1] for xi,lni in zip(X,ln)])
        nuk = np.array([np.atleast_1d(Ytree.query(xi,kni)[0])[-1] for xi,kni in zip(X,kn)])
        return (float(d)*(np.log(nuk/rhol).mean()) + 
                (psi(ln)-psi(kn)).mean() +
                np.log(float(m)/float(n-1)))/np.log(2)
    else:
        raise Exception('Invalid method.')


'''
JENSEN-SHANNON DIVERGENCE
'''

def JSdiv_kd(X, Y, f=2):
    '''
    Calculates the Jensen-Shannon divergence between N-dimensional distributions
    X and Y, approximated with kernel density estimation and integrated by
    sampling f-fold from each estimated distribution.
    '''
    # remove missing data
    sel = ~ma.getmaskarray(X).any(axis=1)
    X = X[sel,:]
    sel = ~ma.getmaskarray(Y).any(axis=1)
    Y = Y[sel,:]
    # generate a Gaussian kernel estimate of the probability dist
    Xkd = ss.gaussian_kde(X.T)
    Ykd = ss.gaussian_kde(Y.T)
    # estimate the relative entropy by sampling from the estimated distributions
    n = int(round(f*(X.shape[0]+Y.shape[0])))
    S = Xkd.resample(n)
    JSx = np.log2(Xkd(S)/(Xkd(S)/2. + Ykd(S)/2.)+1e-10).mean()
    S = Ykd.resample(n)
    JSy = np.log2(Ykd(S)/(Xkd(S)/2. + Ykd(S)/2.)+1e-10).mean()
    return 0.5*JSx + 0.5*JSy


'''
MODULE TESTS
'''

def _testEntropy(sigma=0.3, m=2):
    H_theory = m*entropy_normaldist(sigma)
    print(('The theoretical entropy of a {0}-dimensional normal distribution ' +
           'with a standard deviation of {1} is {2} bits.').format(m, sigma, H_theory))
    # test distribution
    Z = sigma*np.random.randn(1000,m)
    # histogram
    H_hist = entropy_hist(Z)
    print(('The histogram estimate of the entropy is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(H_hist, np.abs(H_hist-H_theory)/H_theory*100.))
    # kernel density
    H_kd = entropy_kd(Z)
    print(('The kernel density estimate of the entropy is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(H_kd, np.abs(H_kd-H_theory)/H_theory*100.))
    # k-NN
    H_knn = entropy_knn(Z, 6)
    print(('The k-NN estimate of the entropy using k=6 is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(H_knn, np.abs(H_knn-H_theory)/H_theory*100.))
    # kd-part
    H_kdpart = entropy_kdpart(Z, 20)
    print(('The kd-partition estimate of the entropy is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(H_kdpart, np.abs(H_kdpart-H_theory)/H_theory*100.))


def _testKLdiv():
    # test distributions
    Sigma0 = 0.5*np.ones((4,4))
    for i in xrange(4):
        Sigma0[i,i] = 1
    mu0 = np.array([0.1, 0.3, 0.6, 0.9])
    Z0 = np.random.multivariate_normal(mu0, Sigma0, 1000)

    Sigma1 = 0.1*np.ones((4,4))
    for i in xrange(4):
        Sigma1[i,i] = 1
    mu1 = np.zeros(4,)
    Z1 = np.random.multivariate_normal(mu1, Sigma1, 1000)
    # theory
    D_theory = KLdiv_multivarnormal(mu0, Sigma0, mu1, Sigma1)
    print('The theoretical divergence is {0} bits ({1} nats).'.format(D_theory,
                                                                      D_theory/np.log2(np.e)))
    # kernel density
    D_kd = KLdiv_kd(Z0, Z1)
    print(('The kernel density estimate of the divergence is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(D_kd, np.abs(D_kd-D_theory)/D_theory*100.))
    # k-NN
    D_knn = KLdiv_knn(Z0, Z1, 1)
    print(('The 1-nearest neighbors estimate of the divergence is {0} bits, a relative error of '+
           '{1}% from the theoretical value.').format(D_knn, np.abs(D_knn-D_theory)/D_theory*100.))


def _testJSdiv():
    # test distributions
    Sigma0 = 0.5*np.ones((4,4))
    for i in xrange(4):
        Sigma0[i,i] = 1
    mu0 = np.array([0.1, 0.3, 0.6, 0.9])
    Z0 = np.random.multivariate_normal(mu0, Sigma0, 1000)

    Sigma1 = 0.1*np.ones((4,4))
    for i in xrange(4):
        Sigma1[i,i] = 1
    mu1 = np.zeros(4,)
    Z1 = np.random.multivariate_normal(mu1, Sigma1, 1000)
    # theory
    # ???
    # kernel density
    D_kd = JSdiv_kd(Z0, Z1)
    print('The JS-divergence is {0}.'.format(D_kd))
    D_kd_overlap = JSdiv_kd(Z0, Z0)
    print('The JS-divergence of identical datasets is {0}.'.format(D_kd_overlap))
    D_kd_nonoverlap = JSdiv_kd(Z0, Z0 + np.ones(Z0.shape[1])*100)
    print('The JS-divergence of non-overlapping datasets is {0}.'.format(D_kd_nonoverlap))

