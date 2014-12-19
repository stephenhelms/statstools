import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tsstats  # stephen's library

'''
Example usage:
Starting with an nObservation x nVar time series Y.

Test orders:
Shows the R2 value for each channel for various AR orders
plotARfitR2(Y, (1,2,3))

Fit model and examine residuals:
p = 1  # AR order
W = fitARmodel(Y, p)  # coefficient matrix
eps = residualARmodel(Y, W, p)  # residuals
plotARdiagostics(Y, eps, 115)  # look at model output

Simulate model with real residuals to test output:
Ysim = armodel.simulateARmodel(W, eps)
for i in xrange(Y.shape[1]):
    plt.subplot(Y.shape[1], 1, i+1)
    plt.plot(Y[:,i], 'k-')
    plt.plot(Ysim[:,i], 'r-')
    plt.xlabel('Time (frames)')
    plt.ylabel('Channel '+str(i+1))
plt.show()

Analyze model dynamics:
Wf = trans_frobenius(W)  # frobenius transformed coefficients
tau, T = model_dynamics(Wf, 11.5)  # model dynamics in s (for 11.5 Hz sampling)
print tau
print T
'''


def getLaggedSamples(Y, i, p):
    '''
    Returns a vector of p time lags at observation i for the
    nObservation x nVar time series Y.
    '''
    return ma.hstack(tuple(Y[i-pj, :] for pj in range(1, p+1)))


def tsToAutoregressiveForm(Y, p):
    '''
    Y is a matrix of observations, where columns are the variables and rows
    are time points
    p is the max order
    Returns: Yf, the observations; Yp, the lagged predictors
    Note: The first p rows of the output are set to masked so the output can
    be directly compared with the input matrix Y.
    '''
    if len(Y.shape)==1:
        Y = Y[:, np.newaxis]

    nObs, nVar = Y.shape

    Yf = Y[p:, :]  # observed values in the fit
    Yp = ma.hstack(tuple(Y[p-pj:-pj, :] for pj in range(1, p+1)))

    # because the lagged data loses the first p values, replace with masked
    lostObsf = ma.array(ma.zeros((p, nVar)), mask=True)
    Yf = ma.vstack((lostObsf, Yf))
    lostObsp = ma.array(ma.zeros((p, nVar*p)), mask=True)
    Yp = ma.vstack((lostObsp, Yp))
    return Yf, Yp


def fitARmodel(Y, p):
    '''
    Fits an AR model of order p to the nObservation x nChannels
    input time series Y.
    Returns the 2D coefficient matrix W.
    '''
    # Get lagged vectors
    Yf, Yp = tsToAutoregressiveForm(Y, p)
    nObs, nVar = Yf.shape

    # rows that have no masked values
    sel = ~np.logical_or(np.any(Yf.mask, axis=1),
                         np.any(Yp.mask, axis=1))
    return np.linalg.lstsq(Yp[sel, :], Yf[sel, :])[0]


def predictARmodel(Y, W, p):
    '''
    Calculates the prediction of the AR model of order p
    specified by W for the input data Y.
    Returns the nObservations x nChannels prediction time series.
    '''
    Yf, Yp = tsToAutoregressiveForm(Y, p)
    return ma.dot(Yp, W)


def residualARmodel(Y, W, p):
    '''
    Calculates the residual for each time point for the AR model
    from the input data Y using the model of order p specified by W.
    Returns the nObservations x nChannels residual.
    '''
    if len(Y.shape)==1:
        Y = Y[:, np.newaxis]
    Yhat = predictARmodel(Y, W, p)
    return Y - Yhat


def calculateR2(Y, eps):
    '''
    Calculates the coefficient of determination of the AR fit
    from the input data Y and the residual eps.
    Returns the R2 values for each channel.
    '''
    sel = ~np.logical_or(np.any(eps.mask, axis=1),
                         np.any(Y.mask, axis=1))
    return 1. - ((eps[sel, :]**2).sum(axis=0) /
                 ((Y[sel, :]-Y[sel, :].mean(axis=0))**2).sum(axis=0))


def plotARdiagostics(Y, eps, maxLag=115):
    for i in xrange(Y.shape[1]):
        plt.subplot(Y.shape[1], 3, i*3+1)
        C = tsstats.acf(Y[:, i], maxLag)
        plt.plot(C, 'k-')
        Cr = tsstats.acf(eps[:, i], maxLag)
        plt.plot(Cr, 'r-')
        plt.xlabel('Lag (frame)')
        plt.ylabel('ACF')
        plt.subplot(Y.shape[1], 3, i*3+2)
        plt.plot(eps[:, i], 'r-')
        plt.xlabel('Time (frames)')
        plt.ylabel('Channel '+str(i+1)+' Residual')
        plt.subplot(Y.shape[1], 3, i*3+3)
        sns.kdeplot(Y[:, i], color='k', label='Input')
        sns.kdeplot(eps[:, i], color='r', label='Residual')
        plt.xlabel('Channel '+str(i+1))
        plt.ylabel('Probability')
        plt.legend()
    plt.tight_layout()
    plt.show()


def plotARfitR2(Y, p):
    p = np.atleast_1d(p)
    w = 1./float(len(p)+2)  # bar width
    colors = sns.color_palette('GnBu_d', len(p))
    for i, pj in enumerate(p):
        W = fitARmodel(Y, pj)
        eps = residualARmodel(Y, W, pj)
        R2 = calculateR2(Y, eps)
        plt.bar(np.arange(0.5, Y.shape[1]+0.5)+float(i+1)*w, R2,
                width=w, color=colors[i], edgecolor='None',
                label='AR{0}'.format(str(pj)))
    plt.xticks(xrange(1, Y.shape[1]+1))
    plt.xlabel('Channel')
    plt.ylabel('$R^2$')
    plt.legend(loc=3, frameon=True)
    plt.show()


def model_dynamics(Wf, Fs=1.):
    '''
    Computes the damping times and periods associated with lag order p
    Returns a tuple in which the first component is an array of the
    damping times and the second is an array of the periods
    Wf is the Frobenius norm parameter matrix for lag order p
    '''
    l = np.linalg.eig(Wf)[0]
    # calculate damping times from real part
    tau = -np.sign(np.real(l))/np.log(np.abs(l))
    # calculate oscillation period from imag part
    T = 2.*np.pi/ma.abs(ma.angle(l))
    # ignore oscilaltion periods for components that are ~purely real
    T[np.abs(np.imag(l)) < 1e-8] = np.inf
    # convert to s if a sampling freq was provided
    return (tau/Fs, T/Fs)


def trans_frobenius(W):
    '''
    Coef is the coefficient matrix in which we have vertically stacked dim*dim A matrices
    This needs detailed comments Antonio
    '''
    nVar = W.shape[1]
    p = W.shape[0]/W.shape[1]

    At_row = []
    for i in range(1, p+1):
        At_row.append(W[4*(i-1):nVar*i])
    Ap = np.hstack(At_row)
    Atr = np.zeros((nVar*p, nVar*p), complex)
    for i in range(nVar):
        for j in range(nVar*p):
            Atr[i, j] = Ap[i, j]
    i = nVar
    j = 0
    while j < nVar*p:
        while i < nVar*p:
            Atr[i, j] = 1
            j += 1
            i += 1
        j += 1
    return Atr


def simulateARmodel(W, eps):
    '''
    Simulates the AR model of order p specified by nVar*p x nVar
    coefficient matrix W using the nObservation x nVar noise matrix eps.
    Respects masked values and masks the first p observations.
    Returns the simulated time series Ysim.

    The model can be checked by running this function with the real
    residual from residualARmodel, the results should match the original
    data.
    '''
    nVar = W.shape[1]
    p = W.shape[0]/W.shape[1]
    nSamples = eps.shape[0]
    Ysim = ma.zeros((nSamples, nVar))
    Ysim[:p, :] = eps[:p, :]
    for i in xrange(p, nSamples):
        Ysim[i, :] = ma.dot(getLaggedSamples(Ysim, i, p), W) + eps[i, :]
    Ysim[:p, :] = ma.masked
    return Ysim
