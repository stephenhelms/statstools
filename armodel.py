import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import itertools
import collections
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def tsToAutoregressiveForm(Y, p):
    '''
    Y is a matrix of observations, where columns are the variables and rows
    are time points
    p is the max order
    Returns: Yf, the observations; Yp, the lagged predictors
    Note: The first p rows of the output are set to masked so the output can
    be directly compared with the input matrix Y.
    '''
    Y = ma.atleast_2d(Y)
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
    Yhat = predictARmodel(Y, W, p)
    return Y - Yhat


def calculateR2(Y, eps):
    '''
    Calculates the coefficient of determination of the AR fit
    from the input data Y and the residual eps.
    Returns the R2 values for each channel.
    '''
    return 1. - eps.var(axis=0)/(Y-Y.mean(axis=0)).var(axis=0)


def plotARfit(Y, p):
    W = fitARmodel(Y, p)
    Yhat = predictARmodel(Y, W, p)
    for i in xrange(Y.shape[1]):
        plt.subplot(Y.shape[1], 2, i*2+1)
        plt.plot(Y[:, i], 'k-')
        plt.plot(Yhat[:, i], 'r-')

        plt.subplot(Y.shape[1], 2, i*2+2)
        plt.plot(Y[:, i] - Yhat[:, i], 'k-')
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


def f(List,i,p):
    x=[]
    for j in range(1,p+1):
        x.append(List[p-j:-j,:])
    xmatrix=np.hstack(x)
    return xmatrix[i,:]


# In[ ]:

def simul(List,p,Coef):
    y1=List.copy()
    for i in range(len(y1)-p-1):
        y1[p+i,:]=np.dot(f(y1,i,p),Coef)+y1[p+i,:]
    return y1

