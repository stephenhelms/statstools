import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import itertools
import collections
from scipy import stats
import matplotlib.pyplot as plt


def tsToAutoregressiveForm(Y, p):
    '''
    Y is a matrix of observations, where columns are the variables and rows
    are time points
    p is the max order
    Returns: Yf, the observations; Yp, the lagged predictors
    Note: The first p rows of the output are set to masked so the output can be directly
    compared with the input matrix Y.
    '''
    # if a 1D time series is provided, make it 2D to avoid axis problems below
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    nObs, nVar = Y.shape
    
    Yf = Y[p:, :]  # observed values in the fit
    Yp = ma.hstack(tuple(Y[p-pj:-pj, :] for pj in range(1, p+1)))

    # because the lagged data loses the first p values, replace them with masked
    lostObsf = ma.array(ma.zeros((p, nVar)), mask=True)  # placeholder for first p obs
    Yf = ma.vstack((lostObsf, Yf))
    lostObsp = ma.array(ma.zeros((p, nVar*p)), mask=True)  # placeholder for first p obs
    Yp = ma.vstack((lostObsp, Yp))
    return Yf, Yp


def fitARmodel(Y, p):
    '''
    '''
    # Get lagged vectors
    Yf, Yp = tsToAutoregressiveForm(Y, p)
    nObs, nVar = Yf.shape

    # rows that have no masked values
    sel = ~np.logical_or(np.any(Yf.mask, axis=1),
                         np.any(Yp.mask, axis=1))
    return np.linalg.lstsq(Yp[sel, :], Yf[sel, :])[0]


def predictARmodel(Y, W, p):
    Yf, Yp = tsToAutoregressiveForm(Y, p)
    return ma.dot(Yp, W)


def residualARmodel(Y, W, p):
    Yhat = predictARmodel(Y, W, p)
    return Y - Yhat


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


def model_dynamics(Ap, Fs=1.):
    '''
    Computes the damping times and periods associated with lag order p
    Returns a tuple in which the first component is an array of the damping times and the second is an array of the periods
    Ap is the parameter matrix for lag order p
    '''
    l = np.linalg.eig(Ap)[0]
    tau = -1/np.log(abs(l))
    T=2.*np.pi/ma.abs(ma.angle(l))
    T[np.abs(np.imag(l)) < 1e-8] = np.inf
    return (tau/Fs, T/Fs)


def trans_frobenius(Coef,dim,p):
    '''
    Coef is the coefficient matrix in which we have vertically stacked dim*dim A matrices
    '''
    A=Coef
    At_row=[]
    for i in range(1,p+1):
        At_row.append(A[4*(i-1):dim*i])
    Ap=np.hstack(At_row)
    Atr=np.zeros((dim*p,dim*p),complex)
    for i in range(dim):
        for j in range(dim*p):
            Atr[i,j]=Ap[i,j]
    i=dim
    j=0
    while j<dim*p:
        while i<dim*p:
            Atr[i,j]=1
            j+=1
            i+=1
        j+=1
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

