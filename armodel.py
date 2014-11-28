import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import itertools
import collections
from scipy import stats


def tsToAutoregressiveForm(Y, p):
    '''
    Y is a matrix of observations, where columns are the variables and rows
    are time points
    p is the max order
    Returns: Yf, the observations; Yp, the lagged predictors
    '''
    # if a 1D time series is provided, make it 2D to avoid axis problems below
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    nObs, nVar = Y.shape
    
    Yf = Y[p:, :]  # observed values in the fit
    Yp = ma.dstack(tuple(Y[p-pj:-pj, :] for pj in range(1, p+1)))

    # because the lagged data loses the first p values, replace them with masked
    lostObsf = ma.array(ma.zeros((p, nVar)), mask=True)  # placeholder for first p obs
    Yf = ma.vstack((lostObsf, Yf))
    lostObsp = ma.array(ma.zeros((p, nVar, p)), mask=True)  # placeholder for first p obs
    Yp = ma.vstack((lostObsp, Yp))
    return Yf, Yp


def coefmatrix(List,p):
    '''
    Returns the coefficient matrix of an AR(p) process
    List is a matrix of observations, where columns are the variables and rows are time points
    N is the number of variables
    p is the lag of the AR(p)
    List is a numpy array for N>1 (with a time series per row) or a list for N=1
    offset is the offset on List
    '''
    l,d=List.shape
    if d==1:
        yf, yp = vectorToAutoregressiveForm(List, p)
        param=np.linalg.lstsq(yp, yf)[0]
        return param
    else:
        y=List
        l,d=y.shape
        yf=y[p:,:]
        ypredictor=[]
        for lag in xrange(1,p+1):
            ypredictor.append(y[p-lag:-lag,:])
        yp=ma.hstack(ypredictor)
        sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))  # rows that have no masked values
        param = np.linalg.lstsq(yp[sel,:], yf[sel,:])[0]
        return param


# In[ ]:

def prediction(List,p):
    '''
    Returns the prediction of an AR(p) process
    List is a matrix of observations, where columns are the variables and rows are time points
    N is the number of variables
    p is the lag of the AR(p)
    List is a numpy array for N>1 (with a time series per row) or a list for N=1
    offset is the offset on List
    '''
    l,d=List.shape
    if d==1:
        yf, yp = vectorToAutoregressiveForm(List, p)
        param=np.linalg.lstsq(yp, yf)[0]
        pred=np.dot(yp,param)
    else:
        y=List
        l,d=y.shape
        yf=y[p:,:]
        ypredictor=[]
        for lag in xrange(1,p+1):
            ypredictor.append(y[p-lag:-lag,:])
        yp=ma.hstack(ypredictor)
        sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))  # rows that have no masked values
        param= np.linalg.lstsq(yp[sel,:], yf[sel,:])[0]
        pred=np.dot(yp,param)
    return pred


## Check ma instead on np

# In[5]:

def error(List,p):
    '''
    Returns the error vectors (collumns are variables and rows are time points)
    List is a matrix of observations, where columns are the variables and rows are time points
    '''
    l,d=List.shape
    if d==1:
        pred=prediction(List,N,p)
        yf, yp = vectorToAutoregressiveForm(List,p)
        eps = yf - pred
    else:
        y=List
        l,d=y.shape
        yf=y[p:,:]
        ypredictor=[]
        for lag in xrange(1,p+1):
            ypredictor.append(y[p-lag:-lag,:])
        yp=ma.hstack(ypredictor)
        sel = ~np.logical_or(np.any(yp.mask, axis=1), np.any(yf.mask, axis=1))  # rows that have no masked values
        AC= np.linalg.lstsq(yp[sel,:], yf[sel,:])[0]
        eps= yf[sel,:]-np.dot(yp[sel,:],AC)
    return eps

def model_dynamics(Ap):
    '''
    Computes the damping times and periods associated with lag order p
    Returns a tuple in which the first component is an array of the damping times and the second is an array of the periods
    Ap is the parameter matrix for lag order p
    '''
    l=np.linalg.eig(Ap)[0]
    tau=-1/np.log(abs(l))
    T=[]
    for i in l:
        if np.real(i)>0 and np.imag(i)==0:
            T.append('inf')
        else:    
            T.append((2*np.pi)/abs(cm.phase(i)))
    return (tau,T)


# In[ ]:

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

