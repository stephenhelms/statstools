import numpy as np
import numpy.ma as ma
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats
from itertools import *


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


class PoissonProcess(object):
    def __init__(self, tau):
        self.tau = tau

    def nextEvent(self):
        return random.exponential(self.tau, 1)[0]

    def eventsByTime(self, maxT):
        events = []
        ti = 0.
        while ti < maxT:
            ti += self.nextEvent()
            if ti <= maxT:
                events.append(ti)
        return np.array(events)

    def eventTimeSeries(self, maxT, dt):
        te = self.eventsByTime(maxT)
        ie = np.round(te / dt).astype(int)
        ts = np.zeros((np.round(maxT / dt),))
        ts[ie] = 1
        return ts

    @staticmethod
    def fit(tEvents):
        dtEvents = np.diff(tEvents)
        tau = stats.expon.fit(dtEvents)[1]
        return tau


class PoissonTwoStateProcess(object):
    def __init__(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2

    def nextEvent(self, prevStateIs1):
        if prevStateIs1:
            taui = self.tau2
        else:
            taui = self.tau1
        return random.exponential(taui, 1)[0]

    def eventsByTime(self, maxT):
        events = []
        states = []
        ti = 0.
        statei = True
        while ti < maxT:
            ti += self.nextEvent(statei)
            if ti <= maxT:
                events.append(ti)
                statei = not statei
                states.append(statei)
        return np.array(events), np.array(states, dtype='bool')

    def stateTimeSeries(self, maxT, dt):
        te, ts = self.eventsByTime(maxT)
        ie = np.round(te / dt).astype(int)
        state = np.zeros((np.round(maxT / dt),), dtype='bool')
        for (iprev, inext), statei in zip(_pairwise(ie), ts):
            state[iprev:inext] = statei
        return state
