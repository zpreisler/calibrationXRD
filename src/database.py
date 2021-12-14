#!/usr/bin/env python

from matplotlib.pyplot import plot,show,figure,subplots,xlim,ylim,vlines,legend,fill_between
from scipy.interpolate import interp1d

from numpy import loadtxt,linspace,asarray,arcsin,pi,zeros,exp,arange,array,histogram,minimum,concatenate,ones
from numpy.random import randint
from numpy import random
from glob import glob

from scipy.optimize import curve_fit,least_squares
from scipy.stats import poisson

from scipy.signal import find_peaks

class Phase(dict):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __len__(self):
        return len(self['_pd_peak_intensity'][0])

    def get_theta(self,l=[1.541],scale=[1.0]):

        d,i = self['_pd_peak_intensity']

        theta = []
        intensity = []

        for _l,s in zip(l,scale):
            g = _l / (2.0 * d)
            theta += [360.0 * arcsin(g) / pi]
            intensity += [i * s]

        theta = concatenate(theta)
        intensity = concatenate(intensity) / 1000.0
        
        self.theta,self.intensity = array(sorted(zip(theta,intensity))).T

        return self.theta,self.intensity

    def plot(self, colors='k', linestyles='solid', label='', **kwargs):

        if not hasattr(self,'theta'):
            self.get_theta()

        vlines(self.theta,0,self.intensity, colors=colors, linestyles=linestyles, label=label, **kwargs)

        return self

class PhaseList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.label = None

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

    def random(self):
        idx = randint(self.__len__())
        return self[idx]

class DatabaseXRD(dict):

    def read_cifs(self,source):
        files = sorted(glob(source + '/*.cif'))

        i = 0

        for file in files:
            phase = Phase(name = file)

            with open(file,'r') as f:
                for line in f:
                    x = line.split()
                    if x:
                        y = x[0]
                        if y == '_chemical_formula_sum':
                            phase[y] = ' '.join(x[1:]).replace("'",'')

                        if y == '_chemical_name_mineral':
                            phase[y] = ' '.join(x[1:]).replace("'",'')

                        if y == '_chemical_name_common':
                            phase[y] = x[1:]

                        if y == '_pd_peak_intensity':
                            z = loadtxt(f,unpack=True,dtype=float)
                            phase[y] = z

            formula = phase['_chemical_formula_sum']

            if '_chemical_name_mineral' in phase:
                key = phase['_chemical_name_mineral']
            else:
                key = formula

            if key in self:
                self[key] += PhaseList([phase])
            else:
                self[key] = PhaseList([phase],label = i)
                i += 1

        return self

    def random(self):
        x = list(self.values())
        idx = randint(len(x))
        return x[idx]

def snip(x,m):
    x = x.copy()
    for p in range(1,m)[::-1]:
        a1 = x[p:-p]
        a2 = (x[:(-2 * p)] + x[(2 * p):]) * 0.5
        x[p:-p] = minimum(a2,a1)
    return x
