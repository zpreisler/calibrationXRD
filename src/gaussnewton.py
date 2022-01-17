from src.database import DatabaseXRD,snip,Phase,MixPhase
from numpy import newaxis,loadtxt,diag,histogram,arange,linspace,sin,concatenate,array,exp,pi,zeros,ones,prod,newaxis,arctan,savetxt,c_,fabs,sqrt,concatenate
from numpy.random import normal
from numpy.linalg import pinv,inv
from scipy.optimize import curve_fit,least_squares
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class GaussNewton():
    
    def __init__(self,phase,spectra,max_theta = 53, min_intensity = 0.05):
        self.phase = phase
        self.spectra = spectra
        
        self.mu,self.i = self.phase.get_theta(max_theta = max_theta,min_intensity = min_intensity)
        
        self.n_channel = len(self.channel)
        
        self.params = ones(len(self) * 2)
        
        self.sigma2 = self.params[:len(self)]
        self.gamma = self.params[len(self):]
        
        self.sigma2[:] = 0.04
        
        self.scale = 1
    
    def __len__(self):
        return self.i.__len__()
        
    @property
    def theta(self):
        return self.spectra.theta
    
    @property
    def channel(self):
        return self.spectra.channel
    
    def core(self,x,mu,sigma2):
        return exp(-0.5 * (x - mu)**2 / sigma2)
    
    def ddsigma2(self,x,mu,sigma2):
        return 0.5 * (x - mu)**2 / sigma2**2
    
    def dda(self,channel,x,a,s,mu,sigma2):
        return -1.0 / sigma2 * 180 / pi * s / ((a + channel)**2 + s**2) * (x - mu)
    
    def dds(self,channel,x,a,s,mu,sigma2):
        return 1.0 / sigma2 * 180 / pi * (a + channel) / ((a + channel)**2 + s**2) * (x - mu)
    
    def ddbeta(self,x,mu,sigma2):
        return -1.0 / sigma2 * (x - mu)

    @property
    def z(self):     
        x = self.theta
        _z = zeros(self.n_channel)
        for mu,I,sigma2,gamma in zip(self.mu, self.i,
                                     self.sigma2, self.gamma):
            c = self.core(x,mu,sigma2)
            _z += gamma * I * c
            
        return _z * self.scale
    
    def _calibration(self):
        
        dopt = zeros((3,self.n_channel))     
        da,dbeta,ds = dopt[:]
        dgamma = []
        
        x = self.theta
        
        for mu,I,sigma2,gamma in zip(self.mu,self.i,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            h = gamma * I * c
            
            dgamma += [I * c]
            
            da += h * self.dda(self.channel,x,*self.spectra.a,*self.spectra.s,mu,sigma2)
            dbeta += h * self.ddbeta(x,mu,sigma2)
            ds += h * self.dds(self.channel,x,*self.spectra.a,*self.spectra.s,mu,sigma2)
        
        return list(dopt) + dgamma
    
    def calibration(self,alpha = 1):

        y = self.spectra.intensity
        dopt = self._calibration()
        
        dz = y - self.z
        d = array(dopt).T
        
        dr = pinv(d) @ dz

        new_gamma = self.gamma + dr[3:] * alpha
        if any(new_gamma < 0):
            f = new_gamma < 0
            d[:,3:][:,f] = 0
            dr = pinv(d) @ dz
   
        self.gamma[:] += dr[3:] * alpha
        self.spectra.opt[:] += dr[:3] * alpha

        self.dz = sum(dz)

    def _calibration_nobeta(self):
        
        dopt = zeros((3,self.n_channel))     
        da,dbeta,ds = dopt[:]
        dgamma = []
        
        x = self.theta
        
        for mu,I,sigma2,gamma in zip(self.mu,self.i,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            h = gamma * I * c
            
            dgamma += [I * c]
            
            da += h * self.dda(self.channel,x,*self.spectra.a,*self.spectra.s,mu,sigma2)
            ds += h * self.dds(self.channel,x,*self.spectra.a,*self.spectra.s,mu,sigma2)
        
        return list(dopt) + dgamma
 

    def calibration_nobeta(self,alpha = 1):

        y = self.spectra.intensity
        dopt = self._calibration_nobeta()
        
        dz = y - self.z
        d = array(dopt).T
        
        dr = pinv(d) @ dz

        new_gamma = self.gamma + dr[3:] * alpha
        if any(new_gamma < 0):
            f = new_gamma < 0
            d[:,3:][:,f] = 0
            dr = pinv(d) @ dz
   
        self.gamma[:] += dr[3:] * alpha
        self.spectra.opt[:] += dr[:3] * alpha

        self.gamma[self.gamma < 0] = 0.0

        self.dz = sum(dz**2)
    
    def _min_gamma(self):
        
        dgamma = []
        x = self.theta
        
        for mu,I,sigma2,gamma in zip(self.mu,self.i,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            dgamma += [I * c]
        
        return dgamma
    
    def min_gamma(self,alpha = 1):

        y = self.spectra.intensity
        dopt = self._min_gamma()
        
        z = self.z
        dz = y - z
        d = array(dopt).T
        
        dr = pinv(d) @ dz
   
        self.gamma[:] += dr * alpha
