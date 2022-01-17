from numpy import loadtxt,arange,arctan,pi,pad,array,newaxis
from numpy import fft
from scipy.optimize import curve_fit
from scipy import signal

from src.database import snip

class Spectra():

    def __init__(self,file=None):
        if file:
            self.read_from_file(file)
            
        self.opt = array([-1264.,51.,2061.])
        self.a,self.beta,self.s = self.opt[:,newaxis]

    def from_array(self,x):
        self.raw_intensity = x
        self.channel = arange(1280)
        self.intensity = self.raw_intensity/self.raw_intensity.max()

        return self
    
    def read_from_file(self,file):
        
        self.raw_theta,self.raw_intensity = loadtxt(file,unpack=True)
        self.channel = arange(1280)
        
        self.intensity = self.raw_intensity/self.raw_intensity.max()

        return self
        
    def fce_calibration(self,x,a,beta,s):
        return (arctan((x+a)/s)) * 180 / pi + beta
    
    def fce_calibration_z(self,x,a,b,z):
        return (arctan((x+a)/(z*sin(b))) + b) * 180 / pi
    
    def calibrate_from_file(self,file):           
        
        x,y = loadtxt(file,unpack=True)
        self.opt,(opt_var) = curve_fit(self.fce_calibration,x,y)
        
        self.a,self.beta,self.s = self.opt[:,newaxis]
        
        self.opt_file = self.opt.copy() 
               
        return self
    
    @property
    def theta(self):
        return self.fce_calibration(self.channel,*self.opt)
    
    def snip(self,m = 24):

        y = self.intensity
        y = y - snip(y,m)
        self.intensity = y/y.max()
        
        return self

    def raw_snip(self,m = 24):

        y = self.intensity
        self._raw_snip = snip(y,m)
        y = self.raw_intensity - self._raw_snip
        self.intensity = y/y.max()
        
        return self
 
    def normalize(self):
        self.intensity /= self.intensity.max()
        return self
    
    def convolve(self,w=24):
        off = 4 * w
        kernel = signal.windows.gaussian(2*off-1,w)

        y = self.raw_intensity
        y_pad = pad(y,(off,off),'edge')

        f = fft.rfft(y_pad)
        w = fft.rfft(kernel,y_pad.shape[-1])
        y = fft.irfft(w * f)

        y = y[off*2:] / sum(kernel)
        self.intensity = y#/y.max()
        
        return self
    
    def plot(self, *args, **kwargs):

        plot(self.theta,self.intensity, *args, **kwargs)
        
        return self
