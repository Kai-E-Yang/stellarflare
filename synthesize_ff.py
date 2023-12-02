import numpy as np
from scipy.integrate import trapezoid as integral
import os
import pandas as pd

class synthesize_ff():
    def __init__(self,tempIn,rhoIn,depth,Gaunt=None,Teff=None,wavel=None):
        # self.metapath='/home/yangkai/codes/mypython/stellarflare'
        self.metapath='/Users/yangkai/Works/StellarFlare1D/laterphase/fitting/meta'
        tmp=pd.read_csv(os.path.join(self.metapath,'tess-response-function-v2.0.csv'))
        sidx=np.where(tmp.rpf>0)[0][0]-1
        eidx=np.where(tmp.rpf>0)[0][-1]+2
        if wavel is None:
            self.wavel=np.repeat(tmp.wl[sidx:eidx].to_numpy().reshape((1,eidx-sidx))*1e-7,tempIn.shape[0],axis=0)
            self.rpf   =np.repeat(tmp.rpf[sidx:eidx].to_numpy().reshape((1,eidx-sidx)),tempIn.shape[0],axis=0)
        else:
            self.wavel=np.repeat(wavel.reshape((1,len(wavel))),tempIn.shape[0],axis=0)
            self.rpf=np.ones(self.wavel.shape)
        if Gaunt is None:
            self.Gaunt = 1.0
        else:
            self.Gaunt = Gaunt
        if Teff is None:
            self.Teff = 6000
        else:
            self.Teff=Teff
#         in cgs unit
        self.h=6.6261e-27
        self.c=2.99792458e10
        self.kb=1.3807e-16
        self.sigmaT=6.65e-25
        self.mu=self.c/self.wavel
        if wavel is None:
            self.temperature=np.repeat(tempIn.reshape((tempIn.shape[0],1)),eidx-sidx,axis=1)
            self.rho=np.repeat(rhoIn.reshape((rhoIn.shape[0],1)),eidx-sidx,axis=1)
        else:
            self.temperature=np.repeat(tempIn.reshape((tempIn.shape[0],1)),len(wavel),axis=1)
            self.rho=np.repeat(rhoIn.reshape((rhoIn.shape[0],1)),len(wavel),axis=1)

        self.D=depth
        self.Paschen=820.4e-7
        self.Brackett=1458.0e-7
        self.Pfund=2279.0e-7
        self.Humphreys=3282.0e-7
        self.PaschenI=3
        self.BrackettI=4
        self.PfundI=5
        self.HumphreysI=6

    def plank_function(self,temp):
        return 2.0*self.h*self.mu**3/(np.exp(self.h*self.mu/self.kb/temp)-1)/self.c**2
    def kappa_mu_ff(self):
        # return 3.69e8*self.rho**2/np.sqrt(self.temperature)/self.mu**3*(1-np.exp(-self.h*self.mu/self.kb/self.temperature))
        return 3.69e8*self.rho**2/np.sqrt(self.temperature)/self.mu**3*(1-np.exp(-self.h*self.mu/self.kb/self.temperature))
    def I_thom(self):
        return self.rho*self.sigmaT*self.plank_function(self.Teff)*self.D
    def kappa_mu_pa(self):
        kappa=self.rho**2*2.0707e-16*2*self.PaschenI**2*(1-np.exp(-1*self.h*self.mu/self.kb/self.temperature))/self.temperature**1.5*np.exp(self.h*self.c/self.Paschen/self.kb/self.temperature)
        alpha=2.815e29/self.PaschenI**5/self.mu**3
        out=np.where(self.wavel<self.Paschen,kappa*alpha,0)
        return out

    def kappa_mu_br(self):
        kappa=self.rho**2*2.0707e-16*2*self.BrackettI**2*(1-np.exp(-1*self.h*self.mu/self.kb/self.temperature))/self.temperature**1.5*np.exp(self.h*self.c/self.Brackett/self.kb/self.temperature)
        alpha=2.815e29/self.BrackettI**5/self.mu**3
        out=np.where(self.wavel<self.Brackett,kappa*alpha,0)
        return out  

    def kappa_mu_pf(self):
        kappa=self.rho**2*2.0707e-16*2*self.PfundI**2*(1-np.exp(-1*self.h*self.mu/self.kb/self.temperature))/self.temperature**1.5*np.exp(self.h*self.c/self.Pfund/self.kb/self.temperature)
        alpha=2.815e29/self.PfundI**5/self.mu**3
        out=np.where(self.wavel<self.Pfund,kappa*alpha,0)
        return out

    def kappa_mu_hu(self):
        kappa=self.rho**2*2.0707e-16*2*self.HumphreysI**2*(1-np.exp(-1*self.h*self.mu/self.kb/self.temperature))/self.temperature**1.5*np.exp(self.h*self.c/self.Humphreys/self.kb/self.temperature)
        alpha=2.815e29/self.HumphreysI**5/self.mu**3
        out=np.where(self.wavel<self.Humphreys,kappa*alpha,0)
        return out

    def kappa_mu_thom(self):
        kappa=self.rho*self.sigmaT
        return kappa
        
    def I_ff(self):
        tau=self.kappa_mu_ff()*self.D
        out=self.plank_function(self.temperature)*tau
        return out

    def I_pa(self):
        tau=self.D*self.kappa_mu_pa()
        F=tau*self.plank_function(self.temperature)
        return F

    def I_br(self):
        tau=self.D*self.kappa_mu_br()
        F=tau*self.plank_function(self.temperature)
        return F

    def I_pf(self):
        tau=self.D*self.kappa_mu_pf()
        F=tau*self.plank_function(self.temperature)
        return F

    def I_hu(self):
        tau=self.D*self.kappa_mu_hu()
        F=tau*self.plank_function(self.temperature)
        return F

    def I_total(self):
        tau_total=self.tau_ff()+self.tau_pa()+self.tau_br()+self.tau_pf()+self.tau_hu()
        F=self.plank_function(self.temperature)*(1-np.exp(-1*tau_total))
        return F
    def tau_thom(self):
        return self.kappa_mu_thom()*self.D
    def tau_ff(self):
        return self.kappa_mu_ff()*self.D
    def tau_pa(self):
        return self.kappa_mu_pa()*self.D
    def tau_br(self):
        return self.kappa_mu_br()*self.D
    def tau_pf(self):
        return self.kappa_mu_pf()*self.D
    def tau_hu(self):
        return self.kappa_mu_hu()*self.D
    def tess_curve(self):
        core=self.cal_rt()*self.rpf
        return integral(core, x=self.wavel,axis=1).reshape(-1)
        
    def tess_curve_thom(self):
        core=self.I_thom()*self.rpf*0.4
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_ff(self):
        core=self.I_ff()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_pa(self):
        core=self.I_pa()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_br(self):
        core=self.I_br()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_pf(self):
        core=self.I_pf()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_hu(self):
        core=self.I_hu()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_bg(self):
        core=self.plank_function(self.Teff)*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c

    def tess_curve_total(self):
        core=self.I_total()*self.rpf
        return integral(core/self.wavel**2, x=self.wavel,axis=1).reshape(-1)*self.c
