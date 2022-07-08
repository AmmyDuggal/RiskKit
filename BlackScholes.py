# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:21:19 2022

@author: dugga
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def calc_payoff(op_type,S,K):
    
    if op_type=="call":
        return np.maximum((S-K),0)
    elif op_type=="put":
        return np.maximum((K-S),0)
    else:
        raise TypeError("Option type must be call or put")



def BlackScholes( S=100, K=100 ,sigma=0.2, rfr=0.1, T=1, op_type="call"):
    
    if op_type=="call":
        payoff = calc_payoff("call",S,K)
    elif op_type == "put":
        payoff = calc_payoff("put",S,K)
    else:
        raise TypeError("Option type must be call or put")
        
    d1 =(1/sigma*np.sqrt(T)) * (np.log(S/K) + (rfr + (sigma**2)/2)*T)
    d2 =(1/sigma*np.sqrt(T)) * (np.log(S/K) + (rfr - (sigma**2)/2)*T)
    
    if op_type=="call":
            return S * ss.norm.cdf( d1 ) - K * np.exp(-rfr * T) * ss.norm.cdf( d2 )
    elif op_type=="put":
            return K * np.exp(-rfr * T) * ss.norm.cdf( -d2 ) - S * ss.norm.cdf( -d1 )
    else:
            raise ValueError("invalid type. Set call or put")
    
#BlackScholes(op_type="put") + 100-100*np.exp(-0.1*1) -> Put call parity 

# x=np.linspace(30,190,100)
# plt.plot(ss.lognorm.pdf(x, sigma**2,scale=np.exp(np.log(S))) + ( rfr - 0.5 * sigma**2 ) * T)
