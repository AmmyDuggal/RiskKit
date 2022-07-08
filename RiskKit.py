# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:56:25 2022

@author: dugga
"""
import pandas as pd
import numpy as np

def drawdowns(returns_series:pd.Series):
    
    wealth_index = 100*(1+returns_series).cumprod()
    wealth_index.plot.line()
    previous_peak=wealth_index.cummax()
    previous_peak.plot.line()
    drawdown = (wealth_index-previous_peak)/previous_peak
    drawdown.plot()
    drawdown.min()
    #max drawdown since 2001:
    drawdown["2001"].min()
    drawdown["2001"].idxmin()#index/date of the max drawdown.
    return(pd.DataFrame({'Wealth_index':wealth_index, 'Previous_Peak':previous_peak, 'Drawdowns':drawdown}))

#drawdowns(rets["Largecap"])["2000":]

#VaR 99% means exclude the worst 1% returns, what is the worst return of the series that is left
# 99% VaR = 100 rs. means 1% of the time we expect to lose more than 100rs.
#VaR is usually expressed as a positive number.
#Conditional VaR -> expected loss beyond VaR CVaR= E(R|R<VaR)
import scipy.stats
def is_normal(x,alpha=0.01):
    """
    1% test by default
    """
    return scipy.stats.jarque_bera(x)[1]>alpha


def var_historic(r,level=5):
    
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level) #applies function to each column.
    elif isinstance(r,pd.Series):
        return abs(np.percentile(r, level))
    else:
        raise TypeError("Input series or df")
import scipy.stats
from scipy.stats import norm
def var_guassian(r,level=5,modified=False):
    
    if modified==False:
        if isinstance(r,pd.DataFrame):
            return r.aggregate(var_guassian,level=level) #applies function to each column.
        elif isinstance(r,pd.Series):
            return abs(r.mean()+norm.ppf(level/100)*r.std())
        else:
            raise TypeError("Input series or df")
    #Cornish Fisher method adjusts for kirtosis and skewness of return series over normal dbn assumption.
    else:
        z = norm.ppf(level/100)
        s  = scipy.stats.skew(r)
        k = 3 + scipy.stats.kurtosis(r)
        
        z = (z+ (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36)
        return( abs(r.mean()+z*r.std()) )

#Expected loss when var limit is crossed = conditional var
def cvar(r,level=5):
    
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar,level=level)
    else:
        raise TypeError("Series or df expected")
        
        
def annualised_vol(r,periods_per_year=252):
    return r.std()*(periods_per_year**0.5)

def annualised_ret(r,periods_per_year=252):
    cumret = (1+r).prod()
    no_periods = r.shape[0]
    return cumret**(periods_per_year/no_periods)-1

def sharpe_ratio(r, rfr = 0.06, periods_per_year=252):
    #rfr is annual 
    rfr = (1+rfr)**(1/periods_per_year) - 1
    excess_ret = r-rfr
    annual_excess_return = annualised_ret(excess_ret,periods_per_year)
    annual_vol = annualised_vol(r,periods_per_year)
    return annual_excess_return / annual_vol

def portfolio_returns(weights,returns):
    """
    weights->returns by matrix multiplication W(transpose)*R
    """
    return weights.T @ returns
    
def portfolio_vol(weights,covam):
    return (weights.T @ covam @ weights)**0.5


def gmv(covam):
    #GMV has only systematic risk, so only the risk u will be rewarded for.
    n=len(covam)
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0, 1.0), )*n
    sum_weights_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
        }
    weights = minimize(portfolio_vol,init_guess, constraints = (sum_weights_1) ,args = (covam, ), bounds = bounds,method = "SLSQP", options={'disp': False})
    return weights.x
    #OR
    #it is also the portfolio when all the assets have same returns
    #return sharpe_maximizer(0.1,np(1,n),covam)

def plot_efficientFrontier(n_points,er, covam, cml=False, rfr=0.1, show_ew = False, show_gmv=False):
    

    weights = optimal_weights(n_points,er, covam)
    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, covam) for w in weights]
    ef_fr = pd.DataFrame({"R":rets, "Vol":vols})
    ax = ef_fr.plot.scatter(x="Vol", y = "R",style=".-", figsize=(10,5))
    if cml:
        ax.set_xlim(left=0)
        rfr = 0.1
        w_msr = sharpe_maximizer(er, covam, rfr)#weights of max sharpe portfolio.
        r_msr = portfolio_returns(w_msr, er)
        vol_msr = portfolio_vol(w_msr, covam)

        cml_x =[0,vol_msr]
        cml_y = [rfr,r_msr]

        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    if show_ew:
        n = er.shape[0]
        wei = np.repeat(1/n, n)
        re = portfolio_returns(wei, er)
        volat = portfolio_vol(wei, covam)
        ax.plot([volat] , [re] , color ="goldenrod", marker = "o", markersize=12)
        
    if show_gmv:
        n=er.shape[0]
        w_gmv = gmv(covam)
        r_gmv = portfolio_returns(w_gmv, er)
        v_gmv = portfolio_vol(w_gmv, covam)
        
        ax.plot([v_gmv], [r_gmv], color ="midnightblue", marker = "o", markersize=12)
        

from scipy.optimize import minimize
def vol_minimizer(er , target_return,covam):
    """
    minimize: portfolio_vol
    constraints: sum of weights = 1 and vol minimization for given return.
    Returns
    -------
    None.

    """
    n = er.shape[0]#Number of Assets.
    init_guess = np.repeat(1/n, n)# equal weight is the intial guess.
    bounds = ((0.0,1.0),)*n #tuple times n 
    return_is_target = {'type': 'eq', #equality
                    'args': (er,),#tuple
                    #function to minimize 
                    'fun': (lambda wt, er: (target_return - portfolio_returns(wt, er)))
    }
                    
    
    sum_of_weight_1 = {
        'type': 'eq',
        #function to minimize
        'fun': lambda wt: np.sum(wt) - 1
        
    }
    
    result = minimize(portfolio_vol, init_guess, bounds = bounds, args = (covam, ), method = "SLSQP", options={'disp': False},
                  constraints= (return_is_target, sum_of_weight_1) )
    
    return result.x

def sharpe_maximizer(er ,covam, rfr):
    """

    -------
    None.

    """
    n = er.shape[0]#Number of Assets.
    init_guess = np.repeat(1/n, n)# equal weight is the intial guess.
    bounds = ((0.0,1.0),)*n #tuple times n 
    
    sum_of_weight_1 = {
        'type': 'eq',
        #function to minimize
        'fun': lambda wt: np.sum(wt) - 1
        
    }
    def neg_sharpe_ratio(weights, rfr, er, covam):
        #print("Weights are", weights)
        #print("returns are", er)
        r = portfolio_returns(weights, er)
        vol = portfolio_vol(weights, covam)
        return -(r - rfr)/vol
        

    result = minimize(neg_sharpe_ratio, init_guess, bounds = bounds, args = (rfr, er , covam ,), method = "SLSQP", options={'disp': False},
                  constraints= (sum_of_weight_1) )
    
    return result.x
#Be Very careful about the order of arguments that you give in.(first input argument in target function should be weights.)



def optimal_weights(n_points,er,covam):
    #n_points=er.shape[0]
    target_r = np.linspace(er.min(),er.max(),n_points)
    weights = [vol_minimizer(er, target_return, covam) for target_return in target_r]
    
    return weights