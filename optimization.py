"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as sco

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], rfr = 0, gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    n_alloc = len(syms)
    allocs = np.random.random(n_alloc)
    # ensure allocs sum to 1
    
    allocs = allocs/np.sum(allocs)
    # find the allocations for the optimal portfolio
    allocs = opt_allocs(prices)
    
    # hope this does not happen since we have constrained the minimization function
    if np.sum(allocs) > 1.0:
        allocs_n = allocs / np.sum(allocs)
        
    # Get daily portfolio value
    # resued the code from analysis.py mc1p1
    port_val = compute_port_val(prices, allocs, start_value = 1)
    
    #Get portfolio statistics (note: std_daily_ret = volatility)
    # used from mc1p1
    cr, adr, sddr, sr = compute_port_stats(port_val, rfr, sf = 252)
    
    # normalize SPY for comaprison
    norm_SPY = prices_SPY/prices_SPY.iloc[0]
    
    

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # create a dataframe to house port val with SPY
        df_temp = pd.concat([port_val, norm_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily Portfolio Value and SPY")
        pass

    return allocs, cr, adr, sddr, sr

def opt_allocs(prices):
    n_allocs  = len(prices.columns)
    # extract log of returns
    returns     = np.log(prices / prices.shift(1))
    
    # initialize empty arrays
    prets = []
    pvols = []
    
    # sample allocs,  make sure they sum to 1
    for i in range (3000):
        allocs = np.random.random(n_allocs)
        allocs /= np.sum(n_allocs)
        
    
    
    
    def get_alloc_stats(allocs):
        
        """
        returns expected port stats in float
        expected returns, expected volatility, 
        expected sharpe_ratio for rf = 0
        """
        allocs = np.array(allocs)
        e_port_rets = np.sum(returns.mean() * allocs) * 252
        e_port_vol  = np.sqrt(np.dot(allocs.T,np.dot(returns.cov() * 252, allocs)))
        return np.array([e_port_rets, e_port_vol, e_port_rets/e_port_vol])
        
        
    def min_sr(allocs):
        return -get_alloc_stats(allocs)[2]
        
    # constraints = all allocs should add upto 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # valid boundas are always btw 0 & 1
    bounds = tuple((0, 1) for x in range(n_allocs))
    # start with an initial guess, lecture 1-08, 09
    ig = n_allocs * [1./n_allocs]
    
    # convex opt using minimize function
    opt_allocs = sco.minimize(min_sr, ig , method='SLSQP',
                       bounds=bounds, constraints=cons)
    # dict key of x returns allocs)

    return (opt_allocs['x'])
    
    
    
    
    
    
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    # add code here to compute daily portfolio values
    port_val = compute_port_val(prices, allocs, start_value = 1)
    # Get portfolio statistics (note: std_daily_ret = volatility)
    # add code here to compute stats
    cr, adr, sddr, sr = compute_port_stats(port_val, rfr, sf = 252)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    # Add code here to properly compute end value
    # ending value is the last value of the portfolio value
    ev = port_val[-1] * sv
    return cr, adr, sddr, sr, ev



def compute_port_stats(port_val, rfr, sf = 252):

    cr          = port_val[-1] / port_val[0] - 1
    daily_rets  = (port_val / port_val.shift(1)) - 1
    adr         = daily_rets.mean()
    sddr        = daily_rets.std()
    # Sharpe ratio: k * mean(_daily_rets_ - _daily_rf_) / std(_daily_rets_)
    # where k = sqrt(252) for daily sampling
    sr          = np.sqrt(sf) * (daily_rets - rfr).mean() / sddr

    return cr, adr, sddr, sr

def compute_port_val(prices, allocs, start_value = 1):
    """
    Helper function to compute value of daily portfolio;
    Step-1 : read in adjusted close prices as input
    Step-2 : Normalize the prices according to the first day.
             The first row for each stock should have a value of 1.0 at this point
    Step-3 : Multiply each column by the allocation to the corresponding equity.
    Step-4 : Multiply these normalized allocations
             by starting value (assume 1) of overall portfolio, to get position values.

    Step-5 : Sum each row (i.e. all position values for each day).
             That is your daily portfolio value
    """
    # Step-1 done
    # Step-2
    # normalize first row to 1 using iloc method
    norm_prices = prices/prices.iloc[0]
    # Step-3
    alloc_mult = norm_prices * allocs
    # Step-4
    position_values = alloc_mult * start_value
    #Step-5, sumc across rows: axis -> 1
    port_val = position_values.copy().sum(axis=1)

    return port_val




def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
    
    
    start_date = dt.datetime(2004,1,1)
    end_date = dt.datetime(2006,1,1)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
    
    start_date = dt.datetime(2004,12,1)
    end_date = dt.datetime(2006,5,31)
    symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
    
    start_date = dt.datetime(2005,12,1)
    end_date = dt.datetime(2006,5,31)
    symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
    
    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
    
    #start_date = dt.datetime(2010,06,01)
    #end_date = dt.datetime(2011,06,01)
    #symbols = ['AAPL', 'GLD','GOOG', 'XOM']
    #allocations = [0.1, 0.4, 0.5, 0.0]
    #start_val = 1000000
    #risk_free_rate = 0.005
    #sample_freq = 252
    # Assess the portfolio

    #cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
    #    syms = symbols, \
    #    allocs = allocations,\
    #    rfr    = risk_free_rate, \
    #    sv = start_val, \
    #    gen_plot = False)
    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
