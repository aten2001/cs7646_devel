"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data



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
    cr, adr, sddr, sr = compute_port_stats(port_val, rfr = 0, sf = 252)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        print 'Plot is WIP'
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    # Add code here to properly compute end value
    # ending value is the last value of the portfolio value
    ev = port_val[-1]
    return cr, adr, sddr, sr, ev
    

    
def compute_port_stats(port_val, rfr = 0, sf = 252):
    
    cr          = port_val[-1] / port_val[0] - 1
    daily_rets  = (port_val / port_val.shift(1)) - 1
    adr         = daily_rets.mean()
    sddr        = daily_rets.std()
    # Sharpe ratio: k * mean(_daily_rets_ - _daily_rf_) / std(_daily_rets_)
    # where k = sqrt(252) for daily sampling
    sr          = (adr / sddr ) * np.sqrt(sf)
   
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
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "End value of porfolio:", ev

if __name__ == "__main__":
    test_code()
