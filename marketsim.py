"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
    return 'raghavendra6'


def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    #  read and sort the orders
    
    orders = pd.read_csv(orders_file, index_col=0, parse_dates=True, sep=',')
    orders = orders.sort_index()
    start_date = orders.index[0].to_datetime()
    end_date = orders.index[-1].to_datetime()
    orders.index = orders.index
    dates = pd.date_range(start_date, end_date)
    # get the reference dates when SPY trades
    dates = get_data(['SPY'], dates).index.get_values()
    symbols = orders.get('Symbol').unique().tolist()
    orders = pd.read_csv( orders_file, index_col='Date')  
    orders['Shares'][orders['Order'].str.upper()=='SELL'] = -orders['Shares'][orders['Order'].str.upper()=='SELL']
    df = get_data(['SPY'], dates)
    
    # Create a data frame to hold a matrix of all the stocks
    symbols = np.unique(orders['Symbol'].values.ravel())  
    for stock in symbols: 
        df[stock]=0   
    
    prices = get_data(symbols, df.index, False)
    prices = prices.fillna(method='ffill', axis=0)
    prices = prices.fillna(method='bfill', axis=0)
    
    df['Cash'] = start_val + 0.0
    prices['Cash'] = 1
    orders['Prices'] = 0
    for ind, row in orders.iterrows():
        # calculate leverage        
        # leverage = (sum(longs) + sum(abs(shorts)) / ((sum(longs) - sum(abs(shorts)) + cash)
        # get temporary table after the transaction is made, and before the transaction is made
        df_chk, df_chk_b4 = df.ix[ind,1:], df.ix[ind,1:]
        df_chk [row['Symbol']] = df[row['Symbol']][ind] + row['Shares']
        df_chk ['Cash'] = df['Cash'][ind] - prices[row['Symbol']][ind] * row['Shares']
        df_chk        = prices.ix[ind] * df_chk
        df_chk_b4  = prices.ix[ind] * df_chk_b4
        # calculate the leverage after and before 
        lev_after = sum(abs(df_chk[:-1])) / sum(df_chk )
        lev_before = sum(abs(df_chk_b4[:-1])) / sum(df_chk_b4 )
        # print lev_after, lev_before, ind
        if lev_after < 1.5 or lev_after < lev_before :      
            df[row['Symbol']][ind:end_date] = df[row['Symbol']][ind:end_date] + row['Shares']
            df['Cash'][ind:end_date] = df['Cash'][ind:end_date] - prices[row['Symbol']][ind] * row['Shares']
        else:
            print "Cancel the order", ind, row['Symbol'], row['Shares'], "Lev before", lev_before , "Lev after",  lev_after 
    
    df = df.iloc[:,1:] * prices
    portvals = df.sum(axis=1)
    

    return portvals
    

# frim mc1p1
def compute_port_stats(portfolio_value, \
                            rfr = 0.0, sf = 252.0):
    daily_returns = ((portfolio_value/portfolio_value.shift(1)) - 1).ix[1:] #calculate Daily Return. Tomorrow - Today, remove the header    (which is 0)
    cr = (portfolio_value.ix[-1]/portfolio_value.ix[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = np.sqrt(sf)*(daily_returns-rfr).mean()/std_daily_return
    start_date = portfolio_value.index[0].to_datetime()
    end_date = portfolio_value.index[-1].to_datetime()
    return cr, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_value, start_date, end_date





def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders3.csv"
    sv = 1000000
    
    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portfolio_value, start_date, end_date = compute_port_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, spy_value, start_date, end_date  = compute_port_stats(get_data(['SPY'], dates = pd.date_range(start_date, end_date)))

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
