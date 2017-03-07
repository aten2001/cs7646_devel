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

    # read in adjusted closed pirces using the util function
    prices_all = get_data(symbols, dates)
    # get prices of only portfolio symbols, eject SPY
    prices = prices_all[symbols]
    prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1)
    # fill in nan values
    prices = prices.fillna(method='ffill') 
    # Add SPY for comparison
    prices_SPY = prices_all['SPY']
    
    # Init leverage dataframe
    leverage = pd.DataFrame(columns=['leverage'], index=[dates])
    leverage.ix[:,['leverage']] = 0

    #  Init cash transactions
    cash_transactions = pd.DataFrame(columns=['cash_transaction'], index=[dates])
    cash_transactions.ix[0, ['cash_transaction']] = start_val
    cash_transactions = cash_transactions.fillna(value=0)
    
    # init share transactions
    share_transactions = pd.DataFrame(columns = symbols, index = [dates])
    share_transactions = share_transactions.fillna(value=0)
    
    # derive positionsleverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
    columns = ['cash']
    positions = pd.DataFrame(columns=columns, index=[dates])
    positions.ix[0] = 0
    # Loop orders
    for i in range(len(orders)):
        if orders.ix[i]['Order'] == 'BUY':
            factor = 1.0
        else:
            factor = -1.0
        share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] = share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] + factor * orders.ix[i]['Shares']
        order_cost = factor * orders.ix[i]['Shares'] * prices.ix[orders.index[i].to_datetime()][orders.ix[i]['Symbol']]
        cash_transactions.ix[orders.index[i].to_datetime(), ['cash_transaction']] = cash_transactions.ix[orders.index[i].to_datetime(),  ['cash_transaction']] - order_cost
        p = share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']]
    
    
    # Sum up all cash transactions
    
    positions.cash = cash_transactions.cash_transaction.cumsum()
    # sum up total numbers in a portfolio
    shares = share_transactions.cumsum()
    
    positions = pd.concat([positions,shares * prices],axis=1)
    # todo filter based on leverage
    
    portvals = positions.sum(axis=1)
    
    

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

    of = "./orders/orders2.csv"
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
