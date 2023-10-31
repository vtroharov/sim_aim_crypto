#!/usr/bin/python

import requests
import datetime
import json
import sys
import coloring
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import csv


def save_data(coin, extra):
    response = requests.get(
        "https://api.coingecko.com/api/v3/coins/" + coin + "/market_chart?vs_currency=usd&days=max&interval=daily")
    data = response.json()
    out_file = open(coin + extra + ".json", "w")
    json.dump(data['prices'], out_file, indent=4)
    out_file.close()


def print_data(coin):
    in_file = open(coin + ".json", "r")
    data = json.load(in_file)
    in_file.close()

    print('Prices:')
    for d in data:
        date = datetime.datetime.fromtimestamp(d[0] / 1000).strftime('%Y-%m-%d')
        print('Date:', date, 'ETH:', d[1])


def get_ema(data, days):
    ema = 0
    multiplier = 2 / (days + 1)
    ema_list = []
    price_list = []
    dates_list = []
    for d in range(len(data)):
        if d < days:
            ema += data[d][1]
            ema_list.append(0)
            price_list.append(data[d][1])
            dates_list.append(data[d][0]/1000)
        elif d == days:
            ema += data[d][1]
            ema = ema / days
            ema_list.append(ema)
            price_list.append(data[d][1])
            dates_list.append(data[d][0]/1000)
        elif d > days:
            ema = (data[d][1] * multiplier) + (ema * (1 - multiplier))
            ema_list.append(ema)
            price_list.append(data[d][1])
            dates_list.append(data[d][0]/1000)
        else:
            print("Error in list")
    return dates_list, ema_list, price_list


def get_usable_dates(data, increment, init_date, end_date):
    i = 1
    data_out = []
    for d in data:
        if init_date <= d[0] <= end_date:
            if increment == 'monthly':
                date = datetime.datetime.fromtimestamp(d[0] / 1000).strftime('%Y-%m-%d').split('-')
                if date[2] == '01':
                    data_out.append([d[0], d[1]])
            else:
                if i == 1 or increment == 1:
                    data_out.append([d[0], d[1]])
                    i += 1
                elif i == increment:
                    i = 1
                else:
                    i += 1
    return data_out


def exec_algo(starting_amount, current_price, crypto, buy_sens_percent, sell_sens_percent, min_transact, ema, ema_percent):
    buy_sens = crypto * current_price * (buy_sens_percent / 100)
    sell_sens = -crypto * current_price * (sell_sens_percent / 100)
    value = starting_amount - (crypto * current_price)
    ema_decision = abs(ema/current_price*100 - 100)
    if float(value) > float(buy_sens):
        if float(value) > float(min_transact) and float(ema_decision) < float(ema_percent):
            print(value, min_transact)
            value = float(value) - float(buy_sens)
            print(coloring.blight_steel_blue("Bot Buys $" + str(round(value, 2)) +" worth of ETH."))
            # Insert Buy API call here
        else:
            value = 0
            print(coloring.bmedium_sea_green("Bot HODLs."))
    elif float(value) < float(sell_sens):
        if abs(float(value)) > float(min_transact) and float(ema_decision) < float(ema_percent):
            value = float(value) - float(sell_sens)
            print(coloring.indian_red("Bot Sells $" + str(abs(round(value, 2))) + " worth of ETH."))
            # Insert Sell API call here
        else:
            value = 0
            print(coloring.bmedium_sea_green("Bot HODLs."))
    else:
        value = 0
        print(coloring.bmedium_sea_green("Bot HODLs."))
    return value


def plot_values(dates, values, hodl):
    plt.subplot(211)
    plt.plot(dates, values, label='AIM')
    plt.plot(dates, hodl, label='HODL')
    plt.ylabel('Value')
    plt.xlabel('Date (Epoch)')
    plt.legend(title='Data', loc='upper left')


def plot_ema(dates, ema, price):
    plt.subplot(212)
    plt.plot(dates, price, label='PRICE')
    plt.plot(dates, ema, label='EMA')
    plt.ylabel('Price')
    plt.xlabel('Date (Epoch)')
    plt.legend(title='Data', loc='upper left')


def simulate_aim(data, increment, starting_amount, init_percent, init_date, end_date, buy_sens_percent,
                 sell_sens_percent, fee_percent, min_transact, ema_percent, ema_days, plot):
    if isinstance(init_date, str) == True and isinstance(end_date, str) == True:
        init_date = init_date.split('-')
        end_date = end_date.split('-')
        init_date = datetime.datetime(int(init_date[0]), int(init_date[1]), int(init_date[2])).timestamp() * 1000
        end_date = datetime.datetime(int(end_date[0]), int(end_date[1]), int(end_date[2])).timestamp() * 1000

    dates_ema, ema_plot, price_plot = get_ema(data, ema_days)
    data = get_usable_dates(data, increment, init_date, end_date)
    crypto = (starting_amount * init_percent / 100) / data[0][1]
    cash = starting_amount - (starting_amount * init_percent / 100)
    starting_amount = starting_amount * init_percent / 100
    dates_plot = []
    values_plot = []
    hodl_plot = []
    hodl_crypto_init = starting_amount / data[0][1]
    hodl_calc = 0
    cash_init = cash
    for d in data:
        date_index = dates_ema.index(d[0]/1000)
        ema = ema_plot[date_index]
        print("Date:", datetime.datetime.fromtimestamp(d[0] / 1000).strftime('%Y-%m-%d'), "\t- Price:", round(d[1], 2),
              "\t- Cash:", round(cash, 2), "\t- Crypto:", round(crypto, 3), "\t- Value:",
              round((crypto * d[1] + cash), 2), "\t- Consejo:", round(starting_amount - (crypto * d[1]), 2))
        result = exec_algo(starting_amount, d[1], crypto, buy_sens_percent, sell_sens_percent, min_transact, ema, ema_percent)
        crypto = crypto + (result / d[1])
        fee = abs(result * fee_percent / 100)
        cash = cash - result - fee

        hodl_calc = hodl_crypto_init * d[1] + cash_init

        if plot == 'plot':
            dates_plot.append(datetime.datetime.fromtimestamp(d[0] / 1000).strftime('%y-%m-%d'))
            # dates_plot.append(d[0]/1000)
            values_plot.append(round((crypto * d[1] + cash), 2))
            hodl_plot.append(round(hodl_calc, 2))

    if plot == 'plot':
        plot_values(dates_plot, values_plot, hodl_plot)
        plot_ema(dates_ema, ema_plot, price_plot)

    return (str(round(hodl_calc))), str(round(cash, 2)), str(round(crypto, 3)), str(round((crypto * data[-1][1] + cash), 2))


def get_market_struct(ticker):
    start = "2018-01-01"
    df = yf.download(tickers=ticker, start=start)
    df = df.reset_index()

    max_drawdown = 35
    x = len(df)
    trend = ''
    Peak = -np.inf
    date_Peak = 0
    Trough = np.inf
    date_Trough = 0
    ddd = np.empty((0, 3), np.datetime64)

    for i in range(0, x):
        up = 0
        dn = 0
        if trend == '' or trend == 'bull':
            if df.loc[i, 'Close'] >= Peak:
                Peak = df.loc[i, 'Close']
                date_Peak = df.loc[i, 'Date']
        if trend == '' or trend == 'bear':
            if df.loc[i, 'Close'] <= Trough:
                Trough = df.loc[i, 'Close']
                date_Trough = df.loc[i, 'Date']
        if Peak != -np.inf:
            dn = (Peak - df.loc[df.index[i], 'Close']) / (Peak / 100.0)
        if Trough != np.inf:
            up = (df.loc[df.index[i], 'Close'] - Trough) / (Trough / 100.0)

        if up >= max_drawdown:
            trend = 'bull'
            ddd = np.append(ddd, np.array([[date_Trough, df.loc[i, 'Date'], 1]]), axis=0)
            Trough = np.inf
            Peak = df.loc[df.index[i], 'Close']
            date_Peak = df.loc[i, 'Date']
        if dn >= max_drawdown:
            trend = 'bear'
            ddd = np.append(ddd, np.array([[date_Peak, df.loc[i, 'Date'], 2]]), axis=0)
            Peak = -np.inf
            Trough = df.loc[df.index[i], 'Close']
            date_Trough = df.loc[i, 'Date']

    df = df.set_index('Date')
    up_trend_s = ddd[ddd[:, 2] == 1, 0]
    up_trend_f = ddd[ddd[:, 2] == 1, 1]
    dn_trend_s = ddd[ddd[:, 2] == 2, 0]
    dn_trend_f = ddd[ddd[:, 2] == 2, 1]

    if ddd[len(ddd) - 1, 2] == 1:  # if the trend is growing, then we are looking for a local maximum
        ind = up_trend_f[len(up_trend_f) - 1]  # get the index of the beginning of the bullish trend
        imax = df.loc[ind:, 'Close'].idxmax()  # local maximum
        if df.loc[imax, 'Close'] > df.loc[ind, 'Close']:  # if the high is greater than the price of the beginning of the bullish trend, then add a peak
            dn_trend_s = np.append(dn_trend_s, imax)
    else:
        ind = dn_trend_f[len(dn_trend_f) - 1]  # bear market
        imin = df.loc[ind:, 'Close'].idxmin()
        if df.loc[imin, 'Close'] < df.loc[ind, 'Close']:
            up_trend_s = np.append(up_trend_s, imin)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'])
    ax.plot(up_trend_s, df.loc[up_trend_s, 'Close'], 'x', color='lime')
    print("Start of Bull Run:\n", df.loc[up_trend_s, 'Close'])
    ax.plot(up_trend_f, df.loc[up_trend_f, 'Close'], 'o', color='lime', markersize=4)
    print("Finish/Mid of Bull Run:\n", df.loc[up_trend_f, 'Close'])
    ax.plot(dn_trend_s, df.loc[dn_trend_s, 'Close'], 'x', color='red')
    print("Start of Bear Run:\n", df.loc[dn_trend_s, 'Close'])
    ax.plot(dn_trend_f, df.loc[dn_trend_f, 'Close'], 'o', color='red', markersize=4)
    print("Finish/Mid of Bear Run:\n", df.loc[dn_trend_f, 'Close'])
    fig.autofmt_xdate()
    plt.show()


def arg_error():
    print("Error using the AIM simulator!")
    print("Proper usage: python sim_aim.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $arg9 $arg10 $arg11 $arg12 $arg13")
    print("$arg1  - command (extract_data, print_data, sim_data, market_struct, vast_data). If $coin.json file does not exist, please run "
          "extract_data first.")
    print("$arg2  - Coin ID. Example 'ethereum'")
    print("$arg3  - Increment. Example 1 (for 1 day), 7 (for weekly) or 'monthly' (for monthly)")
    print("$arg4  - Total amount of funds.")
    print("$arg5  - Percentage of funds to start with.")
    print("$arg6  - Starting Date. (YYYY-MM-DD) Format.")
    print("$arg7  - Ending Date. (YYYY-MM-DD) Format.")
    print("$arg8  - Percentage Buy Sensibility.")
    print("$arg9  - Percentage Sell Sensibility.")
    print("$arg10 - Percentage Transaction Fee.")
    print("$arg11 - Minimum Transaction(AIM).")
    print("$arg12 - Percentage EMA deviation")
    print("$arg13 - EMA Days")


def execute_real_order():
    save_data('ethereum', "_exec")
    in_file = open("ethereum_exec.json", "r")
    data = json.load(in_file)

    first_buy = 1790
    ema_days = 20
    fee_percent = 1
    starting_amount = 10000
    init_percent = 50
    cash = starting_amount * init_percent / 100
    crypto = (starting_amount * init_percent / 100) / first_buy
    dates_ema, ema_plot, price_plot = get_ema(data, ema_days)
    for d in data:
        date_index = dates_ema.index(d[0] / 1000)
        ema = ema_plot[date_index]
    # Execute check every 3 days (Can use Epoch from data file to determine.
    result = exec_algo(cash, data[-1][1], crypto, 5, 5, 100, 30, ema)
    crypto = crypto + (result / d[1])
    fee = abs(result * fee_percent / 100)
    cash = cash - result - fee
    value = str(round((crypto * data[-1][1] + cash), 2))
    print(coloring.bdodger_blue(
        "Final Cash: " + str(cash) + " \t- Final Crypto: " + str(crypto) + " \t- Final Value: " + str(value) + " \t- ETH Price: " + str((data[-1][1]))))


if __name__ == '__main__':
    # ARGS: extract_data, print_data, market_struct, sim_data, vast_data, real_exec
    # try:
        # - Coin ID must be given to save_data, eg. ethereum
        if sys.argv[1] == "extract_data":
            if len(sys.argv) > 2:
                save_data(sys.argv[2], "")
            else:
                save_data('ethereum', "")

        # - Coin ID must be given to print_data, eg. ethereum
        elif sys.argv[1] == "print_data":
            if len(sys.argv) > 2:
                print_data(sys.argv[2])
            else:
                print_data('ethereum')

        # - Coin ticker must be given as per Yahoo Finance, eg. ETH-USD
        elif sys.argv[1] == "market_struct":
            if len(sys.argv) > 2:
                get_market_struct(sys.argv[2])
            else:
                get_market_struct('ETH-USD')

        # - Args:
        # - Coin ID must be given to print_data, eg. ethereum
        # - Increment must be give, eg. 1 (for 1 day), 7 (for weekly) or monthly (for monthly)
        # - Total amount of funds
        # - Initial % invested
        # - Date from in YYYY-MM-DD format
        # - Date to in YYYY-MM-DD format
        # - Buy sensitivity %
        # - Sell sensitivity %
        # - Fee %
        # - Min AIM transaction
        # - % EMA deviation
        # - EMA Days
        # - Plot?
        elif sys.argv[1] == "sim_data":
            if len(sys.argv) < 3:
                in_file = open("ethereum.json", "r")
            else:

                in_file = open(sys.argv[2] + ".json", "r")
            data = json.load(in_file)
            if len(sys.argv) < 13:
                arg_error()
                print("Using DEFAULT settings...(sim_data ethereum monthly 10000 50 2022-01-01 2022-06-01 10 10 1 50 10 50)")
                print("----------------------------------------------------------------------------------------------------")
                # hodl, cash, crypto, value = simulate_aim(data, 'monthly', 10000, 50, '2021-01-01', '2022-06-01', 10, 10, 1, 50, 10, 50, 'plot')
                hodl, cash, crypto, value = simulate_aim(data, 3, 10000, 50, '2022-10-30', '2023-10-30', 5, 5, 1, 100, 30, 20, 'none')
            else:
                hodl, cash, crypto, value = simulate_aim(data, sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7], float(sys.argv[8]),
                             float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11]), float(sys.argv[12]), float(sys.argv[13]), 'plot')
            plt.show(block=False)
            print(coloring.bdodger_blue("Final Cash: " + cash + " \t- Final Crypto: " + crypto + " \t- Final Value: " + value), coloring.bmedium_sea_green(" \t - HODL Value: " + hodl))
            in_file.close()
        elif sys.argv[1] == "vast_data":
            coin_epoch_start = 1538352000000 # 1438905600000 # - start of ETH
            start_date_intervals = 7884000000
            coin = 'ethereum'
            increment = [1, 3, 5, 7, 14, 30]
            init_amount = 100000
            init_percent = [50, 60, 70, 80]
            date_start = [coin_epoch_start]
            for i in range(4):
                date_start.append(coin_epoch_start + start_date_intervals)
                start_date_intervals += 7884000000
            months = [12] #[6, 9, 12]
            buy_sens = [3, 5, 7, 10, 12, 15, 20, 30, 50, 80]
            sell_sens = [3, 5, 7, 10, 12, 15, 20, 30, 50, 80]
            fee = 1
            min_trans = 1000
            ema_deviation = [3, 5, 7, 10, 15, 20, 30]
            ema_days = [20, 50, 100]

            if len(sys.argv) <= 2:
                in_file = open("ethereum.json", "r")
            else:
                in_file = open(sys.argv[2] + ".json", "r")
            data = json.load(in_file)

            csv_file = open((coin + '.csv'), 'w', newline='')
            writer = csv.writer(csv_file)
            header = ['Coin', 'Increment', 'Init_Amount', 'Init_Percent', 'Date_Start', 'Date_End', 'Months', 'Buy_Sens', 'Sell_Sens', 'Fee', 'Min_Trans', 'EMA_Deviation', 'EMA_Days', 'Cash', 'Final_Value', 'HODL']
            writer.writerow(header)
            j = 0
            for a in increment:
                for b in init_percent:
                    for c in date_start:
                        for d in months:
                            date_end = c + (d * 2628000000)
                            if date_end > 1601510400000:            # 1667324019000: # - Current Date
                                break
                            else:
                                for e in buy_sens:
                                    for f in sell_sens:
                                        for g in ema_deviation:
                                            for h in ema_days:
                                                hodl, cash, crypto, value = simulate_aim(data, a, init_amount, b, c, date_end, e, f, fee, min_trans, g, h, 'no_plot')
                                                j += 1
                                                print(j)
                                                # print(coloring.bdodger_blue("Final Cash: " + cash + " \t- Final Crypto: " + crypto + " \t- Final Value: " + value), coloring.bmedium_sea_green(" \t - HODL Value: " + hodl))
                                                row = [coin, a, init_amount, b, (c / 1000), (date_end / 1000), d, e, f, fee, min_trans, g, h, cash, value, hodl]
                                                writer.writerow(row)

            in_file.close()
            csv_file.close()
        elif sys.argv[1] == "real_exec":
            execute_real_order()


        plt.show(block=True)
    # except:
    #     arg_error()
