import numpy as np
import pandas as pd
import ta.momentum as momentum
import ta.trend as trend
import ta.volatility as volatility
import ta.volume as volume
from stockstats import StockDataFrame as sdf
import plotly.graph_objects as go
import re

import xutils.data.pandas_utils as pu

BUY = 2
HOLD = 1
SELL = 0


# def get_stock_data(ticker, download_path=None, start=None, end=None, source='yahoo', api_key=None,
#                    normalize_columns=True):
#     df = web.DataReader(ticker, data_source=source, start=start, end=end, api_key=api_key)
#     if normalize_columns:
#         pu.lower_case_columns(df)
#     return df


# def save_stock_data(ticker, start_date, end_date, file_name="prices.csv", field="Adj Close"):
#     """
#     Gets historical ticker data of given tickers between dates
#     :param ticker: company, or companies whose data is to fetched
#     :type ticker: string or list of strings
#     :param start_date: starting date for ticker prices
#     :type start_date: string of date "YYYY-mm-dd"
#     :param end_date: end date for ticker prices
#     :type end_date: string of date "YYYY-mm-dd"
#     :return: stock_data.csv
#     """
#     fix.pdr_override()
#     i = 1
#     try:
#         all_data = pdr.get_data_yahoo(ticker, start_date, end_date)
#         stock_data = all_data[field]
#         # stock_data.to_csv(f"{ticker}_prices.csv")
#         stock_data.to_csv(file_name)
#     except ValueError:
#         i += 1
#         if i < 5:
#             time.sleep(10)
#             save_stock_data(ticker, start_date, end_date)
#         else:
#             time.sleep(120)
#             save_stock_data(ticker, start_date, end_date)
#
#
# def back_test(model, seq_len, ticker, start_date, end_date, dim):
#     """
#     A simple back test for a given date period
#     :param model: the chosen strategy. Note to have already formed the model, and fitted with training data.
#     :param seq_len: length of the days used for prediction
#     :param ticker: company ticker
#     :param start_date: starting date
#     :type start_date: "YYYY-mm-dd"
#     :param end_date: ending date
#     :type end_date: "YYYY-mm-dd"
#     :param dim: dimension required for strategy: 3dim for LSTM and 2dim for MLP
#     :type dim: tuple
#     :return: Percentage errors array that gives the errors for every test in the given date range
#     """
#     data = pdr.get_data_yahoo(ticker, start_date, end_date)
#     stock_data = data["Adj Close"]
#     errors = []
#     for i in range((len(stock_data) // 10) * 10 - seq_len - 1):
#         x = np.array(stock_data.iloc[i: i + seq_len, 1]).reshape(dim) / 200
#         y = np.array(stock_data.iloc[i + seq_len + 1, 1]) / 200
#         predict = model.predict(x)
#         while predict == 0:
#             predict = model.predict(x)
#         error = (predict - y) / 100
#         errors.append(error)
#         total_error = np.array(errors)
#         # If you want to see the full error list then print the following statement

# Technical indicators

def calc_intervals(df, intervals, prefix, calc_fn, **kwargs):
    return {f"{prefix}_{i}": calc_fn(df, i, **kwargs)
            for i in intervals}


# not used
def rsi(df, intervals, col_name="close"):
    """
    stockstats lib seems to use 'close' column by default so col_name
    not used here.
    This calculates non-smoothed RSI
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        df['rsi_' + str(i)] = df_ss['rsi_' + str(i)]

        del df['close_-1_s']
        del df['close_-1_d']
        del df['rs_' + str(i)]

        df['rsi_' + str(intervals[0])] = momentum.rsi(df[col_name], i, fillna=True)


def rsi_smooth(df, intervals, col_name="close"):
    """
    Momentum indicator
    As per https://www.investopedia.com/terms/r/rsi.asp
    RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
    RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )
    E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
    http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Verified!
    """
    prev_avg_gain = np.inf
    prev_avg_loss = np.inf
    rolling_count = 0

    def calculate_rsi(series, period):
        # nonlocal rolling_count
        nonlocal prev_avg_gain
        nonlocal prev_avg_loss
        nonlocal rolling_count

        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        # sum_gains = series[series >= 0].sum()
        # sum_losses = np.abs(series[series < 0].sum())
        curr_gains = series.where(series >= 0, 0)  # replace 0 where series not > 0
        curr_losses = np.abs(series.where(series < 0, 0))
        avg_gain = curr_gains.sum() / period  # * 100
        avg_loss = curr_losses.sum() / period  # * 100

        if rolling_count == 0:
            # first RSI calculation
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
        else:
            # smoothed RSI
            # current gain and loss should be used, not avg_gain & avg_loss
            rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                     (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))

        # df['rsi_'+str(period)+'_own'][period + rolling_count] = rsi
        rolling_count = rolling_count + 1
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
        return rsi

    diff = df[col_name].diff()[1:]  # skip na
    cols = {}
    for period in intervals:
        # df['rsi_' + str(period)] = np.nan
        # df['rsi_'+str(period)+'_own_1'] = np.nan
        rolling_count = 0
        res = diff.rolling(period).apply(calculate_rsi, args=(period,), raw=False)
        # df['rsi_' + str(period)][1:] = res
        # df.loc[1:, 'rsi_' + str(period)] = res
        cols['rsi_' + str(period)] = res

    return cols

    # df.drop(['diff'], axis = 1, inplace=True)


def williamr(df, interval):
    return momentum.williams_r(df['high'], 
                               df['low'], 
                               df['close'], 
                               interval, 
                               fillna=True)


def mfi(df, interval):
    """
    momentum type indicator
    """
    return volume.money_flow_index(df['high'],
                                   df['low'],
                                   df['close'],
                                   df['volume'],
                                   window=interval,
                                   fillna=True)


def sma(df_ss, interval, col_name="close"):
    """
    Momentum indicator
    """
    return df_ss[col_name + '_' + str(interval) + '_sma']
    # if sma_column_name in df.columns:
    #     print("****DELETE SMA COL 190", sma_column_name in df_cols_orig, sma_column_name)
    #     del df[sma_column_name]


def ema(df_ss, interval, col_name="close"):
    """
    Needs validation
    Momentum indicator
    """
    return df_ss[col_name + '_' + str(interval) + '_ema']
    # if ema_column_name in df.columns:
    #     print("delete ema", ema_column_name, " in orig?", ema_column_name in orig_df_columns)
    #     del df[ema_column_name]


def wavg(rolling_prices, period):
    weights = pd.Series(range(1, period + 1))
    return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()


def wma(df, intervals, col_name="close", hma_step=0):

    """
    Momentum indicator
    """
    temp_col_count_dict = {}
    for i in intervals:
        res = df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
        if hma_step == 0:
            df['wma_' + str(i)] = res
        elif hma_step == 1:
            if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
            else:
                temp_col_count_dict['hma_wma_' + str(i)] = 0
            # after halving the periods and rounding, there may be two intervals with same value e.g.
            # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
            df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
        elif hma_step == 3:
            expr = r"^hma_[0-9]{1}"
            columns = list(df.columns)
            df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res


def hma(df, intervals, col_name="close"):
    expr = r"^wma_.*"

    if not len(list(filter(re.compile(expr).search, list(df.columns)))) > 0:
        wma(df, intervals, col_name)

    intervals_half = np.round([i / 2 for i in intervals]).astype(int)

    # step 1 = WMA for interval/2
    # this creates cols with prefix 'hma_wma_*'
    wma(df, intervals_half, col_name, 1)

    # step 2 = step 1 - WMA
    columns = list(df.columns)
    expr = r"^hma_wma.*"
    hma_wma_cols = list(filter(re.compile(expr).search, columns))
    rest_cols = [x for x in columns if x not in hma_wma_cols]
    expr = r"^wma.*"
    wma_cols = list(filter(re.compile(expr).search, rest_cols))

    df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,
                                            fill_value=0)

    # step 3 = WMA(step 2, interval = sqrt(n))
    intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
    for i, col in enumerate(hma_wma_cols):
        wma(df, [intervals_sqrt[i]], col, 3)
    df.drop(columns=hma_wma_cols, inplace=True)


def trix(df, interval, col_name="close"):
    """
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    return trend.trix(df[col_name], interval, fillna=True)


def dmi(df_ss, interval):
    """
    trend indicator
    TA gave same/wrong result
    """
    return df_ss['adx_' + str(interval) + '_ema']

    # drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
    #                 'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
    # expr1 = r'dx_\d+_ema'
    # expr2 = r'adx_\d+_ema'
    #
    # drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
    # drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
    # print("***dmi df.drop(columns=drop_columns", drop_columns)
    # df.drop(columns=drop_columns, inplace=True)
    # for column in drop_columns:
    #     print("***dmi delete", column, column in df.columns)


def cci(df, interval):
    return trend.cci(df['high'], df['low'], df['close'], interval, fillna=True)


def bb_mav(df_ss, interval, col_name="close"):
    """
    volitility indicator
    """
    return volatility.bollinger_mavg(df_ss[col_name], window=interval, fillna=True)


def cmo(df, intervals, col_name="close"):
    """
    Chande Momentum Oscillator
    As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
    CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
    range = +100 to -100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """

    def calculate_cmo(series):
        sum_gains = series[series >= 0].sum()
        sum_losses = np.abs(series[series < 0].sum())
        calc_cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        return np.round(calc_cmo, 3)

    diff = df[col_name].diff()
    return {
        'cmo_' + str(period): diff.rolling(period).apply(calculate_cmo, raw=False)
        for period in intervals
    }


# not used. on close(12,16): +3, ready to use
def macd(df):
    """
    Not used
    Same for both
    calculated for same 12 and 26 periods on close only!! Not different periods.
    creates colums macd, macds, macdh
    """
    df_ss = sdf.retype(df)
    df['macd'] = df_ss['macd']

    del df['macd_']
    del df['close_12_ema']
    del df['close_26_ema']


# not implemented. period 12,26: +1, ready to use
def ppo(df, col_name="close"):
    """
    As per https://www.investopedia.com/terms/p/ppo.asp
    uses EMA(12) and EMA(26) to calculate PPO value
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    calculated for same 12 and 26 periods only!!
    """
    df_ss = sdf.retype(df)
    df['ema_' + str(12)] = df_ss[col_name + '_' + str(12) + '_ema']
    del df['close_' + str(12) + '_ema']
    df['ema_' + str(26)] = df_ss[col_name + '_' + str(26) + '_ema']
    del df['close_' + str(26) + '_ema']
    df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100

    del df['ema_12']
    del df['ema_26']


def calculate_roc(series):
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100


def roc(df, interval, col_name="close"):
    """
    Momentum oscillator
    As per implement https://www.investopedia.com/terms/p/pricerateofchange.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    ROC = (close_price_n - close_price_(n-1) )/close_price_(n-1) * 100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """
    return df[col_name].rolling(interval + 1).apply(calculate_roc, raw=False)


def dpo(df, interval, col_name="close"):
    """
    Trend Oscillator type indicator
    """
    return trend.dpo(df[col_name], window=interval)


def kst(df, interval, col_name="close"):
    """
    Trend Oscillator type indicator
    """
    return trend.kst(df[col_name], interval)


def cmf(df, interval):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    return volume.chaikin_money_flow(df['high'],
                                     df['low'],
                                     df['close'],
                                     df['volume'],
                                     interval,
                                     fillna=True)


def force_index(df, interval):
    return volume.force_index(df['close'], df['volume'], interval, fillna=True)


def eom(df, interval):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    return volume.ease_of_movement(df['high'], df['low'], df['volume'], window=interval, fillna=True)


# not used. +1
# def get_volume_delta(df):
#     df_ss = sdf.retype(df)
#     df_ss['volume_delta']


# not used. +2 for each interval kdjk and rsv
def kdjk_rsv(df_ss, interval):
    return df_ss['kdjk_' + str(interval)]


def buy_hold_sell(df, col_name, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels with
              size = total-(window_size)+1
    """

    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan

    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = (window_begin + window_end) // 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = SELL
            elif min_index == window_middle:
                labels[window_middle] = BUY
            else:
                labels[window_middle] = HOLD

        row_counter = row_counter + 1

    return labels


def short_long_ma_crossover(df, short, long, col_name="close"):
    """
    if short = 30 and long = 90,
    Buy when 30 day MA < 90 day MA
    Sell when 30 day MA > 90 day MA

    Label code : BUY => 1, SELL => 0, HOLD => 2

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels
    """

    print(f"creating label with {short}_{long}_ma")

    def detect_crossover(diff_prev, diff):
        if diff_prev >= 0 > diff:
            # buy
            return BUY
        elif diff_prev <= 0 < diff:
            return SELL
        else:
            return HOLD

    sma(df, [short, long], col_name)
    labels = np.zeros((len(df)))
    labels[:] = np.nan
    diff = df[col_name + '_sma_' + str(short)] - df[col_name + '_sma_' + str(long)]
    diff_prev = diff.shift()
    df['diff_prev'] = diff_prev
    df['diff'] = diff

    res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
    df.drop(columns=['diff_prev', 'diff'], inplace=True)
    return res


def year_range(df, start_date=None, years=5, col_name="timestamp"):
    if not start_date:
        start_date = df.head(1).iloc[0][col_name]

    end_date = start_date + pd.offsets.DateOffset(years=years)
    df_batch = df[(df[col_name] >= start_date) & (df[col_name] <= end_date)]
    return df_batch


def calc_ibr(df):
    return (df['close'] - df['low']) / (df['high'] - df['low'])


def mean_reversion(df, col_name="close"):
    """
    strategy as described at "https://decodingmarkets.com/mean-reversion-trading-strategy"

    Label code : BUY => 2, SELL => 0, HOLD => 1

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels
    """
    rsi_smooth(df, [3], col_name)  # new column 'rsi_3' added to df
    rsi_3_series = df['rsi_3']
    ibr = calc_ibr(df)
    total_rows = len(df)
    labels = np.zeros(total_rows)
    # labels[:] = np.nan
    labels[:] = HOLD
    count = 0
    for i, rsi_3 in enumerate(rsi_3_series):
        if rsi_3 < 15:  # buy
            count = count + 1

            if 3 <= count < 8 and ibr.iloc[i] < 0.2:  # TODO implement upto 5 BUYS
                labels[i] = BUY

            if count >= 8:
                count = 0
        elif ibr.iloc[i] > 0.7:  # sell
            labels[i] = SELL
        else:
            labels[i] = HOLD

    return labels


def delta(df, interval=1, col_name="close"):
    """
    labels data based on price rise on next day
      next_day - prev_day
    ((s - s.shift()) > 0).astype(np.int)
    """
    # return ((df[col_name] - df[col_name].shift()) > 0).astype(np.int)
    return df[col_name].diff(periods=interval)


def delta_percent(df, interval=1, col_name="close"):
    """
    labels data based on price rise on next day
      next_day - prev_day
    ((s - s.shift()) > 0).astype(np.int)
    """
    return df[col_name].pct_change(periods=interval)


def get_buy_sell(df, threshold=0, buy_positive=True, non_threshold="HOLD"):
    def get_result(x):
        if x > threshold:
            return "BUY" if buy_positive else "SELL"
        elif x < -threshold:
            return "SELL" if buy_positive else "BUY"
        else:
            return non_threshold

    return df.apply(get_result)


def get_up_down(df, no_change=0):
    def get_result(x):
        if x > 0:
            return 1
        elif x == 0:
            return no_change
        else:
            return 0

    return df.apply(get_result)


def as_buy_sell(df, col_name="label", buy_positive=True):
    df[col_name] = get_buy_sell(df[col_name], buy_positive=buy_positive)


def log_change(df, interval, col_name="close"):
    return np.log(df[col_name] / df[col_name].shift(interval))


def technical_indicators(df, intervals, col_name='close'):
    df_ss = sdf.retype(df)

    cols = {}
    # get_RSI(df, col_name, intervals)  # faster but non-smoothed RSI
    cols.update(rsi_smooth(df, intervals, col_name))  # momentum
    cols.update(calc_intervals(df, intervals, "wr", williamr))  # momentum
    cols.update(calc_intervals(df, intervals, "mfi", mfi))  # momentum
    cols.update(calc_intervals(df, intervals, "roc", roc, col_name=col_name))  # momentum
    cols.update(calc_intervals(df, intervals, "cmf", cmf))  # momentum, volume ema
    cols.update(cmo(df, intervals, col_name))  # momentum
    cols.update(calc_intervals(df_ss, intervals, f"{col_name}_sma", sma, col_name=col_name))
    cols.update(calc_intervals(df_ss, intervals, "open_sma", sma, col_name="open"))
    cols.update(calc_intervals(df_ss, intervals, "ema", ema, col_name=col_name))
    df = df.assign(**cols)
    cols = {}
    wma(df, intervals, col_name)
    hma(df, intervals, col_name)
    cols.update(calc_intervals(df_ss, intervals, "trix", trix, col_name=col_name))  # trend
    cols.update(calc_intervals(df, intervals, "cci", cci))  # trend
    cols.update(calc_intervals(df, intervals, "dpo", dpo, col_name=col_name))  # trend oscillator
    cols.update(calc_intervals(df, intervals, "kst", kst, col_name=col_name))  # trend
    cols.update(calc_intervals(df_ss, intervals, "dmi", dmi))  # trend
    cols.update(calc_intervals(df_ss, intervals, "bb", bb_mav, col_name=col_name))  # volatility
    # get_psi(df, col_name, intervals)  # can't find formula
    cols.update(calc_intervals(df, intervals, "fi", force_index))  # volume
    cols.update(calc_intervals(df_ss, intervals, "kdjk", kdjk_rsv))  # ready to use, +2*len(intervals), 2 rows ---? ????
    cols.update(calc_intervals(df, intervals, "eom", eom))  # volume momentum
    # get_volume_delta(df)  # volume +1
    # calc_ibr(df)  # ready to use +1  --> no assignment, bring back with assignment
    cols.update(calc_intervals(df, intervals, "log_change", log_change, col_name=col_name))
    df = df.assign(**cols)
    return df


def graph(df, name):
    layout = go.Figure(
        data=[{
            'x': df.index,
            'open': df.open,
            'close': df.close,
            'high': df.high,
            'low': df.low,
            'type': 'candlestick',
            'name': name,
            'showlegend': True
        }],
        layout_title=name
    )

    layout.show()
