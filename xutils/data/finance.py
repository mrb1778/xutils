import numpy as np
import pandas as pd
import ta.momentum as momentum
import ta.trend as trend
import ta.volatility as volatility
import ta.volume as volume
from stockstats import StockDataFrame as sdf
import xutils.core.net_utils as nu


def download_stock_date_from_alphavantage(data_path, ticker, api_key, update=False):
    return nu.download_if(
        data_path,
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&apikey=" +
        api_key +
        "&datatype=csv&symbol=" + ticker,
        update=update)


# def save_stock_data(ticker, start_date, end_date, file_name="prices.csv", field="Adj Close"):
#     """
#     Gets historical stock data of given tickers between dates
#     :param ticker: company, or companies whose data is to fetched
#     :type ticker: string or list of strings
#     :param start_date: starting date for stock prices
#     :type start_date: string of date "YYYY-mm-dd"
#     :param end_date: end date for stock prices
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

############### Technical indicators ########################


# not used
def get_rsi(df, intervals, col_name="close"):
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


def get_rsi_smooth(df, intervals, col_name="close"):
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
    prev_rsi = np.inf
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
        rsi = -1

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
    for period in intervals:
        df['rsi_' + str(period)] = np.nan
        # df['rsi_'+str(period)+'_own_1'] = np.nan
        rolling_count = 0
        res = diff.rolling(period).apply(calculate_rsi, args=(period,), raw=False)
        df['rsi_' + str(period)][1:] = res

    # df.drop(['diff'], axis = 1, inplace=True)


# not used: +1, ready to use
def get_ibr(df):
    return (df['close'] - df['low']) / (df['high'] - df['low'])


def get_williamr(df, intervals):
    """
    both libs gave same result
    Momentum indicator
    """
    # df_ss = sdf.retype(df)
    for i in intervals:
        # df['wr_'+str(i)] = df_ss['wr_'+str(i)]
        df["wr_" + str(i)] = momentum.wr(df['high'], df['low'], df['close'], i, fillna=True)


def get_mfi(df, intervals):
    """
    momentum type indicator
    """
    for i in intervals:
        df['mfi_' + str(i)] = volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], n=i, fillna=True)


def get_sma(df, intervals, col_name="close"):
    """
    Momentum indicator
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        df[col_name + '_sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
        del df[col_name + '_' + str(i) + '_sma']


def get_ema(df, intervals, col_name="close"):
    """
    Needs validation
    Momentum indicator
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        df['ema_' + str(i)] = df_ss[col_name + '_' + str(i) + '_ema']
        del df[col_name + '_' + str(i) + '_ema']
        # df["ema_"+str(intervals[0])+'_1'] = ema_indicator(df['close'], i, fillna=True)


def get_wma(df, intervals, col_name="close", hma_step=0):
    """
    Momentum indicator
    """
    def wavg(rolling_prices, period):
        weights = pd.Series(range(1, period + 1))
        return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

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
            import re
            expr = r"^hma_[0-9]{1}"
            columns = list(df.columns)
            df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res


def get_hma(df, intervals, col_name="close"):
    import re
    expr = r"^wma_.*"

    if not len(list(filter(re.compile(expr).search, list(df.columns)))) > 0:
        get_wma(df, intervals, col_name)

    intervals_half = np.round([i / 2 for i in intervals]).astype(int)

    # step 1 = WMA for interval/2
    # this creates cols with prefix 'hma_wma_*'
    get_wma(df, intervals_half, col_name, 1)

    # step 2 = step 1 - WMA
    columns = list(df.columns)
    expr = r"^hma_wma.*"
    hma_wma_cols = list(filter(re.compile(expr).search, columns))
    rest_cols = [x for x in columns if x not in hma_wma_cols]
    expr = r"^wma.*"
    wma_cols = list(filter(re.compile(expr).search, rest_cols))

    df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,
                                            fill_value=0)  # .rename(index=str, columns={"close": "col1", "rsi_6": "col2"})
    # df[0:10].copy().reset_index(drop=True).merge(temp.reset_index(drop=True), left_index=True, right_index=True)

    # step 3 = WMA(step 2, interval = sqrt(n))
    intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
    for i, col in enumerate(hma_wma_cols):
        get_wma(df, [intervals_sqrt[i]], col, 3)
    df.drop(columns=hma_wma_cols, inplace=True)


def get_trix(df, intervals, col_name="close"):
    """
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        # df['trix_'+str(i)] = df_ss['trix_'+str(i)+'_sma']
        df['trix_' + str(i)] = trend.trix(df[col_name], i, fillna=True)

    # df.drop(columns=['trix','trix_6_sma',])


def get_dmi(df, intervals):
    """
    trend indicator
    TA gave same/wrong result
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        # df['dmi_'+str(i)] = adx(df['high'], df['low'], df['close'], n=i, fillna=True)
        df['dmi_' + str(i)] = df_ss['adx_' + str(i) + '_ema']

    drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
                    'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
    # drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema',
    #                 'pdm_14', 'close_-1_s', 'tr', 'atr_14', 'pdi_14', 'pdi',
    #                 'mdm', 'mdm_14_ema', 'mdm_14', 'mdi_14', 'mdi', 'dx_14',
    #                 'dx', 'adx', 'adxr']
    expr1 = r'dx_\d+_ema'
    expr2 = r'adx_\d+_ema'
    import re
    drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
    drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
    df.drop(columns=drop_columns, inplace=True)


def get_cci(df, intervals):
    df_ss = sdf.retype(df)
    for i in intervals:
        # df['cci_'+str(i)] = df_ss['cci_'+str(i)]
        df['cci_' + str(i)] = trend.cci(df['high'], df['low'], df['close'], i, fillna=True)


def get_bb_mav(df, intervals, col_name="close"):
    """
    volitility indicator
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        df['bb_' + str(i)] = volatility.bollinger_mavg(df[col_name], n=i, fillna=True)


def get_cmo(df, intervals, col_name="close"):
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

    def calculate_cmo(series, period):
        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        sum_gains = series[series >= 0].sum()
        sum_losses = np.abs(series[series < 0].sum())
        cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        return np.round(cmo, 3)

    diff = df[col_name].diff()[1:]  # skip na
    for period in intervals:
        df['cmo_' + str(period)] = np.nan
        res = diff.rolling(period).apply(calculate_cmo, args=(period,), raw=False)
        df['cmo_' + str(period)][1:] = res


# not used. on close(12,16): +3, ready to use
def get_macd(df):
    """
    Not used
    Same for both
    calculated for same 12 and 26 periods on close only!! Not different periods.
    creates colums macd, macds, macdh
    """
    df_ss = sdf.retype(df)
    df['macd'] = df_ss['macd']
    # df['macd_'+str(i)] = macd(df['close'], fillna=True)

    del df['macd_']
    del df['close_12_ema']
    del df['close_26_ema']


# not implemented. period 12,26: +1, ready to use
def get_ppo(df, col_name="close"):
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


def get_roc(df, intervals, col_name="close"):
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
    def calculate_roc(series, period):
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

    for period in intervals:
        df['roc_' + str(period)] = np.nan
        # for 12 day period, 13th day price - 1st day price
        res = df['close'].rolling(period + 1).apply(calculate_roc, args=(period,), raw=False)
        # print(len(df), len(df[period:]), len(res))
        df['roc_' + str(period)] = res


def get_dpo(df, intervals, col_name="close"):
    """
    Trend Oscillator type indicator
    """
    for i in intervals:
        df['dpo_' + str(i)] = trend.dpo(df[col_name], n=i)


def get_kst(df, intervals, col_name="close"):
    """
    Trend Oscillator type indicator
    """
    for i in intervals:
        df['kst_' + str(i)] = trend.kst(df[col_name], i)


def get_cmf(df, intervals):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    for i in intervals:
        df['cmf_' + str(i)] = volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], i, fillna=True)


def get_force_index(df, intervals):
    for i in intervals:
        df['fi_' + str(i)] = volume.force_index(df['close'], df['volume'], 5, fillna=True)


def get_eom(df, intervals):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    for i in intervals:
        df['eom_' + str(i)] = volume.ease_of_movement(df['high'], df['low'], df['volume'], n=i, fillna=True)


# not used. +1
# def get_volume_delta(df):
#     df_ss = sdf.retype(df)
#     df_ss['volume_delta']


# not used. +2 for each interval kdjk and rsv
def get_kdjk_rsv(df, intervals):
    df_ss = sdf.retype(df)
    for i in intervals:
        df['kdjk_' + str(i)] = df_ss['kdjk_' + str(i)]


def get_buy_hold_sell(df, col_name, window_size=11):
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
                labels[window_middle] = 0
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 2

        row_counter = row_counter + 1

    return labels


def get_short_long_ma_crossover(df, short, long, col_name="close"):
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

    print("creating label with {}_{}_ma".format(short, long))

    def detect_crossover(diff_prev, diff):
        if diff_prev >= 0 > diff:
            # buy
            return 1
        elif diff_prev <= 0 < diff:
            return 0
        else:
            return 2

    get_sma(df, col_name, [short, long])
    labels = np.zeros((len(df)))
    labels[:] = np.nan
    diff = df[col_name + '_sma_' + str(short)] - df[col_name + '_sma_' + str(long)]
    diff_prev = diff.shift()
    df['diff_prev'] = diff_prev
    df['diff'] = diff

    res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
    df.drop(columns=['diff_prev', 'diff'], inplace=True)
    return res


def get_year_range(df, start_date=None, years=5, col_name="timestamp"):
    if not start_date:
        start_date = df.head(1).iloc[0][col_name]

    end_date = start_date + pd.offsets.DateOffset(years=years)
    df_batch = df[(df[col_name] >= start_date) & (df[col_name] <= end_date)]
    return df_batch


def get_mean_reversion(df, col_name="close"):
    """
    strategy as described at "https://decodingmarkets.com/mean-reversion-trading-strategy"

    Label code : BUY => 1, SELL => 0, HOLD => 2

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels
    """
    get_rsi_smooth(df, [3], col_name)  # new column 'rsi_3' added to df
    rsi_3_series = df['rsi_3']
    ibr = get_ibr(df)
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    count = 0
    for i, rsi_3 in enumerate(rsi_3_series):
        if rsi_3 < 15:  # buy
            count = count + 1

            if 3 <= count < 8 and ibr.iloc[i] < 0.2:  # TODO implement upto 5 BUYS
                labels[i] = 1

            if count >= 8:
                count = 0
        elif ibr.iloc[i] > 0.7:  # sell
            labels[i] = 0
        else:
            labels[i] = 2

    return labels


def add_technical_indicators(df, col_name, intervals):
    # get_RSI(df, col_name, intervals)  # faster but non-smoothed RSI
    get_rsi_smooth(df, intervals, col_name)  # momentum
    get_williamr(df, intervals)  # momentum
    get_mfi(df, intervals)  # momentum
    # get_macd(df, col_name, intervals)  # momentum, ready to use +3
    # get_ppo(df, col_name, intervals)  # momentum, ready to use +1
    get_roc(df, intervals, col_name)  # momentum
    get_cmf(df, intervals)  # momentum, volume ema
    get_cmo(df, intervals, col_name)  # momentum
    get_sma(df, intervals, col_name)
    get_sma(df, intervals, 'open')
    get_ema(df, intervals, col_name)
    get_wma(df, intervals, col_name)
    get_hma(df, intervals, col_name)
    get_trix(df, intervals, col_name)  # trend
    get_cci(df, intervals)  # trend
    get_dpo(df, intervals, col_name)  # trend oscillator
    get_kst(df, intervals, col_name)  # trend
    get_dmi(df, intervals)  # trend
    get_bb_mav(df, intervals, col_name)  # volatility
    # get_psi(df, col_name, intervals)  # can't find formula
    get_force_index(df, intervals)  # volume
    get_kdjk_rsv(df, intervals)  # ready to use, +2*len(intervals), 2 rows
    get_eom(df, intervals)  # volume momentum
    # get_volume_delta(df)  # volume +1
    get_ibr(df)  # ready to use +1


def add_price_rise(df, col_name="close"):
    """
    labels data based on price rise on next day
      next_day - prev_day
    ((s - s.shift()) > 0).astype(np.int)
    """
    df["labels"] = ((df[col_name] - df[col_name].shift()) > 0).astype(np.int)
    df = df[1:]
    df.reset_index(drop=True, inplace=True)
