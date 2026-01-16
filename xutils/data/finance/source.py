import pandas as pd
import yfinance as yf

from xutils.core import net_utils as nu
import xutils.core.file_utils as fu
import xutils.data.pandas_utils as pu


def download_alphavantage(path, ticker, api_key, update=False):
    return nu.download_if(
        path,
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&apikey=" +
        api_key +
        "&datatype=csv&symbol=" + ticker,
        update=update)


def download_yahoo(ticker: str,
                   save_path=None,
                   auto_adjust: bool = False,
                   update_if_older_than=7,
                   use_adj_close: bool = True,
                   force_update=False) -> pd.DataFrame:
    def _download():
        y_ticker = yf.Ticker(ticker.replace(".", "-"))
        df = y_ticker.history(
            period="max",
            auto_adjust=auto_adjust)
        # df.reset_index(level=0, inplace=True)
        df = df.reset_index()
        pu.lower_case_columns(df)
        df.rename(columns={
                "date": "timestamp",
                "stock splits": "split",
                "dividends": "dividend"
            },
            inplace=True)
        if use_adj_close:
            df.drop(columns=["close"], inplace=True)
            df.rename(columns={"adj close": "close"}, inplace=True)
        return df

    if save_path is None:
        return _download()
    else:
        def _save_and_download(path):
            df = download_yahoo(ticker)
            df.to_csv(path, index=False)

        fu.create_file_if(path=save_path,
                          create_fn=_save_and_download,
                          update_if_older_than=update_if_older_than,
                          update=force_update)
        df = pu.read(save_path, parse_dates=["timestamp"])
        df = df.reset_index()
        # df['timestamp'] = pd.to_datetime(df['timestamp'])

        # df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = pu.cast_to_timestamp_timezone(df, 'America/New_York', "timestamp")
        return df
