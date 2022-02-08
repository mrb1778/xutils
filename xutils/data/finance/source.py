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


def download_yahoo(ticker, save_path=None, update=False):
    def _download():
        import yfinance as yf
        y_ticker = yf.Ticker(ticker)
        df = y_ticker.history(period="max")
        df.reset_index(level=0, inplace=True)
        df.rename(columns={
            "Date": "timestamp",
            "Stock Splits": "split",
            "Dividends": "dividend"
        },
            inplace=True)
        pu.lower_case_columns(df)
        print(pu.describe_data(df))
        return df

    if save_path is None:
        return _download()
    else:
        def _save_and_download(path):
            df = download_yahoo(ticker)
            df.to_csv(path, index=False)
        return fu.create_file_if(save_path,
                                 _save_and_download,
                                 update=update)
