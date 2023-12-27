from typing import Callable, Union, Dict

import pandas as pd


class Series:
    def __init__(self, context=None, start_callback=None, run_callback=None) -> None:
        super().__init__()
        self.context = context if context is not None else {}
        self.start_callback = start_callback
        self.run_callback = run_callback
        self.history = []

    def start(self):
        self.prepare_data()
        self.before_run()
        self.run()
        return self.finish()

    def prepare_data(self):
        pass

    def before_run(self):
        if self.start_callback:
            self.start_callback(self)

    def run(self):
        self.exec_run_callback()

    def exec_run_callback(self):
        if self.run_callback:
            print(self.run_callback)
            self.run_callback(self)

    def finish(self):
        pass


class PandasSeries(Series):
    def __init__(self,
                 context=None,
                 start_callback=None,
                 run_callback=None,
                 data=None,
                 run_step=1) -> None:
        super().__init__(context=context, start_callback=start_callback, run_callback=run_callback)
        self.data = data
        self._prepared_data: Dict[str, pd.DataFrame] = None
        self.run_step = run_step
        self.current_index = 0

    def get_data_names(self):
        return list(self.data.keys())

    def prepare_data(self):
        super().prepare_data()
        if isinstance(self.data, pd.DataFrame):
            self._prepared_data = {"DEFAULT": self.data}
        elif isinstance(self.data, dict):
            self._prepared_data = self.data
        else:
            raise Exception("Invalid data")

        self._prepared_data = {name: self._filter_df(df) for name, df in self._prepared_data.items()}

    def __iter__(self):
        self.current_index = -1
        self.prepare_data()
        self.before_run()
        return self

    def __next__(self):
        self.current_index = self.current_index + 1
        while self.current_index < self.num_rows():
            if self.current_index % self.run_step == 0:
                return self
            self.current_index = self.current_index + 1

        if self.current_index >= self.num_rows():
            self.finish()
            raise StopIteration

    def run(self):
        for index in range(self.num_rows()):
            self.current_index = index
            if index % self.run_step == 0:
                self.exec_run_callback()

            self.run_post()

    @property
    def current_row(self, force_dict: bool = False) -> Union[Dict, pd.Series]:
        row_dict = {key: value.iloc[self.current_index] for key, value in self._prepared_data.items()}
        if len(self._prepared_data) == 1 and not force_dict:
            return next(iter(row_dict.values()))
        else:
            return row_dict

    def run_post(self):
        pass

    def num_rows(self):
        return max([len(data) for data in self._prepared_data.values()])
        # merged_data = zip(*(df.iterrows() for df in self._prepared_data.values()))
        # return enumerate(merged_data)

    # noinspection PyMethodMayBeStatic
    def _filter_df(self, df):
        return df

    def get_value(self,
                  name: str = None,
                  offset: int = 0,
                  column: str = None,
                  value_fn: Callable = None,
                  as_dict: bool = False):
        time = self.current_index - offset
        if time < 0:
            return None

        if name is None:
            values = {name: self.get_value(name=name, offset=offset, column=column)
                      for name, value in self._prepared_data.items()}
            sorted_items = sorted(values.items(), key=lambda entry: entry[1])

            return dict(sorted_items) if as_dict else sorted_items
        else:
            df = self._prepared_data[name]
            if len(df) > time:
                if column is not None:
                    if column in df.columns:
                        return df.iloc[time][column]
                    else:
                        raise Exception(f"Column {column} is not in {df.columns}")
                elif value_fn is not None:
                    return value_fn(df.iloc[time])
                else:
                    return df.iloc[time].to_dict()
            else:
                return None

    def get_delta(self, name=None, offset=0, duration=1, column=None, as_percent=False, as_dict=False):
        if name is None:
            deltas = {name: self.get_delta(name=name,
                                           offset=offset,
                                           duration=duration,
                                           column=column,
                                           as_percent=as_percent)
                      for name, value in self._prepared_data.items()}

            sorted_items = sorted(deltas.items(), key=lambda entry: entry[1])
            return dict(sorted_items) if as_dict else sorted_items
        else:
            current_value = self.get_value(name=name, offset=offset, column=column)
            offset_value = self.get_value(name=name, offset=offset + duration, column=column)

            if current_value is None or offset_value is None:
                return 0
            else:
                # noinspection PyUnresolvedReferences
                delta = current_value - offset_value
                return delta / current_value if as_percent else delta
