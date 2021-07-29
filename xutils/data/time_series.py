import math
import pandas as pd
import xutils.data.pandas_utils as pu


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
        self.do_run()
        return self.finish()

    def prepare_data(self):
        pass

    def before_run(self):
        if self.start_callback:
            self.start_callback(self)

    def do_run(self):
        pass

    def run(self):
        if self.run_callback:
            self.run_callback(self)

    def finish(self):
        pass


class PandasSeries(Series):
    def __init__(self, context=None, start_callback=None, run_callback=None, data=None, run_step=1) -> None:
        super().__init__(context=context, start_callback=start_callback, run_callback=run_callback)
        self.data = data
        self._prepared_data = None

        self.current_index = 0

        self.current_row = None
        self.run_step = run_step

    def get_data_names(self):
        return list(self.data.keys())

    def prepare_data(self):
        super().prepare_data()
        if isinstance(self.data, pd.DataFrame):
            self._prepared_data = {"DEFAULT": self.data}
        else:
            self._prepared_data = self.data

        self._prepared_data = {name: self._filter_df(df) for name, df in self._prepared_data.items()}

    def do_run(self):
        # for index, row in self._iterator():
        for index in range(self.num_rows()):
            self.current_index = index
            if index % self.run_step == 0:
                self.run()

            self.do_run_post()

    def do_run_post(self):
        pass

    def num_rows(self):
        return max([len(data) for data in self._prepared_data.values()])
        # merged_data = zip(*(df.iterrows() for df in self._prepared_data.values()))
        # return enumerate(merged_data)

    # noinspection PyMethodMayBeStatic
    def _filter_df(self, df):
        return df

    def get_value(self,
                  name=None,
                  offset=0,
                  column=None,
                  value_fn=None,
                  as_dict=False):
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
                if column is not None and column in df.columns:
                    return df.iloc[time][column]
                elif value_fn is not None:
                    return value_fn(df.iloc[time])
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


class PandasMoneySeries(PandasSeries):
    def __init__(self,
                 name=None,
                 context=None,
                 start_callback=None,
                 run_callback=None,
                 run_step=1,
                 data=None,
                 balance=0,
                 price_column="price",
                 dividend_column="dividend_amount",
                 split_column="split_coefficient",
                 date_column="timestamp",
                 start_date=None,
                 end_date=None
                 ) -> None:
        super().__init__(context=context,
                         start_callback=start_callback,
                         run_callback=run_callback,
                         data=data,
                         run_step=run_step)
        self.name = name
        self.initial_balance = balance
        self.balance = self.initial_balance
        self.dividends = 0
        self.balance_history = [self.initial_balance]
        self.assets = {}

        self.price_column = price_column
        self.dividend_column = dividend_column
        self.split_column = split_column

        self.date_column = date_column
        self.start_date = start_date
        self.end_date = end_date

    def _filter_df(self, df):
        if self.date_column and (self.start_date or self.end_date):
            pu.sort(df, self.date_column, set_index=True)
            return df.loc[self.start_date:self.end_date]
            # date_query = self.date_column
            # if self.start_date:
            #     date_query = self.start_date + ' < ' + date_query
            # if self.end_date:
            #     date_query = date_query + ' < ' + self.end_date
            # return df.query(date_query)
        else:
            return super()._filter_df(df)

    def do_run_post(self):
        super().do_run_post()

        dividends = self.get_value(column=self.dividend_column)
        for asset, split_multiplier in dividends:
            if split_multiplier > 0:
                quantity_owned = self.get_asset_quantity(asset)
                if quantity_owned > 0:
                    total_dividend_amount = quantity_owned * split_multiplier
                    self.balance = self.balance + total_dividend_amount
                    self.dividends = self.dividends + total_dividend_amount
                    self.history.append({
                        "type": "dividend",
                        "name": asset,
                        "quantity": quantity_owned,
                        "price": split_multiplier,
                        "total": total_dividend_amount,
                        "index": self.current_index,
                    })

        splits = self.get_value(column=self.split_column)
        for asset, split_multiplier in splits:
            split_multiplier = math.floor(split_multiplier)
            if split_multiplier != 1:
                quantity_owned = self.get_asset_quantity(asset)
                if quantity_owned > 0:
                    total_owned = quantity_owned * split_multiplier

                    history = {
                        "type": "split",
                        "name": asset,
                        "pre_quantity": quantity_owned,
                        "post_quantity": total_owned,
                        "split": split_multiplier,
                        "index": self.current_index,
                    }
                    # print("split", name, split_multiplier, quantity_owned, total_owned)
                    asset = self.get_asset(asset)
                    asset["quantity"] = total_owned
                    asset["history"].append(history)
                    self.history.append(history)

    def get_price(self, name=None, offset=0, as_dict=False):
        return self.get_value(name=name,
                              offset=offset,
                              column=self.price_column,
                              as_dict=as_dict)

    def get_price_delta(self, name=None, offset=0, duration=1, as_percent=False, as_dict=False):
        return self.get_delta(name=name,
                              offset=offset,
                              duration=duration,
                              column=self.price_column,
                              as_percent=as_percent,
                              as_dict=as_dict)

    def get_asset(self, name):
        return self.assets.get(name)

    def get_asset_field(self, name=None, field=None, calc_fn=None, as_dict=False):
        if name is None:
            values = {name: self.get_asset_field(name=name, field=field, calc_fn=calc_fn)
                      for name in self.assets.keys()}
            sorted_items = sorted(values.items(), key=lambda entry: entry[1])
            return dict(sorted_items) if as_dict else sorted_items
        else:
            asset = self.assets.get(name)
            if asset is not None:
                if calc_fn is not None:
                    return calc_fn(asset)
                else:
                    return asset[field]
            else:
                return 0

    def get_asset_quantity(self, name=None, as_dict=False):
        return self.get_asset_field(name, field="quantity", as_dict=as_dict)

    def get_asset_value(self, name=None, as_dict=False):
        return self.get_asset_field(
            name,
            calc_fn=lambda asset: asset["quantity"] * self.get_price(asset["name"]),
            as_dict=as_dict)

    def get_asset_value_percent(self, name=None, as_dict=False):
        all_amounts = self.get_asset_value(as_dict=True)
        total_amount = sum(all_amounts.values())

        if name is None:
            return self.get_asset_field(
                name,
                calc_fn=lambda asset: 100 * (asset["quantity"] * self.get_price(asset["name"])) / total_amount,
                as_dict=as_dict
            )
        else:
            return 100 * all_amounts[name] / total_amount

    def trade(self, name, quantity):
        if quantity == 0:
            return False

        if self.get_asset_quantity(name) + quantity < 0:
            return False

        price = self.get_price(name)
        total = quantity * price

        if self.balance - total < 0:
            return False

        # print("trade: ", name, quantity, price)
        self.balance = self.balance - total
        self.balance_history.append(self.balance)

        asset_history = {
            "type": "buy" if quantity > 0 else "sell",
            "name": name,
            "quantity": quantity,
            "price": price,
            "total": total,
            "index": self.current_index,
        }
        # if self.date_column is not None:
        #     asset_history["date"] = self.get_value(name, column=self.date_column)

        self.history.append(asset_history)

        stored_asset = self.get_asset(name)
        if not stored_asset:
            stored_asset = {
                "name": name,
                "quantity": 0,
                "history": [],
                "inventory": []
            }
            self.assets[name] = stored_asset

        stored_asset = self.assets[name]
        stored_asset["quantity"] += quantity
        stored_asset["history"].append(asset_history)

        inventory = stored_asset["inventory"]
        if quantity > 0:
            inventory.append(asset_history)
        else:
            quantity_left = quantity

            while quantity_left > 0:
                oldest = inventory[0]
                oldest_quantity = oldest["quantity"]
                if oldest_quantity > quantity_left:
                    oldest["quantity"] = oldest_quantity - quantity_left
                    quantity_left = 0
                else:
                    pass

        return True

    def cash_out(self):
        # print("before cash out balance", self.balance)
        for asset, asset_details in self.assets.items():
            quantity = asset_details["quantity"]
            if quantity > 0:
                # print("cash out asset", asset, "->", quantity)
                self.trade(asset, -quantity)
        # print("after cash out balance", self.balance)

    def get_summary(self):
        return {
            "name": self.name,
            "delta_percent": 100 * (self.balance - self.initial_balance) / self.initial_balance,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "balance_delta": self.balance - self.initial_balance,
            "dividends": self.dividends,
            "transactions": len(self.history),
            "buys": sum([h["type"] == "buy" for h in self.history]),
            "sells": sum([h["type"] == "sell" for h in self.history])
        }

    def finish(self):
        super().finish()
        self.cash_out()
        return self.get_summary()
