import math

from xutils.data import pandas_utils as pu
from xutils.data.time_series import PandasSeries


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
        self.balance_pre_cash_out = 0
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

    def run_post(self):
        super().run_post()

        dividends = self.get_value(column=self.dividend_column)
        for asset, split_multiplier in dividends:
            if split_multiplier > 0:
                quantity_owned = self.asset_quantity(asset)
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
                quantity_owned = self.asset_quantity(asset)
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
                    asset = self.asset(asset)
                    asset["quantity"] = total_owned
                    asset["history"].append(history)
                    self.history.append(history)

    def price_of(self, name=None, offset=0, as_dict=False):
        return self.get_value(name=name,
                              offset=offset,
                              column=self.price_column,
                              as_dict=as_dict)

    def price_delta(self, name=None, offset=0, duration=1, as_percent=False, as_dict=False):
        return self.get_delta(name=name,
                              offset=offset,
                              duration=duration,
                              column=self.price_column,
                              as_percent=as_percent,
                              as_dict=as_dict)

    def asset(self, name):
        return self.assets.get(name)

    def asset_field(self, name=None, field=None, calc_fn=None, as_dict=False):
        if name is None:
            values = {name: self.asset_field(name=name, field=field, calc_fn=calc_fn)
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

    def asset_quantity(self, name=None, as_dict=False):
        return self.asset_field(name, field="quantity", as_dict=as_dict)

    def asset_value(self, name=None, as_dict=False):
        return self.asset_field(
            name,
            calc_fn=lambda asset: asset["quantity"] * self.price_of(asset["name"]),
            as_dict=as_dict)

    def asset_value_percent(self, name=None, as_dict=False):
        all_amounts = self.asset_value(as_dict=True)
        total_amount = sum(all_amounts.values())

        if name is None:
            return self.asset_field(
                name,
                calc_fn=lambda asset: 100 * (asset["quantity"] * self.price_of(asset["name"])) / total_amount,
                as_dict=as_dict
            )
        else:
            return 100 * all_amounts[name] / total_amount

    def trade(self, name, quantity, **kwargs):
        if quantity == 0:
            return False

        if self.asset_quantity(name) + quantity < 0:
            return False

        price = self.price_of(name)
        total = quantity * price

        if self.balance - total < 0:
            return False

        # print("trade: ", name, quantity, price)
        self.balance = self.balance - total
        self.balance_history.append(self.balance)

        transaction_history = {
            "type": "buy" if quantity > 0 else "sell",
            "name": name,
            "quantity": quantity,
            "price": price,
            "total": total,
            "index": self.current_index,
            **kwargs
        }
        # if self.date_column is not None:
        #     transaction_history["date"] = self.get_value(name, column=self.date_column)

        self.history.append(transaction_history)

        stored_asset = self.asset(name)
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
        stored_asset["history"].append(transaction_history)

        inventory = stored_asset["inventory"]
        if quantity > 0:
            inventory.append(transaction_history)
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
        self.balance_pre_cash_out = self.balance
        for asset, asset_details in self.assets.items():
            quantity = asset_details["quantity"]
            if quantity > 0:
                self.trade(asset, -quantity)

    @property
    def summary(self):
        return {
            "name": self.name,
            "delta_percent": 100 * (self.balance - self.initial_balance) / self.initial_balance,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "balance_pre_cash_out": self.balance_pre_cash_out,
            "balance_delta": self.balance - self.initial_balance,
            "dividends": self.dividends,
            "transactions": len(self.history),
            "buys": sum([h["type"] == "buy" for h in self.history]),
            "sells": sum([h["type"] == "sell" for h in self.history])
        }

    def finish(self):
        super().finish()
        self.cash_out()
        return self.summary


def run_conditional(series: PandasMoneySeries,
                    column=None,
                    duration=None,
                    as_percent=True,
                    value_fn=None,
                    buy_high=True,
                    buy_value=None,
                    sell_value=None,
                    cutoff=0.0,
                    quantity=1,
                    threshold=0.0):
    if duration:
        values = series.get_delta(column=column, duration=duration, as_percent=as_percent)
    elif column is not None:
        values = series.get_value(column=column, offset=0)
    elif value_fn is not None:
        values = series.get_value(value_fn=value_fn)
    else:
        raise Exception("Unknown data type")

    buys = []
    sells = []
    for key, value in values:
        if buy_value is not None or sell_value is not None:
            if value == buy_value:
                # series.trade(name=key, quantity=1, value=value)
                buys.append({"name": key, "value": value})
            elif value == sell_value:
                # series.trade(key, -1, value=value)
                sells.append({"name": key, "value": value})

        elif value > cutoff + threshold:
            # series.trade(key, 1 if buy_high else -1, value=value)
            (buys if buy_high else sells).append({"name": key, "value": value})
        elif value < cutoff - threshold:
            # series.trade(key, -1 if buy_high else 1, value=value)
            (sells if buy_high else buys).append({"name": key, "value": value})

    for tx in sells:
        series.trade(quantity=-quantity, **tx)

    for tx in buys:
        series.trade(quantity=quantity, **tx)
