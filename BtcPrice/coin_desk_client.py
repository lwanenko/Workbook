import requests
from datetime import date, timedelta

coindeskApi = " "


def get_currencies():
    coindesk_currencies = requests.get(coindeskApi + "supported-currencies.json").json()
    currency_list = []
    for currency in coindesk_currencies:
        currency_list.append(currency["currency"])
    return currency_list


def get_current_price(currency):
    return float(requests.get(coindeskApi + "currentprice/" + currency + ".json").json()["bpi"][currency]["rate"]
                 .replace(",", ""))


def get_historical_price(
        currency,
        start=date.today()-timedelta(days=2000),
        end=date.today()
        ):
    start_string = f"{start.year:#02d}-{start.month:#02d}-{start.day:#02d}"
    end_string = f"{end.year:#02d}-{end.month:#02d}-{end.day:#02d}"
    return requests.get(f"{coindeskApi}historical/close.json?"
                        f"currency={currency}&start={start_string}&end={end_string}") \
                        .json()["bpi"]
