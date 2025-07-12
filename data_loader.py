# Basic libraries
import os
import ta
import sys
import json
import math
import time
import pickle
import random
import requests
import collections
import numpy as np
from os import walk
import pandas as pd
import yfinance as yf
import datetime as dt
from tqdm import tqdm
from scipy.stats import linregress
from datetime import datetime, timedelta
from feature_generator import TAEngine
import warnings
from binance.client import Client as Binance_Client
from dotenv import load_dotenv
from ib_insync import *
import numpy as np
import traceback
import asyncio
import aiohttp

load_dotenv()
warnings.filterwarnings("ignore")

semaphore = asyncio.Semaphore(8)  # Only x active request at a time
REQUEST_INTERVAL = 2  # seconds between requests

class DataEngine():
    def __init__(self, history_to_use, data_granularity_minutes, is_save_dict, is_load_dict, dict_path, min_volume_filter, is_test, future_bars_for_testing, volatility_filter, stocks_list, data_source, ib):
        print("Data engine has been initialized...")
        self.DATA_GRANULARITY_MINUTES = data_granularity_minutes
        self.IS_SAVE_DICT = is_save_dict
        self.IS_LOAD_DICT = is_load_dict
        self.DICT_PATH = dict_path
        self.VOLUME_FILTER = min_volume_filter
        self.FUTURE_FOR_TESTING = future_bars_for_testing
        self.IS_TEST = is_test
        self.VOLATILITY_THRESHOLD = volatility_filter
        self.DATA_SOURCE = data_source
        self.ib = ib

        if ib is None:
            self.ib = IB()

        #ibapi init
        if data_source =='ibgate':
            self.clientId = random.randint(10000, 99999)
            self.ib.connect('127.0.0.1', 4001, self.clientId, readonly=True)
        elif data_source == 'tws':
            self.clientId = random.randint(10000, 99999)
            self.ib.connect('127.0.0.1', 7496, self.clientId, readonly=True)

        # Stocks list
        self.directory_path = str(os.path.dirname(os.path.abspath(__file__)))
        self.stocks_file_path = self.directory_path + f"/stocks/{stocks_list}"
        self.stocks_list = []

        # Load stock names in a list
        self.load_stocks_from_file()

        # Load Technical Indicator engine
        self.taEngine = TAEngine(history_to_use = history_to_use)

        # Dictionary to store data. This will only store and save data if the argument is_save_dictionary is 1.
        self.features_dictionary_for_all_symbols = {}

        # Data length
        self.stock_data_length = []
        
        # Create an instance of the Binance Client with no api key and no secret (api key and secret not required for the functionality needed for this script)
        self.binance_client = Binance_Client("","")

    def tickPrice(self, reqId, tickType, price, attrib):
                if tickType == 4:  # Last price
                    self.latest_prices[reqId] = price

    def load_stocks_from_file(self):
        """
        Load stock names from the file
        """
        print("Loading all stocks from file...")
        stocks_list = open(self.stocks_file_path, "r").readlines()
        stocks_list = [str(item).strip("\n") for item in stocks_list]

        # Load symbols
        stocks_list = list(sorted(set(stocks_list)))
        print("Total number of stocks: %d" % len(stocks_list))
        self.stocks_list = stocks_list

    def get_most_frequent_key(self, input_list):
        counter = collections.Counter(input_list)
        counter_keys = list(counter.keys())
        frequent_key = counter_keys[0]
        return frequent_key
    
    def get_data(self, symbol):
        """
        Get stock data.
        """

        # Find period
        if self.DATA_GRANULARITY_MINUTES == 1:
            period = "7d"
        else:
            period = "30d"

        try:
            # get crytpo price from Binance
            if(self.DATA_SOURCE == 'binance'):
                # Binance clients doesn't like 60m as an interval
                if(self.DATA_GRANULARITY_MINUTES == 60):
                    interval = '1h'
                else:
                    interval = str(self.DATA_GRANULARITY_MINUTES) + "m"
                stock_prices = self.binance_client.get_klines(symbol=symbol, interval = interval)
                # ensure that stock prices contains some data, otherwise the pandas operations below could fail
                if len(stock_prices) == 0:
                    return [], [], True
                # convert list to pandas dataframe
                stock_prices = pd.DataFrame(stock_prices, columns=['Datetime', 'Open', 'High', 'Low', 'Close',
                                             'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                stock_prices['Datetime'] = stock_prices['Datetime'].astype(float)
                stock_prices['Open'] = stock_prices['Open'].astype(float)
                stock_prices['High'] = stock_prices['High'].astype(float)
                stock_prices['Low'] = stock_prices['Low'].astype(float)
                stock_prices['Close'] = stock_prices['Close'].astype(float)
                stock_prices['Volume'] = stock_prices['Volume'].astype(float)
               
            # get stock prices from yahoo finance
            else:
                stock_prices = yf.download(
                                tickers = symbol,
                                period = period,
                                interval = str(self.DATA_GRANULARITY_MINUTES) + "m",
                                auto_adjust = False,
                                progress=False)
            stock_prices = stock_prices.reset_index()
            stock_prices = stock_prices[['Datetime','Open', 'High', 'Low', 'Close', 'Volume']]
            data_length = len(stock_prices.values.tolist())
            self.stock_data_length.append(data_length)

            # After getting some data, ignore partial data based on number of data samples
            if len(self.stock_data_length) > 5:
                most_frequent_key = self.get_most_frequent_key(self.stock_data_length)
                if data_length != most_frequent_key:
                    return [], [], True

            if self.IS_TEST == 1:
                stock_prices_list = stock_prices.values.tolist()
                stock_prices_list = stock_prices_list[1:]  # For some reason, yfinance gives some 0 values in the first index
                future_prices_list = stock_prices_list[-(self.FUTURE_FOR_TESTING + 1):]
                historical_prices = stock_prices_list[:-self.FUTURE_FOR_TESTING]
                historical_prices = pd.DataFrame(historical_prices)
                historical_prices.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume']
            else:
                # No testing
                stock_prices_list = stock_prices.values.tolist()
                stock_prices_list = stock_prices_list[1:]
                historical_prices = pd.DataFrame(stock_prices_list)
                historical_prices.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume']
                future_prices_list = []

            if len(stock_prices.values.tolist()) == 0:
                return [], [], True
        except Exception as e: 
            # print(e)
            # traceback.print_exc()
            return [], [], True

        return historical_prices, future_prices_list, False

    async def get_data_async(self, symbol, session):
        """
        Get stock data.
        """

        # Find period
        if self.DATA_GRANULARITY_MINUTES == 1:
            period = "7 D"
        else:
            period = "30 D"

        try:
            if(self.DATA_GRANULARITY_MINUTES == 60):
                interval = '1 hour'
            else:
                interval = str(self.DATA_GRANULARITY_MINUTES) + " mins"
            contract = Stock(symbol)
            contract.exchange = 'SMART'
            contract.currency = 'USD'

            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(contract, endDateTime="", durationStr=period,
                                            barSizeSetting='1 hour',
                                            whatToShow='TRADES',
                                            useRTH=True,
                                            formatDate=1)

            stock_prices = util.df(bars)
            
            # ensure that stock prices contains some data, otherwise the pandas operations below could fail
            if stock_prices is None or stock_prices.empty:
                return symbol, [], [], True

            stock_prices['date'] = stock_prices['date'].dt.tz_convert('America/New_York')
            stock_prices['open'] = stock_prices['open'].astype(float)
            stock_prices['high'] = stock_prices['high'].astype(float)
            stock_prices['low'] = stock_prices['low'].astype(float)
            stock_prices['close'] = stock_prices['close'].astype(float)
            stock_prices['volume'] = stock_prices['volume'].astype(float)

            stock_prices = stock_prices.rename(columns={'date': 'Datetime', 'open': 'Open', 'high': 'High',
                                                        'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
        
            stock_prices = stock_prices.reset_index()
            stock_prices = stock_prices[['Datetime','Open', 'High', 'Low', 'Close', 'Volume']]
            data_length = len(stock_prices.values.tolist())
            self.stock_data_length.append(data_length)

            # After getting some data, ignore partial data based on number of data samples
            if len(self.stock_data_length) > 5:
                most_frequent_key = self.get_most_frequent_key(self.stock_data_length)
                if data_length != most_frequent_key:
                    return symbol, [], [], True

            if self.IS_TEST == 1:
                stock_prices_list = stock_prices.values.tolist()
                stock_prices_list = stock_prices_list[1:]  # For some reason, yfinance gives some 0 values in the first index
                future_prices_list = stock_prices_list[-(self.FUTURE_FOR_TESTING + 1):]
                historical_prices = stock_prices_list[:-self.FUTURE_FOR_TESTING]
                historical_prices = pd.DataFrame(historical_prices)
                historical_prices.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume']
            else:
                # No testing
                stock_prices_list = stock_prices.values.tolist()
                stock_prices_list = stock_prices_list[1:]
                historical_prices = pd.DataFrame(stock_prices_list)
                historical_prices.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume']
                future_prices_list = []

            if len(stock_prices.values.tolist()) == 0:
                return symbol, [], [], True
        except Exception as e: 
            print(e)
            traceback.print_exc()
            return symbol, [], [], True

        return symbol, historical_prices, future_prices_list, False

    def calculate_volatility(self, stock_price_data):
        CLOSE_PRICE_INDEX = 4
        stock_price_data_list = stock_price_data.values.tolist()
        close_prices = [float(item[CLOSE_PRICE_INDEX]) for item in stock_price_data_list]
        close_prices = [item for item in close_prices if item != 0]
        volatility = np.std(close_prices)
        return volatility

    def collect_data_for_all_tickers(self):
        """
        Iterates over all symbols and collects their data
        """

        print("Loading data for all stocks...")
        features = []
        symbol_names = []
        historical_price_info = []
        future_price_info = []

         # Any stock with very low volatility is ignored. You can change this line to address that.
        for i in tqdm(range(len(self.stocks_list))):
            symbol = self.stocks_list[i]
            try:
                stock_price_data, future_prices, not_found = self.get_data(symbol)
                    
                if not not_found:
                    volatility = self.calculate_volatility(stock_price_data)

                    # Filter low volatility stocks
                    if volatility < self.VOLATILITY_THRESHOLD:
                        continue
                        
                    features_dictionary = self.taEngine.get_technical_indicators(stock_price_data)
                    feature_list = self.taEngine.get_features(features_dictionary)

                    # Add to dictionary
                    self.features_dictionary_for_all_symbols[symbol] = {"features": features_dictionary, "current_prices": stock_price_data, "future_prices": future_prices}

                    # Save dictionary after every 100 symbols
                    if len(self.features_dictionary_for_all_symbols) % 100 == 0 and self.IS_SAVE_DICT == 1:
                        np.save(self.DICT_PATH, self.features_dictionary_for_all_symbols)

                    if np.isnan(feature_list).any() == True:
                        continue

                    # Check for volume
                    average_volume_last_30_tickers = np.mean(list(stock_price_data["Volume"])[-30:])
                    if average_volume_last_30_tickers < self.VOLUME_FILTER:
                        continue

                    # Add to lists
                    features.append(feature_list)
                    symbol_names.append(symbol)
                    historical_price_info.append(stock_price_data)
                    future_price_info.append(future_prices)

            except Exception as e:
                print("Exception", e)
                continue

        # Sometimes, there are some errors in feature generation or price extraction, let us remove that stuff
        features, historical_price_info, future_price_info, symbol_names = self.remove_bad_data(features, historical_price_info, future_price_info, symbol_names)

        return features, historical_price_info, future_price_info, symbol_names

    async def collect_data_for_all_tickers_async(self):
        print("Loading data for all stocks (async)...")

        features = []
        symbol_names = []
        historical_price_info = []
        future_price_info = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.throttled_fetch(symbol, session) for symbol in self.stocks_list
            ]

        results = await asyncio.gather(*tasks)

        try:
            for result in results:
                if result is None:
                    continue

                symbol = result[0]
                stock_price_data = result[1]
                future_prices = result[2]
                not_found = result[3]

                if not_found:
                    continue

                volatility = self.calculate_volatility(stock_price_data)

                if volatility < self.VOLATILITY_THRESHOLD:
                    continue

                features_dictionary = self.taEngine.get_technical_indicators(stock_price_data)
                feature_list = self.taEngine.get_features(features_dictionary)

                if np.isnan(feature_list).any():
                    continue

                avg_vol = np.mean(stock_price_data["Volume"].tail(30))
                if avg_vol < self.VOLUME_FILTER:
                    continue

                self.features_dictionary_for_all_symbols[symbol] = {
                    "features": features_dictionary,
                    "current_prices": stock_price_data,
                    "future_prices": future_prices
                }

                # Saving
                features.append(feature_list)
                symbol_names.append(symbol)
                historical_price_info.append(stock_price_data)
                future_price_info.append(future_prices)

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            return None, None, None, None

        features, historical_price_info, future_price_info, symbol_names = self.remove_bad_data(
                features, historical_price_info, future_price_info, symbol_names)

        return features, historical_price_info, future_price_info, symbol_names


    def load_data_from_dictionary(self):
        # Load data from dictionary
        print("Loading data from dictionary")
        dictionary_data = np.load(self.DICT_PATH, allow_pickle = True).item()
        
        features = []
        symbol_names = []
        historical_price_info = []
        future_price_info = []
        for symbol in dictionary_data:
            feature_list = self.taEngine.get_features(dictionary_data[symbol]["features"])
            current_prices = dictionary_data[symbol]["current_prices"]
            future_prices = dictionary_data[symbol]["future_prices"]
            
            # Check if there is any null value
            if np.isnan(feature_list).any() == True:
                continue

            features.append(feature_list)
            symbol_names.append(symbol)
            historical_price_info.append(current_prices)
            future_price_info.append(future_prices)

        # Sometimes, there are some errors in feature generation or price extraction, let us remove that stuff
        features, historical_price_info, future_price_info, symbol_names = self.remove_bad_data(features, historical_price_info, future_price_info, symbol_names)

        return features, historical_price_info, future_price_info, symbol_names

    def remove_bad_data(self, features, historical_price_info, future_price_info, symbol_names):
        """
        Remove bad data i.e data that had some errors while scraping or feature generation
        """
        length_dictionary = collections.Counter([len(feature) for feature in features])
        length_dictionary = list(length_dictionary.keys())
        if length_dictionary == []:
            return [], [], [], []
        else:
            most_common_length = length_dictionary[0]

        filtered_features, filtered_historical_price, filtered_future_prices, filtered_symbols = [], [], [], []
        for i in range(0, len(features)):
            if len(features[i]) == most_common_length:
                filtered_features.append(features[i])
                filtered_symbols.append(symbol_names[i])
                filtered_historical_price.append(historical_price_info[i])
                filtered_future_prices.append(future_price_info[i])

        return filtered_features, filtered_historical_price, filtered_future_prices, filtered_symbols

    async def throttled_fetch(self, symbol, session):
        async with semaphore:
            print(f"[{symbol}] Requesting at", time.strftime("%X"))
            symbol, stock_price_data, future_prices, not_found = await self.get_data_async(symbol, session)  # your IB async fetch
            await asyncio.sleep(REQUEST_INTERVAL)  # Enforce spacing
            return symbol, stock_price_data, future_prices, not_found 
