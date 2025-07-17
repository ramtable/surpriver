# Basic libraries
import os
import time
import random
import collections
import numpy as np
from os import walk
import pandas as pd
import yfinance as yf
import datetime as dt
from tqdm import tqdm
from scipy.stats import linregress
from scipy.signal import argrelextrema
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
    def __init__(self, history_to_use, data_granularity_minutes, is_save_dict, is_load_dict, dict_path, min_volume_filter, is_test, future_bars_for_testing, volatility_filter, stocks_list, data_source, ib, cup_n_handle):
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
        self.CUP_N_HANDLE = cup_n_handle

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
        #self.stocks_list = list(filter(lambda x: x == "NVDA", stocks_list))

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

                if self.DATA_GRANULARITY_MINUTES == 1:
                    period = "7d"
                    interval = str(self.DATA_GRANULARITY_MINUTES) + "m",
                elif self.CUP_N_HANDLE:
                    period = "5y"
                    interval = "1d"
                else:
                    period = "30d"
                    interval = str(self.DATA_GRANULARITY_MINUTES) + "m",

                stock_prices = yf.download(
                                tickers = symbol,
                                period = period,
                                interval = interval,
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
            barSizeSetting = '1 hour'
        elif self.CUP_N_HANDLE < 1440:
            period = "30 D"
            barSizeSetting = '1 hour'
        else:
            period = "5 Y"
            barSizeSetting = '1 day'

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
                                            barSizeSetting=barSizeSetting,
                                            whatToShow='TRADES',
                                            useRTH=True,
                                            formatDate=1)

            stock_prices = util.df(bars)
            
            # ensure that stock prices contains some data, otherwise the pandas operations below could fail
            if stock_prices is None or stock_prices.empty:
                return symbol, [], [], True

            stock_prices['date'] = pd.to_datetime(stock_prices['date'])
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
    
    def detect_cup_n_handle(self, symbol_names, historical_price_info):
        """
        Detect cup and handle patterns in the historical price data.
        """
        cup_n_handle_results = []

        # Default distance thresholds if not provided.
        distance_thresholds = {
                            'a_b': 10,  # Threshold for distance between 'a' and 'b'.
                            'b_c': 10,  # Threshold for distance between 'b' and 'c'.
                            'c_d': 10,  # Threshold for distance between 'c' and 'd'.
                            'd_e': 10   # Threshold for distance between 'd' and 'e'.
                        }

        # Default price thresholds (tuned for ~3 trading days of 5-minute samples ≈ 234 points)
        # These values are intentionally lenient to allow pattern detection in short, noisy sequences.
        # For longer datasets or real-world applications, adjust these thresholds upward
        # to better reflect meaningful movements and reduce false positives.     
        price_thresholds = {
                            'a_b': 0.005,  # drop from a to b.
                            'b_c': 0.005,  # rise from b to c.
                            'a_c': 0.005,   # diff from a to c.
                            'c_d': 0.005,  # drop from c to d.
                            'b_d': 0.005,   # rise from b to d.
                            'd_e': 0.005   # rise from d to e.
                        }
        for index in range(len(symbol_names)):
            symbol = symbol_names[index]
            df = historical_price_info[index]

            # Detect cup and handle pattern
            try:
                cups = []

                df = df.dropna().reset_index(drop=True)

                prices = df['Close'].values
                min_idx = argrelextrema(prices, np.less, order=5)[0]
                max_idx = argrelextrema(prices, np.greater, order=5)[0]

                for i in min_idx:
                    left_max = max([j for j in max_idx if j < i], default=None)
                    right_max = min([j for j in max_idx if j > i], default=None)

                    if left_max is None or right_max is None:
                        continue

                    left = prices[left_max]
                    right = prices[right_max]
                    bottom = prices[i]

                    # Check shape: symmetry and depth
                    if abs(left - right) / max(left, right) < 0.1:  # shoulders roughly equal
                        if bottom < left and bottom < right:
                            depth = min(left, right) - bottom
                            if depth / bottom > 0.05:  # cup depth ≥5%
                                cups.append((left_max, i, right_max))
                            else:
                                continue

                    valid_patterns = []
                        
                    # Iterate over the maxima and minima lists.
                    for i in range(len(max_idx) - 4):  # We need at least 5 points: a, b, c, d, e.
                        a = max_idx[i]

                        # TODO:
                        # Optimize this brute-force search by limiting the search window size between each pair (a-b, b-c, c-d, d-e).
                        # This can reduce the complexity from O(m^5) to something closer to O(m^2–m^3) by avoiding unnecessary combinations.
                        # For example:
                        #   - Look for b within a sliding window after a.
                        #   - Limit c to points close to a in price and within b+N.
                        #   - Use early exit (return True) once a valid pattern is found.

                        # Check for the corresponding 'b' (minima after 'a').
                        for b in min_idx:
                            if b > a and self.distance_is_valid(a, b, distance_thresholds, 'a_b'):
                                # Check the price difference condition immediately.
                                if not self.price_difference_is_valid(a, b, None, None, None, prices, price_thresholds):
                                    continue  # Skip if price difference is invalid.

                                # Now find the next 'c' (maxima after 'b').
                                for c in max_idx:
                                    if c > b and self.distance_is_valid(b, c, distance_thresholds, 'b_c'):
                                        # Check the price difference condition immediately.
                                        if not self.price_difference_is_valid(a, b, c, None, None, prices, price_thresholds):
                                            continue  # Skip if price difference is invalid.

                                        # Now find the next 'd' (minima after 'c').
                                        for d in min_idx:
                                            if d > c and self.distance_is_valid(c, d, distance_thresholds, 'c_d'):
                                                # Check the price difference condition immediately.
                                                if not self.price_difference_is_valid(a, b, c, d, None, prices, price_thresholds):
                                                    continue  # Skip if price difference is invalid.

                                                # Finally, check for 'e' (maxima after 'd').
                                                for e in max_idx:
                                                    if e > d and self.distance_is_valid(d, e, distance_thresholds, 'd_e'):
                                                        # Now check if all price difference conditions are met.
                                                        if self.price_difference_is_valid(a, b, c, d, e, prices, price_thresholds):
                                                            valid_patterns.append((a, b, c, d, e))

                if valid_patterns is not None and len(valid_patterns) > 0:
                    cup_n_handle_results.append((symbol, index, valid_patterns))
            except Exception as e:
                print(f"Error detecting cup and handle for {symbol}: {e}")
                continue

        return cup_n_handle_results
    
    def distance_is_valid(self, a, b, distance_thresholds, pair_name):
        """
        Check if the distance between two points is valid, given a distance threshold.
        pair_name specifies which pair is being checked: 'a_b', 'b_c', 'c_d', 'd_e'.
        """
        distance = abs(a - b)  # Distance between indices.
        if pair_name in distance_thresholds:
            return distance >= distance_thresholds[pair_name] and distance <= distance_thresholds[pair_name] * 2  # Compare against the threshold.
        return False
    
    def price_difference_is_valid(self, a, b, c, d, e, prices, price_thresholds):
        """    
        Applies constraints to validate shape of the cup and handle based on price deltas.
        """
        valid = True

        # Check price difference between a and b (b should be at least price_thresholds['a_b'] % lower than a).
        if a is not None and b is not None and prices[a] - prices[b] <= price_thresholds['a_b'] * prices[a]:
            valid = False

        # Check price difference between b and c (c should be at least price_thresholds['b_c'] % higher than b).
        if b is not None and c is not None and prices[c] - prices[b] <= price_thresholds['b_c'] * prices[b]:
            valid = False

        # Check price difference between a and c (c should be at most price_thresholds['a_c'] % higher/lower than a).
        if a is not None and c is not None and abs(prices[c] - prices[a]) >= price_thresholds['a_c'] * prices[a]:
            valid = False

        # Check price difference between c and d (d should be at least price_thresholds['c_d'] % lower than c).
        if c is not None and d is not None and prices[c] - prices[d] <= price_thresholds['c_d'] * prices[c]:
            valid = False

        # Check price difference between b and d (d should be at least price_thresholds['b_d'] % lower than b).
        if b is not None and d is not None and prices[d] - prices[b] <= price_thresholds['b_d'] * prices[b]:
            valid = False

        # Check price difference between d and e (e should be at least price_thresholds['d_e'] % higher than d).
        if d is not None and e is not None and prices[e] - prices[d] <= price_thresholds['d_e'] * prices[d]:
            valid = False

        return valid

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
