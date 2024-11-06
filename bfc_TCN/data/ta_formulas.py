import pandas as pd

def rsi(close_prices, window):
    # Calculate the daily price changes (ΔPrice)
    price_change = close_prices.diff()

    # Separate the positive and negative price changes
    up = price_change.apply(lambda x: max(0, x))
    down = price_change.apply(lambda x: abs(min(0, x)))

    # Calculate the average of N-day's up closes and down closes
    avg_up = up.rolling(window=window).mean()
    avg_down = down.rolling(window=window).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_up / avg_down

    # Calculate the RSI using the formula
    rsi = 100 - (100 / (1 + rs))

    return rsi

def roc(data_series, window):
    # Calculate the ROC as the percentage change over the specified window
    roc = (data_series / data_series.shift(window) - 1) * 100

    return roc


def cmf(data_df, window):
    # Calculate Money Flow Volume (MFV)
    typical_price = (data_df['h'] + data_df['l'] + data_df['c']) / 3
    money_flow = typical_price * data_df['v']
    
    # Calculate 20-day sum of positive and negative MFV
    positive_mfv = (money_flow * (data_df['c'] > data_df['c'].shift(1))).rolling(window=window).sum()
    negative_mfv = (money_flow * (data_df['c'] < data_df['c'].shift(1))).rolling(window=window).sum()
    
    # Calculate CMF
    cmf = positive_mfv / (positive_mfv + negative_mfv)
    
    return cmf

def adx(data_df, window):
    # Calculate True Range (TR)
    data_df['High-Low'] = data_df['h'] - data_df['l']
    data_df['High-Close'] = (data_df['h'] - data_df['c'].shift(1)).abs()
    data_df['Low-Close'] = (data_df['l'] - data_df['c'].shift(1)).abs()
    data_df['TR'] = data_df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    # Calculate Directional Movement (DM) and Directional Index (DI)
    data_df['H-L'] = data_df['h'] - data_df['l']
    data_df['H-PrevClose'] = data_df['h'] - data_df['c'].shift(1)
    data_df['PrevClose-L'] = data_df['c'].shift(1) - data_df['l']
    
    data_df['DMplus'] = (data_df['H-PrevClose'] > data_df['PrevClose-L']) & (data_df['H-PrevClose'] > 0)
    data_df['DMminus'] = (data_df['PrevClose-L'] > data_df['H-PrevClose']) & (data_df['PrevClose-L'] > 0)
    
    data_df['DIplus'] = (data_df['DMplus'] * (data_df['H-PrevClose'] / data_df['H-L'])).rolling(window=window).mean()
    data_df['DIminus'] = (data_df['DMminus'] * (data_df['PrevClose-L'] / data_df['H-L'])).rolling(window=window).mean()
    
    # Calculate DX (Directional Index)
    data_df['DX'] = (data_df['DIplus'] / data_df['DIminus']).abs()
    
    # Calculate ADX (Average Directional Index)
    adx = (data_df['DX'] * 100).rolling(window=window).mean()
    
    # Return the last ADX value (most recent)
    return adx


def slope(series, window_size):
    """
    Calculate the slope of a pandas Series over a specified window of time without using NumPy.

    Args:
    - series (pd.Series): The input pandas Series.
    - window_size (int): The size of the window for calculating the slope.

    Returns:
    - pd.Series: A new Series containing the calculated slopes.
    """
    slopes = []

    for i in range(len(series)):
        if i < window_size:
            slopes.append(None)  # None for the initial values with insufficient data
        else:
            x = list(range(window_size))
            y = list(series[i - window_size + 1:i + 1])
            n = len(x)

            sum_x = sum(x)
            sum_y = sum(y)
            sum_x_squared = sum(x[i] ** 2 for i in range(n))
            sum_xy = sum(x[i] * y[i] for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
            slopes.append(slope)

    return pd.Series(slopes, index=series.index)


def calculate_macd_line(data, short_period, long_period):
    """
    Calculate the MACD Line.

    Args:
    - data (pd.Series): The input pandas Series of closing prices.
    - short_period (int): The short-term exponential moving average period.
    - long_period (int): The long-term exponential moving average period.

    Returns:
    - pd.Series: The MACD Line.
    """
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    return macd_line

def calculate_signal_line(macd_line, signal_period):
    """
    Calculate the Signal Line (9-day EMA) for MACD.

    Args:
    - macd_line (pd.Series): The MACD Line.
    - signal_period (int): The period for the signal line (usually 9).

    Returns:
    - pd.Series: The Signal Line.
    """
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return signal_line


def calculate_macd_histogram(macd_line, signal_line):
    """
    Calculate the MACD Histogram.

    Args:
    - macd_line (pd.Series): The MACD Line.
    - signal_line (pd.Series): The Signal Line.

    Returns:
    - pd.Series: The MACD Histogram.
    """
    macd_histogram = macd_line - signal_line
    return macd_histogram

def macd(close):    
    macd_line = calculate_macd_line(close, short_period=12, long_period=26)
    signal_line = calculate_signal_line(macd_line, signal_period=9)
    macd_histogram = calculate_macd_histogram(macd_line, signal_line)
    return macd_histogram


def calculate_middle_band(data, window):
    """
    Calculate the Middle Band (Simple Moving Average).

    Args:
    - data (pd.Series): The input pandas Series of closing prices.
    - window (int): The window period for the moving average.

    Returns:
    - pd.Series: The Middle Band (Simple Moving Average).
    """
    middle_band = data.rolling(window=window).mean()
    return middle_band


def calculate_bollinger_bands(data, window, num_std_dev):
    """
    Calculate the Upper and Lower Bollinger Bands.

    Args:
    - data (pd.Series): The input pandas Series of closing prices.
    - window (int): The window period for the moving average.
    - num_std_dev (int): The number of standard deviations for the bands.

    Returns:
    - pd.Series: The Upper Bollinger Band.
    - pd.Series: The Lower Bollinger Band.
    """
    middle_band = calculate_middle_band(data, window)
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, lower_band


def bbands(close,window=20, num_std_dev=2):
    upper_band, lower_band = calculate_bollinger_bands(close, window, num_std_dev)
    middle_band = calculate_middle_band(close, window)
    return upper_band, lower_band, middle_band

def bbands_category(close,upper_band, lower_band):
    if close > upper_band:
        return "1"
    elif close < lower_band:
        return "-1"
    else:
        return "0"
    

def calculate_ema(data, window):
    """
    Calculate the Exponential Moving Average (EMA).

    Args:
    - data (pd.Series): The input pandas Series of closing prices.
    - window (int): The window period for the moving average.

    Returns:
    - pd.Series: The Exponential Moving Average.
    """
    ema = data.ewm(span=window, adjust=False).mean()
    return ema

def calculate_sma(data,window):
    sma = data.rolling(window=window).mean()
    return sma

def calculate_pct_high(data, window):
    """
    Calculate the Percentage of Highs.

    Args:
    - data (pd.DataFrame): The input pandas DataFrame of OHLCV data.
    - window (int): The window period for the calculation.

    Returns:
    - pd.Series: The Percentage of Highs.
    """
    pct_high = (data['h'].rolling(window=window).max() - data['c']) / data['c']
    return pct_high

def calculate_pct_low(data, window):
    """
    Calculate the Percentage of Lows.

    Args:
    - data (pd.DataFrame): The input pandas DataFrame of OHLCV data.
    - window (int): The window period for the calculation.

    Returns:
    - pd.Series: The Percentage of Lows.
    """
    pct_low = (data['l'].rolling(window=window).min() - data['c']) / data['c']
    return pct_low


def calculate_stddev(data, window):
    """
    Calculate the Standard Deviation.

    Args:
    - data (pd.Series): The input pandas Series of closing prices.
    - window (int): The window period for the calculation.

    Returns:
    - pd.Series: The Standard Deviation.
    """
    stddev = data.rolling(window=window).std()
    return stddev