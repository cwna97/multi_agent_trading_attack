from typing import Annotated
import pandas as pd
import os
from .config import DATA_DIR, get_config
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
from .reddit_utils import fetch_top_from_category
from tqdm import tqdm
import glob


def get_local_data_dir():
    """Get the local_data directory path.

    Supports custom local_data_path in config for adversarial testing.
    """
    config = get_config()

    # Check if custom local_data_path is specified (for adversarial testing)
    if "local_data_path" in config and config["local_data_path"]:
        return config["local_data_path"]

    # Default behavior
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")

def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )

def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data

def get_finnhub_news(
    query: Annotated[str, "Search query or ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    """
    Retrieve news about a company within a time frame

    Args
        query (str): Search query or ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    result = get_data_in_range(query, start_date, end_date, "news_data", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {query} News, from {start_date} to {end_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=15)  # Default 15 days lookback
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=15)  # Default 15 days lookback
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )

def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, period=None):
    """
    Gets finnhub data saved and processed on disk.
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        data_type (str): Type of data from finnhub to fetch. Can be insider_trans, SEC_filings, news_data, insider_senti, or fin_as_reported.
        data_dir (str): Directory where the data is saved.
        period (str): Default to none, if there is a period specified, should be annual or quarterly.
    """

    if period:
        data_path = os.path.join(
            data_dir,
            "finnhub_data",
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, "finnhub_data", data_type, f"{ticker}_data_formatted.json"
        )

    data = open(data_path, "r")
    data = json.load(data)

    # filter keys (date, str in format YYYY-MM-DD) by the date range (str, str in format YYYY-MM-DD)
    filtered_data = {}
    for key, value in data.items():
        if start_date <= key <= end_date and len(value) > 0:
            filtered_data[key] = value
    return filtered_data

def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_reddit_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back (default 7)
        limit: Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing the latest news articles posts on reddit
    """

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from before to curr_date
    curr_iter_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (curr_date_dt - curr_iter_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {curr_date}", total=total_iterations)

    while curr_iter_date <= curr_date_dt:
        curr_date_str = curr_iter_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            limit,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_iter_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    query: Annotated[str, "Search query or ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        query: Search query or ticker symbol
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news articles posts on reddit
    """

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = start_date_dt

    total_iterations = (end_date_dt - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {query} from {start_date} to {end_date}",
        total=total_iterations,
    )

    while curr_date <= end_date_dt:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            10,  # max limit per day
            query,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{query} News Reddit, from {start_date} to {end_date}:\n\n{news_str}"


# ===== Local Data Functions (from prefetched local_data directory) =====

def get_local_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> pd.DataFrame:
    """
    Get stock data from local_data directory.

    Args:
        symbol: Ticker symbol
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        DataFrame with OHLCV data
    """
    local_dir = get_local_data_dir()
    price_dir = os.path.join(local_dir, "market_data", "price_data")

    # Find matching file using glob pattern
    pattern = os.path.join(price_dir, f"{symbol}-YFin-data-*.csv")
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No local stock data found for {symbol} in {price_dir}")

    # Use the most recent file if multiple exist
    data_file = sorted(matching_files)[-1]
    data = pd.read_csv(data_file)

    # Ensure Date column is string for comparison
    if "Date" in data.columns:
        data["DateOnly"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Date column not found in {data_file}")

    # Filter by date range
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ].copy()

    filtered_data = filtered_data.drop("DateOnly", axis=1)
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def get_local_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get (or 'all' for all indicators)"],
    curr_date: Annotated[str, "current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Get technical indicators from local_data directory.

    Args:
        symbol: Ticker symbol
        indicator: Specific indicator name or 'all' for all indicators
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back

    Returns:
        String with indicator data
    """
    local_dir = get_local_data_dir()
    indicators_file = os.path.join(local_dir, "indicators", f"{symbol}_indicators.json")

    if not os.path.exists(indicators_file):
        raise FileNotFoundError(f"No local indicators found for {symbol}")

    with open(indicators_file, 'r') as f:
        all_indicators = json.load(f)

    # Calculate date range
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # Indicator descriptions
    indicator_descriptions = {
        "close_50_sma": "50 SMA: A medium-term trend indicator.",
        "close_200_sma": "200 SMA: A long-term trend benchmark.",
        "close_10_ema": "10 EMA: A responsive short-term average.",
        "macd": "MACD: Computes momentum via differences of EMAs.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line.",
        "macdh": "MACD Histogram: Shows the gap between MACD and signal.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions.",
        "boll": "Bollinger Middle: A 20 SMA basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: Signals potential overbought conditions.",
        "boll_lb": "Bollinger Lower Band: Indicates potential oversold conditions.",
        "atr": "ATR: Averages true range to measure volatility.",
        "vwma": "VWMA: A moving average weighted by volume.",
        "mfi": "MFI: Money Flow Index, a volume-weighted RSI."
    }

    # Filter by date range
    filtered_data = {}
    for date_str, indicators_data in all_indicators.items():
        if start_date <= date_str <= curr_date:
            filtered_data[date_str] = indicators_data

    if not filtered_data:
        return f"No indicator data found for {symbol} from {start_date} to {curr_date}"

    # Return specific indicator or all
    if indicator and indicator.lower() != 'all':
        result_lines = [f"## {indicator.upper()} values for {symbol} from {start_date} to {curr_date}:\n"]
        for date_str in sorted(filtered_data.keys()):
            ind_data = filtered_data[date_str]
            if indicator in ind_data and ind_data[indicator] is not None:
                result_lines.append(f"{date_str}: {ind_data[indicator]:.4f}")
        if indicator in indicator_descriptions:
            result_lines.append(f"\n\n{indicator_descriptions[indicator]}")
        return "\n".join(result_lines)
    else:
        result_lines = [f"## Technical Indicators for {symbol} from {start_date} to {curr_date}:\n"]
        for date_str in sorted(filtered_data.keys()):
            ind_data = filtered_data[date_str]
            result_lines.append(f"\n### {date_str}")
            for ind_name, ind_value in ind_data.items():
                if ind_value is not None:
                    result_lines.append(f"  {ind_name}: {ind_value:.4f}")
        return "\n".join(result_lines)


def _filter_financial_by_date(df: pd.DataFrame, curr_date: str, report_lag_days: int = 45) -> pd.DataFrame:
    """
    Filter financial statement columns by curr_date.

    yfinance financial statements have report dates as columns (e.g., 2024-09-30).
    We assume reports are published ~45 days after the report date (conservative estimate).
    This prevents data leakage by only showing data that would have been available at curr_date.

    Args:
        df: DataFrame with report dates as columns
        curr_date: Current trading date yyyy-mm-dd
        report_lag_days: Days after report date when data becomes available (default 45)

    Returns:
        Filtered DataFrame with only columns available at curr_date
    """
    if curr_date is None:
        return df

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")

    # Filter columns (report dates) that would be published before curr_date
    valid_columns = []
    for col in df.columns:
        try:
            # Column names are dates like "2024-09-30"
            report_date = datetime.strptime(str(col)[:10], "%Y-%m-%d")
            # Assume data is available report_lag_days after report date
            publish_date = report_date + relativedelta(days=report_lag_days)
            if publish_date <= curr_date_dt:
                valid_columns.append(col)
        except (ValueError, TypeError):
            # Keep non-date columns (like index)
            continue

    if not valid_columns:
        return pd.DataFrame()

    return df[valid_columns]


def get_local_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual / quarterly"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
) -> str:
    """
    Get balance sheet from local_data directory.
    Filters by curr_date to prevent data leakage.
    """
    local_dir = get_local_data_dir()
    fund_dir = os.path.join(local_dir, "fundamental_data", ticker)

    filename = f"balance_sheet_{freq}.csv"
    filepath = os.path.join(fund_dir, filename)

    if not os.path.exists(filepath):
        return f"No local balance sheet data found for {ticker} ({freq})"

    df = pd.read_csv(filepath, index_col=0)

    # Filter by curr_date to prevent data leakage
    df = _filter_financial_by_date(df, curr_date)

    if df.empty:
        return f"No balance sheet data available for {ticker} ({freq}) before {curr_date}"

    # Format as string
    result = f"## {freq.capitalize()} Balance Sheet for {ticker} (as of {curr_date}):\n\n"
    result += df.to_string()

    return result


def get_local_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual / quarterly"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
) -> str:
    """
    Get cash flow statement from local_data directory.
    Filters by curr_date to prevent data leakage.
    """
    local_dir = get_local_data_dir()
    fund_dir = os.path.join(local_dir, "fundamental_data", ticker)

    filename = f"cashflow_{freq}.csv"
    filepath = os.path.join(fund_dir, filename)

    if not os.path.exists(filepath):
        return f"No local cash flow data found for {ticker} ({freq})"

    df = pd.read_csv(filepath, index_col=0)

    # Filter by curr_date to prevent data leakage
    df = _filter_financial_by_date(df, curr_date)

    if df.empty:
        return f"No cash flow data available for {ticker} ({freq}) before {curr_date}"

    result = f"## {freq.capitalize()} Cash Flow Statement for {ticker} (as of {curr_date}):\n\n"
    result += df.to_string()

    return result


def get_local_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual / quarterly"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
) -> str:
    """
    Get income statement from local_data directory.
    Filters by curr_date to prevent data leakage.
    """
    local_dir = get_local_data_dir()
    fund_dir = os.path.join(local_dir, "fundamental_data", ticker)

    filename = f"income_statement_{freq}.csv"
    filepath = os.path.join(fund_dir, filename)

    if not os.path.exists(filepath):
        return f"No local income statement data found for {ticker} ({freq})"

    df = pd.read_csv(filepath, index_col=0)

    # Filter by curr_date to prevent data leakage
    df = _filter_financial_by_date(df, curr_date)

    if df.empty:
        return f"No income statement data available for {ticker} ({freq}) before {curr_date}"

    result = f"## {freq.capitalize()} Income Statement for {ticker} (as of {curr_date}):\n\n"
    result += df.to_string()

    return result


def get_local_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
) -> str:
    """
    Get insider transactions from local_data directory.
    Filters by curr_date to avoid data leakage - only returns transactions
    that occurred on or before curr_date.
    """
    local_dir = get_local_data_dir()
    fund_dir = os.path.join(local_dir, "fundamental_data", ticker)

    filepath = os.path.join(fund_dir, "insider_transactions.csv")

    if not os.path.exists(filepath):
        return f"No local insider transactions data found for {ticker}"

    df = pd.read_csv(filepath)

    # Filter by curr_date to prevent data leakage
    # Insider transactions are filed with SEC and become public on filing date
    # "Start Date" column contains the transaction date
    if curr_date and "Start Date" in df.columns:
        df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
        curr_date_dt = pd.to_datetime(curr_date)
        df = df[df["Start Date"] <= curr_date_dt]
        df["Start Date"] = df["Start Date"].dt.strftime("%Y-%m-%d")

    if df.empty:
        return f"No insider transactions found for {ticker} on or before {curr_date}"

    result = f"## Insider Transactions for {ticker} (as of {curr_date}):\n\n"
    result += df.to_string()

    return result


def get_local_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
) -> str:
    """
    Get company fundamentals from local_data directory.
    Combines balance sheet, cash flow, and income statement data.
    Filters by curr_date to prevent data leakage.
    """
    results = []

    # Get balance sheet (quarterly)
    balance_sheet = get_local_balance_sheet(ticker, "quarterly", curr_date)
    if not balance_sheet.startswith("No"):
        results.append(balance_sheet)

    # Get cash flow (quarterly)
    cashflow = get_local_cashflow(ticker, "quarterly", curr_date)
    if not cashflow.startswith("No"):
        results.append(cashflow)

    # Get income statement (quarterly)
    income = get_local_income_statement(ticker, "quarterly", curr_date)
    if not income.startswith("No"):
        results.append(income)

    if not results:
        return f"No fundamental data available for {ticker} before {curr_date}"

    header = f"## Company Fundamentals for {ticker} (as of {curr_date}):\n\n"
    return header + "\n\n".join(results)


def get_local_news(
    query: Annotated[str, "ticker symbol or search query"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Get news from local_data directory.
    Prioritizes Alpha Vantage data (news_data/alpha_vantage/) which includes
    sentiment scores and better summaries.

    If config["poisoned_news"] is True, uses news_data_poisoned/ directory instead.

    Args:
        query: Ticker symbol or search query
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string with news articles
    """
    local_dir = get_local_data_dir()
    config = get_config()

    # Check if poisoned news should be used (for adversarial testing)
    use_poisoned = config.get("poisoned_news", False)

    if use_poisoned:
        # Use poisoned news data directory
        poisoned_file = os.path.join(local_dir, "news_data_poisoned", f"{query}_news.json")
        if os.path.exists(poisoned_file):
            news_file = poisoned_file
            source_type = "poisoned"
        else:
            return f"No poisoned news data found for {query}"
    else:
        # news_source config로 서브디렉토리 선택 (기본: alpha_vantage)
        news_source = config.get("news_source", "alpha_vantage")
        source_file = os.path.join(local_dir, "news_data", news_source, f"{query}_news.json")
        legacy_file = os.path.join(local_dir, "news_data", f"{query}_news.json")

        if os.path.exists(source_file):
            news_file = source_file
            source_type = news_source
        elif os.path.exists(legacy_file):
            news_file = legacy_file
            source_type = "legacy"
        else:
            return f"No local news data found for {query}"

    with open(news_file, 'r', encoding='utf-8') as f:
        all_news = json.load(f)

    # Filter by date range
    filtered_news = []
    for date_str, articles in all_news.items():
        if start_date <= date_str <= end_date:
            for article in articles:
                article["date"] = date_str
                filtered_news.append(article)

    if not filtered_news:
        return f"No news found for {query} from {start_date} to {end_date}"

    # Format as string
    result_lines = [f"## {query} News from {start_date} to {end_date}:\n"]

    for article in filtered_news:
        result_lines.append(f"\n### {article.get('headline', 'No Title')} ({article.get('date', 'Unknown date')})")

        # Use full content if available, otherwise fall back to summary
        if article.get('content') and len(article.get('content', '')) > 100:
            result_lines.append(article['content'])
        elif article.get('summary'):
            result_lines.append(article['summary'])

    return "\n".join(result_lines)


def get_local_social(
    query: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Get social media posts (Reddit, X) from local_data/social_data/ directory.
    Reads {query}_social.json filtered by date range.

    If config["poisoned_social"] is True, reads from social_data_poisoned/ instead.
    """
    local_dir = get_local_data_dir()
    config = get_config()

    use_poisoned = config.get("poisoned_social", False)

    if use_poisoned:
        social_file = os.path.join(local_dir, "social_data_poisoned", f"{query}_social.json")
        if not os.path.exists(social_file):
            return f"No poisoned social data found for {query}"
    else:
        social_source = config.get("social_source", "social_data")
        social_file = os.path.join(local_dir, social_source, f"{query}_social.json")
        if not os.path.exists(social_file):
            return f"No local social data found for {query}"

    with open(social_file, 'r', encoding='utf-8') as f:
        all_posts = json.load(f)

    filtered_posts = []
    for date_str, posts in all_posts.items():
        if start_date <= date_str <= end_date:
            for post in posts:
                post_copy = dict(post)
                post_copy["date"] = date_str
                filtered_posts.append(post_copy)

    if not filtered_posts:
        return f"No social media data found for {query} from {start_date} to {end_date}"

    result_lines = [f"## {query} Social Media Posts from {start_date} to {end_date}:\n"]

    for post in filtered_posts:
        platform = post.get("platform", "unknown").upper()
        date = post.get("date", "Unknown date")
        title = post.get("title") or ""
        text = post.get("text") or ""
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        subreddit = post.get("subreddit")

        header = f"\n### [{platform}]"
        if subreddit:
            header += f" r/{subreddit}"
        header += f" ({date})"
        if title:
            header += f"\n**{title}**"
        result_lines.append(header)

        if text:
            result_lines.append(text[:500])

        result_lines.append(f"*Score: {score} | Comments: {num_comments}*")

    return "\n".join(result_lines)