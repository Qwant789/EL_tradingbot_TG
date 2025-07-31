# Alpaca Trading Bot
#
# This script connects to an Alpaca trading account and executes a specific basic strategy daily:
# 1. At 14:00 UTC, sell all current holdings.
# 2. Immediately after, buy the top 30 US stock gainers from TradingView with $100 each.
#
# SETUP INSTRUCTIONS:
#
# 1. Sign up to alpaca trading website and quire your API keys
#
# 2. Install python then install all necessary Python libraries via the powershell terminal:
#    pip install alpaca-py python-dotenv schedule tradingview-screener pandas
#
# 3. Create a file named .env in the same directory (folder) as this script.
#    This file will securely store your Alpaca API keys on device.
#
# 4. Add your Alpaca API keys as variable=python string to the .env file like this:
#    LPCA_API_KEY_ID='YOUR ACTUAL API KEY ID'
#    LPCA_API_SECRET_KEY='YOUR ACTUAL SECRET API KEY'
#
# 5. Configure the BASE_URL in the script below to point to either your
#    paper trading account or your live trading account.
#
# 6. Remember to use the apropriate python code interpreter version 3.13.2
#
# Then you can launch it by simply runing the script or test it's trading right away by typing 
# python my_trading_bot001.py test 
# in the powershell terminal and pressing enter
#
# if you want, you can also run it as a background service in (Windows) Runing in background (it won't close when you close terminal)
# pythonw tradinbotscript011.py


import os
import time
import schedule
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

# Checking if tradingview-screener is available at startup
try:
    from tradingview_screener import Query, Column
    TRADINGVIEW_AVAILABLE = True
    print("✅ TradingView screener library loaded successfully")
except ImportError:
    TRADINGVIEW_AVAILABLE = False
    print("❌ Warning: tradingview-screener library not found. Install with: pip install tradingview-screener")

# --- CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Your Alpaca API credentials
API_KEY = os.getenv('LPCA_API_KEY_ID')
SECRET_KEY = os.getenv('LPCA_API_SECRET_KEY')

# Specify paper or live trading endpoint
# Paper trading: "https://paper-api.alpaca.markets/v2"
# Live trading: "https://api.alpaca.markets"
BASE_URL = "https://paper-api.alpaca.markets/v2" 

# --- ALPACA API CLIENT ---

# Instantiate the trading client
try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True if 'paper' in BASE_URL else False)
    print("Successfully connected to Alpaca API.")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")
    exit()

# --- TRADING LOGIC ---

def liquidate_portfolio():
    """
    Sells all current holdings in the portfolio.
    """
    print("\n--- Starting Portfolio Liquidation ---")
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            print("Portfolio is already empty. No sell orders needed.")
            return

        print(f"Found {len(positions)} positions to liquidate.")
        for position in positions:
            try:
                # Ensure we are closing the exact quantity we hold
                print(f"Selling {position.qty} shares of {position.symbol}...")
                trading_client.close_position(position.symbol)
                print(f"Market sell order for {position.symbol} placed successfully.")
            except APIError as e:
                print(f"Error selling {position.symbol}: {e}")
        
        print("--- Portfolio Liquidation Complete ---")

    except APIError as e:
        print(f"Error fetching portfolio positions: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during liquidation: {e}")

def is_market_open():
    """
    Checking if the US stock market is likely to be open.
    This is a basic check only for first production use, consider using Alpaca's market calendar API.
    """
    now = datetime.now(timezone.utc)
    
    # Check if it's a weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Convert to EST/EDT (approximate)
    # This is simplified - in production, use proper timezone handling
    est_hour = (now.hour - 5) % 24  # Rough EST conversion
    
    # US Market is generally open 9:30 AM - 4:00 PM EST
    if 9 <= est_hour <= 16:
        return True
    
    return False
        
def get_top_tradingview_gainers():
    """
    Uses the tradingview-screener library to get the top 20 gaining US stocks.
    Returns a list of ticker symbols.
    """
    print("\n--- Identifying Top Gainers from TradingView ---")
    
    if not TRADINGVIEW_AVAILABLE:
        print("❌ TradingView screener library not available")
        return get_fallback_stocks()
    
    try:
        from tradingview_screener import Query, Column
        
        # Properly configured query for US stock gainers - removed market cap filter
        query = (Query()
                .select('name', 'close', 'change', 'change_abs', 'volume')
                .where(
                    Column('volume') > 100_000,  # Volume > 100k
                    Column('change') > 0,  # Only gainers
                    Column('type').isin(['stock']),  # Only stocks
                    Column('subtype').isin(['common']),  # Only common stocks
                    Column('exchange').isin(['NASDAQ', 'NYSE', 'AMEX'])  # US exchanges
                )
                .order_by('change', ascending=False)  # Sort by percentage change descending
                .limit(50))  # Get top 50 to have some buffer
        
        # Execute the query
        result_df = query.get_scanner_data()[1]
        
        if result_df is None or result_df.empty:
            print("Could not retrieve data from TradingView.")
            return get_fallback_stocks()
        
        print(f"Total stocks retrieved: {len(result_df)}")
        print("Sample of data structure:")
        print(result_df.head(3))
        
        # Extract valid US stock tickers - FIXED VERSION
        us_stocks = []
        
        # Check if 'ticker' column exists, otherwise use the index or try to extract from other columns
        if 'ticker' in result_df.columns:
            ticker_source = result_df['ticker']
            print("Using 'ticker' column for symbols")
        else:
            # The ticker symbols are likely in the index or we need to extract them differently
            print("Ticker column not found, checking index and other methods...")
            print("DataFrame columns:", result_df.columns.tolist())
            print("DataFrame index sample:", result_df.index[:5].tolist())
            
            # Try to get the scanner data with ticker information included
            try:
                # Re-query with explicit ticker selection
                query_with_ticker = (Query()
                        .select('ticker', 'name', 'close', 'change', 'change_abs', 'volume')
                        .where(
                            Column('volume') > 100_000,
                            Column('change') > 0,
                            Column('type').isin(['stock']),
                            Column('subtype').isin(['common']),
                            Column('exchange').isin(['NASDAQ', 'NYSE', 'AMEX'])
                        )
                        .order_by('change', ascending=False)
                        .limit(50))
                
                result_with_ticker = query_with_ticker.get_scanner_data()
                if len(result_with_ticker) > 1 and result_with_ticker[1] is not None:
                    result_df = result_with_ticker[1]
                    print("Re-queried with ticker column successfully")
                    print("New columns:", result_df.columns.tolist())
            except Exception as e:
                print(f"Re-query failed: {e}")
            
            # If we still don't have ticker column, try to extract from index or name
            if 'ticker' in result_df.columns:
                ticker_source = result_df['ticker']
            elif hasattr(result_df.index, 'str'):
                # Index might contain ticker symbols
                ticker_source = result_df.index
                print("Using index as ticker source")
            else:
                print("Cannot find ticker symbols, trying alternative method...")
                return get_alternative_gainers()
        
        # Process each stock
        for i, (idx, row) in enumerate(result_df.iterrows()):
            try:
                # Extract ticker symbol
                if 'ticker' in result_df.columns:
                    raw_ticker = str(row['ticker']).strip().upper()
                else:
                    raw_ticker = str(idx).strip().upper()
                
                # Clean the ticker symbol
                ticker = raw_ticker
                
                # Remove exchange prefix if present (e.g., "NASDAQ:AAPL" -> "AAPL")
                if ':' in ticker:
                    ticker = ticker.split(':')[-1]
                
                # Get stock data
                change_pct = float(row['change'])
                volume = float(row['volume'])
                
                print(f"Processing: {raw_ticker} -> {ticker}, Change: {change_pct:.2f}%, Vol: {volume:,.0f}")
                
                # Validate ticker format and data - removed market cap filter
                if (ticker and 
                    len(ticker) >= 1 and len(ticker) <= 5 and  # Valid ticker length
                    ticker.replace('.', '').replace('-', '').isalnum() and  # Allow dots and dashes
                    not ticker.isdigit() and  # Exclude pure numbers
                    change_pct > 0 and  # Must be a gainer
                    volume > 50000):  # Volume threshold
                    
                    us_stocks.append((ticker, change_pct, volume))
                    print(f"✅ Added {ticker} to buy list")
                    
                    if len(us_stocks) >= 20:
                        break
                else:
                    print(f"❌ Rejected {ticker}: invalid format or insufficient data")
                    
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        if len(us_stocks) == 0:
            print("No valid stocks found. Trying alternative method...")
            return get_alternative_gainers()
        
        # Sort by change percentage (highest first)
        us_stocks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"--- Top {len(us_stocks)} TradingView Gainers ---")
        for i, (ticker, change_pct, volume) in enumerate(us_stocks[:20], 1):
            print(f"{i:2d}. {ticker:5s}: +{change_pct:6.2f}% | Vol: {volume:,}")
        
        # Return just the ticker symbols
        return [stock[0] for stock in us_stocks[:20]]

    except ImportError:
        print("Error: tradingview-screener library not found. Please install it with: pip install tradingview-screener")
        return get_fallback_stocks()
        
    except Exception as e:
        print(f"Error fetching gainers from TradingView: {e}")
        print("Trying alternative method...")
        return get_alternative_gainers()

def get_alternative_gainers():
    """
    Kinda really unnecesary but saw someone adding this while building something similar. Its and Alternative method to get top gainers using a different approach.
    """
    try:
        from tradingview_screener import Query, Column
        
        print("Trying alternative TradingView query method...")
        
        # Try getting data with different column selection - removed market cap filter
        query = (Query()
                .select('close', 'change', 'volume')
                .where(
                    Column('change') > 0,
                    Column('volume') > 50000
                )
                .order_by('change', ascending=False)
                .limit(100))
        
        result_df = query.get_scanner_data()[1]
        
        if result_df is not None and not result_df.empty:
            print("Alternative method retrieved data successfully")
            print("DataFrame index sample:", result_df.index[:5].tolist())
            
            us_stocks = []
            for ticker_raw in result_df.index:
                ticker = str(ticker_raw).strip().upper()
                
                # Remove exchange prefix if present
                if ':' in ticker:
                    ticker = ticker.split(':')[-1]
                
                # Basic validation for US stock tickers
                if (ticker and 
                    len(ticker) >= 1 and len(ticker) <= 5 and
                    ticker.replace('.', '').replace('-', '').isalnum() and
                    not ticker.isdigit()):  # Exclude pure numbers
                    
                    try:
                        change_pct = result_df.loc[ticker_raw, 'change']
                        if change_pct > 0:
                            us_stocks.append((ticker, change_pct))
                            print(f"✅ Alternative method added: {ticker} (+{change_pct:.2f}%)")
                    except:
                        continue
                    
                    if len(us_stocks) >= 20:
                        break
            
            if us_stocks:
                us_stocks.sort(key=lambda x: x[1], reverse=True)
                print(f"--- Alternative Method: Top {len(us_stocks)} Gainers ---")
                for i, (ticker, change_pct) in enumerate(us_stocks[:20], 1):
                    print(f"{i:2d}. {ticker}: +{change_pct:.2f}%")
                return [stock[0] for stock in us_stocks[:20]]
            else:
                print("❌ Alternative method found no valid stocks")
        
    except Exception as e:
        print(f"Alternative method failed: {e}")
    
    return get_fallback_stocks()

def get_fallback_stocks():
    """
    Returns empty list when TradingView fails completely.
    """
    print("❌ All TradingView methods failed completely")
    return []
  
def buy_top_gainers():
    """
    Places market buy orders for the top gainers from TradingView.
    Buys whole shares with total cost as close to $100 as possible (but not over).
    """
    print("\n--- Placing Buy Orders for Top Gainers ---")
    top_stocks = get_top_tradingview_gainers()

    if not top_stocks:
        print("❌ CRITICAL FAILURE: No top gaining stocks could be retrieved from TradingView")
        print("❌ STRATEGY ABORTED: Cannot execute buy orders without valid stock data")
        return

    print(f"Found {len(top_stocks)} stocks to buy")
    successful_orders = 0
    failed_orders = 0

    for stock_symbol in top_stocks:
        try:
            # Get current stock price to calculate number of shares
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestQuoteRequest
                
                # Create data client for getting stock prices
                data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
                
                # Get latest quote for the stock
                latest_quote_request = StockLatestQuoteRequest(symbol_or_symbols=[stock_symbol])
                latest_quote = data_client.get_stock_latest_quote(latest_quote_request)
                
                # Use the ask price (or bid price as fallback)
                if stock_symbol in latest_quote:
                    quote_data = latest_quote[stock_symbol]
                    current_price = float(quote_data.ask_price) if quote_data.ask_price > 0 else float(quote_data.bid_price)
                    print(f"Current price for {stock_symbol}: ${current_price:.2f}")
                else:
                    raise Exception("No quote data available")
                    
            except Exception as price_error:
                print(f"❌ Could not get price for {stock_symbol}: {price_error}")
                failed_orders += 1
                continue
            
            # Calculate number of shares to buy (whole shares only, max $100)
            if current_price > 100:
                print(f"❌ {stock_symbol} price (${current_price:.2f}) exceeds $100 budget, skipping...")
                failed_orders += 1
                continue
            
            shares_to_buy = int(100 / current_price)  # Whole shares only
            if shares_to_buy == 0:
                print(f"❌ Cannot afford even 1 share of {stock_symbol} at ${current_price:.2f}")
                failed_orders += 1
                continue
            
            estimated_cost = shares_to_buy * current_price
            print(f"Placing buy order for {shares_to_buy} shares of {stock_symbol} (estimated cost: ${estimated_cost:.2f})...")
            
            market_order_data = MarketOrderRequest(
                symbol=stock_symbol,
                qty=shares_to_buy,  # Number of shares instead of dollar amount
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(order_data=market_order_data)
            print(f"✅ Successfully placed buy order for {shares_to_buy} shares of {stock_symbol} (Order ID: {order.id})")
            successful_orders += 1
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.1)
            
        except APIError as e:
            print(f"❌ Could not place buy order for {stock_symbol}: {e}")
            failed_orders += 1
        except Exception as e:
            print(f"❌ Unexpected error buying {stock_symbol}: {e}")
            failed_orders += 1
    
    print(f"\n--- Buy Orders Summary ---")
    print(f"✅ Successful orders: {successful_orders}")
    print(f"❌ Failed orders: {failed_orders}")
    print(f"Total attempted: {len(top_stocks)}")
    
    if successful_orders == 0:
        print("❌ CRITICAL FAILURE: No buy orders were successfully placed")
    elif successful_orders < len(top_stocks) / 2:
        print("⚠️  WARNING: Less than half of the buy orders succeeded")


def run_trading_strategy():
    """
    The main function that orchestrates the trading strategy.
    """
    print(f"\n Running trading strategy at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Check if market is likely open
    if not is_market_open():
        print("⚠️  Market appears to be closed")
        print("Proceeding anyway...")
    
    # 1. Sell everything
    liquidate_portfolio()
    
    # Wait a moment for sell orders to process
    print("Waiting 15 seconds for most sell orders to process...")
    time.sleep(15)
    
    # 2. Buy top gainers
    buy_top_gainers()
    
    print(f"\n✅ Trading strategy execution finished at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("Next execution scheduled for tomorrow at 14:00 UTC")


# --- SCHEDULER ---

def test_instantly():
    """
    Test the trading strategy immediately without waiting for 14:00 UTC schedule.
    """
    print(" TESTING MODE - Running strategy immediately")
    run_trading_strategy()

def main():
    """
    Main function to schedule and run the bot.
    """
    import sys
    
    print(" Trading Bot 001 Started")
    print(" Strategy is scheduled to run daily at 14:00 UTC")
    print(f" Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Check TradingView availability
    if not TRADINGVIEW_AVAILABLE:
        print("❌ TradingView library not available")
    else:
        print("✅ TradingView library available")
    
    # Check if user wants to test immediately
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        test_instantly()
        return
    
    # Schedule the job for 14:00 UTC daily
    schedule.every().day.at("14:00").do(run_trading_strategy)
    
    # Show next scheduled run
    next_run = schedule.next_run()
    if next_run:
        print(f"⏰ Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    print("\n⏳ Bot is now waiting for scheduled execution time...")
    print(" To test immediately, run: python your_script_name_here.py test")
    
    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(1)  # Checks every second (for more complex startegies you would either need powerful compute or just adjust looping frequency which can be done here)

if __name__ == "__main__":
    main()