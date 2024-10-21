'''
Stock Trading Strategy: SMA Crossover with RSI Filter

This strategy combines two technical indicators:
1. Moving Average Crossover: Uses short-term (20-day) and long-term (50-day) Simple Moving Averages (SMA) to identify trend direction.
2. Relative Strength Index (RSI): A 14-day RSI is used to filter out overbought and oversold conditions.

Key Points:
- Limited to 10 trades per month
- Buy Signal: Short-term SMA crosses above Long-term SMA, and RSI is below 70 (not overbought)
- Sell Signal: Short-term SMA crosses below Long-term SMA, and RSI is above 30 (not oversold)

The strategy aims to capture trend reversals while avoiding extreme market conditions, balancing between responsiveness to price changes and protection against false signals.
'''
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import socket
import io
import base64


## bedrock area
import boto3
import os
from botocore.exceptions import ClientError
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).parent
load_dotenv(project_root / '.env')

# Email parameters
SENDER_EMAIL = os.environ.get('SENDER_EMAIL_TRADE')
SENDER_PSWRD = os.environ.get('SENDER_PASSWORD_TRADE')
EMAIL_RECIPIENTS = ['noam.konja@gmail.com']

# Bedrock parameters
REGION_NAME = os.environ.get("REGION_NAME", 'eu-central-1')
BEDROCK_REGION_NAME = os.environ.get("REGION_NAME", 'eu-central-1')
MODEL_NAME = os.environ.get("MODEL_NAME", "anthropic.claude-3-sonnet-20240229-v1:0")
TEMPERATURE = os.environ.get("TEMPERATURE", 0)
MAX_TOKENS = os.environ.get("max_tokens", 4096)
TOP_K = os.environ.get("TOP_K", 250)
TOP_P = os.environ.get("TOP_P", 1)
STOP_SEQUENCES = os.environ.get("STOP_SEQUENCES", ["\n\nHuman"])

session = boto3.Session(
    aws_access_key_id=os.getenv('PROD_NLQ_USER_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('PROD_NLQ_USER_SECRET_KEY'),
    region_name='eu-central-1'
)

def get_response_from_bedrock(prompt):
    try:
        # Create a Bedrock Runtime client
        client = session.client("bedrock-runtime", region_name=BEDROCK_REGION_NAME)

        # Prepare the conversation
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        # Send the message to the model
        response = client.converse(
            modelId=MODEL_NAME,
            messages=conversation,
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
            additionalModelRequestFields={"top_k": TOP_K, "top_p": TOP_P}
        )

        # Extract and return the response text
        return response["output"]["message"]["content"][0]["text"]

    except ClientError as e:
        logger.error(f"ClientError in get_response_from_bedrock: {str(e)}")
        raise

## End of Bedrock area

def create_one_prompt(recommendations, num_activities=10, all_symbols=[], start_date=None):
    
    prompt_manual = f"""
    Strategy Details:
    - Short-term SMA: 20 days
    - Long-term SMA: 50 days
    - RSI period: 14 days
    - Buy when: Short SMA > Long SMA and RSI < 70
    - Sell when: Short SMA < Long SMA and RSI > 30
    - Start date: {start_date}
    Top {num_activities} trading suggestions:
    """
    
    for i, rec in enumerate(recommendations[:num_activities], 1):
        prompt_manual += f"{i}. {rec['strategy']} {rec['symbol']} at ${rec['last_price']:.2f} (Expected Return: {rec['return']:.2f}%, Sharpe Ratio: {rec['sharpe_ratio']:.2f}, Max Drawdown: {rec['max_drawdown']:.2f}%)\n"
    
    prompt = f"""
    You are an AI assistant tasked with analyzing a stock trading strategy and its recommendations.
    Your goal is to interpret the given information and provide insights on the strategy and individual stock recommendations.

    first, get all the symbols I checked in my trade strategy to examine investment opportunities.

    <all_symbols_analyzed>
    {', '.join(all_symbols)}
    </all_symbols_analyzed>

    second, review the strategy details and recommendations provided below:

    <manual_prompt>
    {prompt_manual}
    </manual_prompt>

    Your task is to analyze this information and provide insights. Follow these guidelines:

    1. For each recommended stock:
       a. Analyze the recommendation (buy/sell)
       b. Interpret the metrics (Expected Return, Sharpe Ratio, Max Drawdown)
       c. Provide insights on potential risks and opportunities

    Structure your response as follows:

    <h2>ניתוח מניות</h2>
    <p>
    For each stock, provide your analysis here. Use separate paragraphs for each stock with a <br> tag.
    </p>

    Remember to base your analysis solely on the provided information and use your expertise to offer valuable insights for investors considering this strategy and these specific stock recommendations.
    Please provide your entire response in Hebrew.
    """

    return prompt

def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
    return data

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class PerformanceMetrics:
    def __init__(self, data, daily_pct_returns, daily_equity, trade_equity, duration):
        self.data = data
        self.daily_pct_returns = daily_pct_returns
        self.daily_equity = daily_equity
        self.trade_equity = trade_equity
        self.duration = duration

    def geometric_mean(self, returns: pd.Series) -> float:
        returns = returns.fillna(0) + 1
        if np.any(returns <= 0):
            return 0
        return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

    def calculate_sharpe_ratio(self, risk_free_rate=0.00):
        daily_returns = self.daily_equity.resample('D').last().dropna().pct_change()
        gmean_day_return = self.geometric_mean(daily_returns)
        annual_trading_days = float( 365 if self.daily_equity.index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252)
        annualized_volatility = np.sqrt((daily_returns.var(ddof=int(bool(daily_returns.shape))) + (1 + gmean_day_return)**2)**annual_trading_days - (1 + gmean_day_return)**(2*annual_trading_days)) * 100

        annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
        return_ann = annualized_return * 100
        
        if annualized_volatility != 0:
            sharpe_ratio = (return_ann - risk_free_rate) / annualized_volatility
            return sharpe_ratio
        else:
            return 0

    def calculate_max_drawdown(self):
        cumulative_returns = (1 + self.daily_pct_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_sortino_ratio(self, risk_free_rate=0.02):
        downside_returns = self.daily_pct_returns[self.daily_pct_returns < 0]
        downside_std = downside_returns.std()
        if downside_std != 0:
            sortino_ratio = (self.daily_pct_returns.mean() - risk_free_rate) / downside_std
            return sortino_ratio
        else:
            return 0

class StockStrategy(Strategy):
    def init(self):
        self.sma_short = self.I(calculate_sma, self.data.df, 20)
        self.sma_long = self.I(calculate_sma, self.data.df, 50)
        self.rsi = self.I(calculate_rsi, self.data.df, 14)

    def next(self):
        if self.sma_short[-1] > self.sma_long[-1] and self.rsi[-1] < 70:
            self.buy()
        elif self.sma_short[-1] < self.sma_long[-1] and self.rsi[-1] > 30:
            self.sell()

def backtest_strategy(symbol, start_date, end_date):
    data = get_stock_data(symbol, start_date, end_date)
    bt = Backtest(data, StockStrategy, cash=1000, commission=.0025)
    results = bt.run()
    trades = results['_trades']
    equity_curve = results['_equity_curve']
    return results, data, trades, equity_curve

def generate_recommendations(symbols, start_date, end_date):
    recommendations = []
    for symbol in symbols:
        try:
            results, data, trades, equity_curve = backtest_strategy(symbol, start_date, end_date)

            # daily_pct_returns = equity_curve['Equity'].pct_change().dropna()
            # EntryTime = trades['EntryTime']
            # ExitTime = trades['ExitTime']
            # daily_equity = equity_curve['Equity']
            # trade_equity = daily_equity.loc[EntryTime[0]:ExitTime[0]]
            # duration = results['Duration'].days
            # metrics = PerformanceMetrics(data, daily_pct_returns, daily_equity, trade_equity, duration)
            # sharpe_ratio = metrics.calculate_sharpe_ratio()
            # max_drawdown = metrics.calculate_max_drawdown()
            # sortino_ratio = metrics.calculate_sortino_ratio()

            # print(f"{symbol} Sharpe Ratio: {sharpe_ratio:.3f}")
            # print(f"{symbol} Max Drawdown: {max_drawdown:.3f}")
            # print(f"{symbol} Sortino Ratio: {sortino_ratio:.3f}")

            recommendations.append({
                'symbol': symbol,
                'EntryTime': trades['EntryTime'],
                'ExitTime': trades['ExitTime'],
                'return': results['Return [%]'],
                'equity_final': results['Equity Final [$]'],
                'sharpe_ratio': results['Sharpe Ratio'],
                'max_drawdown': results['Max. Drawdown [%]'],
                'last_price': data['Close'].iloc[-1],  # Use .iloc[-1] instead of [-1]
                'strategy': 'Buy' if results['Return [%]'] > 0 else 'Sell',
                'data': data
            })
        except ValueError as e:
            print(f"Error processing {symbol}: {str(e)}")
    return sorted(recommendations, key=lambda x: x['return'], reverse=True)

def plot_stock_indicators(data, symbol):
    # Get the last 75 days of data
    last_75_days = data.iloc[-75:]  # Use .iloc to get the last 75 rows

    sma_short = calculate_sma(last_75_days, 20)
    sma_long = calculate_sma(last_75_days, 50)
    rsi = calculate_rsi(last_75_days, 14)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{symbol} - Stock Indicators (Last 75 Days)', fontsize=16)

    ax1.plot(last_75_days.index, last_75_days['Close'], label='Close Price')
    ax1.plot(last_75_days.index, sma_short, label='SMA 20')
    ax1.plot(last_75_days.index, sma_long, label='SMA 50')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Add numbers to the price chart
    for i in range(len(last_75_days) - 1, -1, -15):  # Start from the end (today) and move backwards
        price = last_75_days['Close'].iloc[i]
        date = last_75_days.index[i]
        ax1.annotate(f'Price: {price:.2f}\nDate: {date.date()}\nSMA20: {sma_short.iloc[i]:.2f}\nSMA50: {sma_long.iloc[i]:.2f}', 
                     (date, price), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax2.plot(last_75_days.index, rsi, label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()

    # Add numbers to the RSI chart
    for i in range(len(rsi) - 1, -1, -15):  # Start from the end (today) and move backwards
        rsi_value = rsi.iloc[i]
        date = last_75_days.index[i]
        ax2.annotate(f'{rsi_value:.2f}\n{date.date()}', (date, rsi_value), textcoords="offset points", xytext=(0,5), ha='center')

    plt.xlabel('Date')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    plt.close()

    return img_data, symbol

def create_html_content(rcontent, start_date, symbols, recommendations):
    html_content = f"""
    <html>
        <body dir="rtl" style="text-align: right;">
            <h2>סקירת האסטרטגיה</h2>
            <p>
            <strong>הממלצות מבוססות על ניתונים מהתאריך: {start_date.strftime('%d/%m/%Y')}</strong>
            <br>
            <strong>הסמלים שנבדקו: {', '.join(symbols)}</strong>
            <br><br>
            <strong>האסטרטגיה המוצגת היא אסטרטגיית מסחר המבוססת על שילוב של מוצרי מדד טכניים: ממוצעים נעים פשוטים (SMA) ומדד כוח היתר (RSI). העקרונות הבסיסיים של האסטרטגיה הם:</strong>
            <br>
            <ol>
                <li>
                    <strong>ממוצעים נעים (SMA - Simple Moving Average):</strong> כאשר הממוצע הנע הפשוט הקצר (20 ימים) חוצה מעל הממוצע הנע הפשוט הארוך (50 ימים), זהו סימן לתחילת מגמה עולה. כאשר הממוצע הנע הפשוט הקצר חוצה מתחת לממוצע הנע הפשוט הארוך, זהו סימן לתחילת מגמה יורדת.
                </li>
                <li>
                    <strong>מדד כוח היתר (RSI):</strong> משמש כמדד לזיהוי מצבי רכישה/מכירה יתר. ערך RSI מתחת ל-30 מצביע על מצב רכישה יתר, בעוד ערך מעל 70 מצביע על מצב מכירה יתר.
                </li>
                <li>
                    <strong>אותות מסחר:</strong> האסטרטגיה ממליצה לרכוש מניות כאשר הממוצע הנע הקצר חוצה מעל הממוצע הנע הארוך ומדד RSI נמוך מ-70, ולמכור מניות כאשר הממוצע הנע הקצר חוצה מתחת לממוצע הנע הארוך ומדד RSI גבוה מ-30.
                </li>
            </ol>
            <br>
            <strong>מדדים עיקריים:</strong>
            <ul>
                <li><strong>תשואה צפויה (Expected Return):</strong> האחוז הצפוי של רווח או הפסד על ההשקעה.</li>
                <li><strong>יחס שארפ (Sharpe Ratio):</strong> מדד לתשואה מתואמת סיכון, המשווה בין התגמול לסיכון.</li>
                <li><strong>שפל מרבי (Max Drawdown):</strong> הירידה הגדולה ביותר מפסגה לשפל במהלך התקופה.</li>
            </ul>
            </p>

            {rcontent}

            <p><strong>מצורפים גרפים של 3 המדדים עם התשואה הגבוהה ביותר</strong></p>
    """

    return html_content

def send_email(remail, rsubject, rcontent, start_date, symbols, recommendations):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ', '.join(remail)
    msg['Subject'] = rsubject

    # Create the HTML content
    html_content = create_html_content(rcontent, start_date, symbols, recommendations)

    # Attach the HTML content
    msg.attach(MIMEText(html_content, 'html'))

    # Extract top 3 symbols with highest Expected Return
    top_3_symbols = sorted(recommendations, key=lambda x: x['return'], reverse=True)[:3]
    top_3_symbols = [rec['symbol'] for rec in top_3_symbols]

    # Generate and attach images for top 3 symbols
    for symbol in top_3_symbols:
        rec = next(r for r in recommendations if r['symbol'] == symbol)
        img_data, symbol = plot_stock_indicators(rec['data'], symbol)
        image = MIMEImage(img_data)
        image.add_header('Content-ID', f'<{symbol}_image>')
        image.add_header('Content-Disposition', 'inline', filename=f'{symbol}_chart.png')
        msg.attach(image)

        # Add image reference to HTML
        html_content += f'<img src="cid:{symbol}_image" alt="{symbol} Stock Indicators"><br>'

    try:
        with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:     
            smtp.ehlo()
            smtp.starttls() 
            smtp.login(SENDER_EMAIL, SENDER_PSWRD)
            smtp.send_message(msg)
            print("Email sent to", remail)
    except socket.gaierror as e:
        print(f"Failed to connect to SMTP server: {e}")
    except smtplib.SMTPAuthenticationError:
        print("SMTP authentication failed. Please check your email and password.")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")

def monthly_trading_suggestion(symbols, num_activities=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use 1 year of historical data
    
    recommendations = generate_recommendations(symbols, start_date, end_date)
    
    prompt = create_one_prompt(recommendations, num_activities=10, all_symbols=symbols, start_date=start_date)

    bedrock_response = get_response_from_bedrock(prompt)
    
    email_subject = "המלצות מסחר חודשיות"
    email_body = f"{bedrock_response}"

    send_email(EMAIL_RECIPIENTS, email_subject, email_body, start_date, symbols, recommendations)

# Example usage
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'DIS', 'NFLX', 'ADBE']
monthly_trading_suggestion(symbols)
