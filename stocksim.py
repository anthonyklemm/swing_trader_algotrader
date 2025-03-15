# %%
%matplotlib qt

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf

# Parameters
stock_name = "RKLB"
start_date = "2022-01-08"
end_date = "2025-03-16"
interval = "1d"
print(stock_name)
stock = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
stock.reset_index(inplace=True)

# Ensure we have a unique datetime column called 'TradeDate'
if "Datetime" in stock.columns:
    stock.rename(columns={"Datetime": "TradeDate"}, inplace=True)
elif "Date" in stock.columns:
    stock.rename(columns={"Date": "TradeDate"}, inplace=True)

stock["TradeDate"] = pd.to_datetime(stock["TradeDate"])
stock = stock.loc[:, ~stock.columns.duplicated()]

# Strategy parameters
initial_investment = 500.0
weekly_deposit = 50.0
cash_available = 0.0
dip_threshold = 0.95
gain_threshold = 1.08
stop_loss_threshold = 0.93

interval_to_window = {
    "1m": 60 * 24,    # 1 day in minutes
    "15m": 4 * 24,    # 1 day in 15-min bars
    "1h": 24,         # 1 day in hours
    "4h": 6,          # 1 day in 4-hour bars
    "1d": 3,
    "90m": 18
}
window_size = interval_to_window.get(interval, 96)
# window_size = 20
starting_price = float(stock["Close"].iloc[0])  # Force as float
base_qty = initial_investment / starting_price
holding_qty = base_qty
balances = []
buy_signals, sell_signals, buy_prices = [], [], []
consecutive_buys = 0
max_consecutive_buys = 2
last_buy_index = -window_size
# Track how much total money we've put in so far (cost basis).
# Start with the initial investment:
invested_so_far = initial_investment
invested_so_far_list = []
# Robust week tracking
stock["YearWeek"] = stock["TradeDate"].dt.strftime("%Y-%U")

# Rolling average lines for visualization
stock["RollingAvg"] = stock["Close"].rolling(window_size).mean()
stock["DipLine"]    = stock["RollingAvg"] * dip_threshold
stock["GainLine"]   = stock["RollingAvg"] * gain_threshold

current_week = stock["YearWeek"].iloc[0]
last_buy_time = None
buy_cooldown = pd.Timedelta(hours=4)

# Main Simulation Loop
for i, row in stock.iterrows():
    current_price = float(row["Close"])
    date_val = row["TradeDate"]

    if isinstance(date_val, pd.Series):
        date_val = date_val.iloc[0]

    week_of_year = date_val.strftime("%Y-%U")

    # Weekly deposit
    if week_of_year != current_week:
        invested_so_far += weekly_deposit  # we've invested an additional 50
        cash_available += weekly_deposit
        current_week = week_of_year

    recent_prices = stock["Close"].iloc[max(0, i - interval_to_window.get(interval, 96)):i]
    recent_avg_price = float(recent_prices.mean()) if not recent_prices.empty else current_price

    avg_buy_price = float(np.mean(buy_prices)) if buy_prices else starting_price

    # Buy logic with cooldown
    if (current_price <= recent_avg_price * dip_threshold and
            cash_available > 0 and
            consecutive_buys < 2 and
            (last_buy_time is None or date_val - last_buy_time >= buy_cooldown)):
        spend = cash_available * 0.5
        qty_bought = spend / current_price
        holding_qty += qty_bought
        cash_available -= spend
        buy_prices.append(current_price)
        consecutive_buys += 1
        last_buy_time = date_val
        buy_signals.append((date_val, current_price))

    # Sell logic
    elif (current_price >= avg_buy_price * gain_threshold and holding_qty > base_qty):
        extra_qty = holding_qty - base_qty
        proceeds = extra_qty * current_price
        cash_available += proceeds
        holding_qty = base_qty
        buy_prices = []
        consecutive_buys = 0
        sell_signals.append((date_val, current_price))

    # Stop-loss logic
    elif (current_price <= avg_buy_price * stop_loss_threshold and holding_qty > base_qty):
        extra_qty = holding_qty - base_qty
        proceeds = extra_qty * current_price
        cash_available += proceeds
        holding_qty = base_qty
        buy_prices = []
        consecutive_buys = 0
        sell_signals.append((date_val, current_price))

    balances.append(cash_available + holding_qty * current_price)
    invested_so_far_list.append(invested_so_far)

# Assign balances to a new column
stock["Portfolio Value"] = balances
stock["Invested So Far"] = invested_so_far_list

# Now compute PnL = (portfolio value) - (invested so far)
stock["PnL"] = stock["Portfolio Value"] - stock["Invested So Far"]


# --- Calculate Metrics ---
final_value = balances[-1]
total_weeks = stock["YearWeek"].nunique() - 1
total_invested = initial_investment + weekly_deposit * total_weeks
market_return = float((stock["Close"].iloc[-1] - starting_price) / starting_price * 100)
profit_loss = final_value - total_invested
profit_loss_pct = (profit_loss / total_invested) * 100

info_str = (
    f"Market Return: {market_return:.2f}%\n"
    f"Total Invested: ${total_invested:.2f}\n"
    f"Final Value: ${final_value:.2f}\n"
    f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)\n"
    f"Total Buys: {len(buy_signals)}\n"
    f"Total Sells: {len(sell_signals)}"
)
print(info_str)
# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# 1) Shrink the main plot area to 70% of the figure width
plt.subplots_adjust(right=0.7)

# # Plot the portfolio value on ax1
# ax1.plot(stock["TradeDate"], stock["Portfolio Value"], color="blue")
# ax1.set_xlabel("Date-Time")
# ax1.set_ylabel("Portfolio Value ($)", color="blue")
# ax1.grid(True)

# 1) Plot the PnL on the left axis (like a brokerage profit/loss line)
ax1.plot(stock["TradeDate"], stock["PnL"], color="blue", label="PnL")
ax1.set_xlabel("Date-Time")
ax1.set_ylabel("Profit/Loss ($)", color="blue")
ax1.grid(True)

# Right Y-axis: BTC Price + Rolling Averages
ax2 = ax1.twinx()
ax2.plot(stock["TradeDate"], stock["Close"], color="gray", alpha=0.5, label=stock_name)
ax2.set_ylabel(f"{stock_name} Price ($)", color="gray")

# Rolling average lines
ax2.plot(stock["TradeDate"], stock["RollingAvg"], color="orange", linestyle="--", label="Rolling Avg")
ax2.plot(stock["TradeDate"], stock["DipLine"],   color="green",  linestyle="--", label="Dip Threshold")
ax2.plot(stock["TradeDate"], stock["GainLine"],  color="red",    linestyle="--", label="Gain Threshold")

# Mark Buy/Sell points
for t, p in buy_signals:
    ax2.annotate("Buy", xy=(t, p), xytext=(0, 15), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="green"), color="green")
for t, p in sell_signals:
    ax2.annotate("Sell", xy=(t, p), xytext=(0, -20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="red"), color="red")

ax2.legend(loc="upper left")

# Title & Info Box
# plt.title(f"{stock_name} Strategy: Rolling {window_size}-interval window + ${weekly_deposit}/week")
plt.title(f"{stock_name} Strategy (PnL) | Window={window_size}, ${weekly_deposit}/week")

plt.figtext(
    0.72, 0.92,
    info_str,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5)
)

fig.tight_layout()
plt.show()

#%%
