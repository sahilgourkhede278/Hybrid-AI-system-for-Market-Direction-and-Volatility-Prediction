import uuid
from streamlit_cookies_manager import EncryptedCookieManager
from zoneinfo import ZoneInfo
import os
import sqlite3
import hashlib
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from streamlit_autorefresh import st_autorefresh


# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(page_title="AI Market Dashboard", layout="wide")

cookies = EncryptedCookieManager(
    prefix="ai_market_dashboard/",
    password="change-this-to-a-strong-password"
)

if not cookies.ready():
    st.stop()

# =====================================================
# STOCK MASTER LIST (50+ STOCKS)
# =====================================================
STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Larsen & Toubro": "LT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Grasim": "GRASIM.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Titan": "TITAN.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Hindalco": "HINDALCO.NS",
    "BPCL": "BPCL.NS",
    "Shree Cement": "SHREECEM.NS",
    "SBI Life": "SBILIFE.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "UPL": "UPL.NS",
    "Bharat Electronics": "BEL.NS",
    "Trent": "TRENT.NS",
    "Pidilite": "PIDILITIND.NS",
    "DLF": "DLF.NS",
    "Jio Financial Services": "JIOFIN.NS"
}

PORTFOLIO_STOCKS = {k: v for k, v in STOCKS.items() if k != "NIFTY 50"}
PRACTICE_STOCKS = {k: v for k, v in STOCKS.items() if k != "NIFTY 50"}


# =====================================================
# DATABASE
# =====================================================
DB_NAME = os.path.join(os.getcwd(), "market_app.db")

def get_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except:
        pass
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock TEXT NOT NULL,
            ticker TEXT NOT NULL,
            qty INTEGER NOT NULL,
            buying_date TEXT NOT NULL,
            buying_price REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dummy_accounts (
            user_id INTEGER PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 100000,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dummy_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock TEXT NOT NULL,
            price REAL NOT NULL,
            qty INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS networth_data (
            user_id INTEGER PRIMARY KEY,
            cash REAL NOT NULL DEFAULT 0,
            investments REAL NOT NULL DEFAULT 0,
            property_val REAL NOT NULL DEFAULT 0,
            vehicles REAL NOT NULL DEFAULT 0,
            gold REAL NOT NULL DEFAULT 0,
            home_loan REAL NOT NULL DEFAULT 0,
            home_rate REAL NOT NULL DEFAULT 0,
            home_years INTEGER NOT NULL DEFAULT 0,
            car_loan REAL NOT NULL DEFAULT 0,
            car_rate REAL NOT NULL DEFAULT 0,
            car_years INTEGER NOT NULL DEFAULT 0,
            education_loan REAL NOT NULL DEFAULT 0,
            education_rate REAL NOT NULL DEFAULT 0,
            education_years INTEGER NOT NULL DEFAULT 0,
            credit_card REAL NOT NULL DEFAULT 0,
            credit_rate REAL NOT NULL DEFAULT 0,
            credit_years INTEGER NOT NULL DEFAULT 0,
            other_loan REAL NOT NULL DEFAULT 0,
            other_rate REAL NOT NULL DEFAULT 0,
            other_years INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_token TEXT PRIMARY KEY,
            user_id INTEGER,
            expires_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# =====================================================
# AUTHENTICATION SYSTEM
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(full_name, username, password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users (full_name, username, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (full_name, username, hash_password(password), datetime.now().isoformat()))
        user_id = cur.lastrowid

        cur.execute("""
            INSERT INTO dummy_accounts (user_id, balance)
            VALUES (?, ?)
        """, (user_id, 100000.0))

        cur.execute("""
            INSERT INTO networth_data (
                user_id, cash, investments, property_val, vehicles, gold,
                home_loan, home_rate, home_years,
                car_loan, car_rate, car_years,
                education_loan, education_rate, education_years,
                credit_card, credit_rate, credit_years,
                other_loan, other_rate, other_years,
                updated_at
            )
            VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?)
        """, (user_id, datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True, "Account created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists"


def verify_user(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, full_name, username, password_hash
        FROM users
        WHERE username = ?
    """, (username,))
    user = cur.fetchone()
    conn.close()

    if user and user[3] == hash_password(password):
        return {
            "id": user[0],
            "full_name": user[1],
            "username": user[2]
        }
    return None

def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, full_name, username
        FROM users
        WHERE id = ?
    """, (user_id,))
    user = cur.fetchone()
    conn.close()

    if user:
        return {
            "id": user[0],
            "full_name": user[1],
            "username": user[2]
        }
    return None


def create_user_session(user_id):
    token = str(uuid.uuid4())
    expiry = datetime.now() + timedelta(days=7)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO user_sessions (session_token, user_id, expires_at)
        VALUES (?, ?, ?)
    """, (token, user_id, expiry.isoformat()))

    conn.commit()
    conn.close()
    return token


def get_user_from_session(token):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_id, expires_at
        FROM user_sessions
        WHERE session_token = ?
    """, (token,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return None

    user_id, expires_at = row

    try:
        expiry_dt = datetime.fromisoformat(expires_at)
    except Exception:
        conn.close()
        return None

    if datetime.now() > expiry_dt:
        cur.execute("""
            DELETE FROM user_sessions
            WHERE session_token = ?
        """, (token,))
        conn.commit()
        conn.close()
        return None

    conn.close()
    return get_user_by_id(user_id)


def delete_user_session(token):
    if not token:
        return

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM user_sessions
        WHERE session_token = ?
    """, (token,))

    conn.commit()
    conn.close()

# =====================================================
# PORTFOLIO FUNCTIONS
# =====================================================
def get_user_portfolio(user_id):
    conn = get_connection()
    query = """
        SELECT stock, ticker, qty, buying_date, buying_price
        FROM portfolio
        WHERE user_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df


def add_or_update_portfolio(user_id, stock, ticker, qty, buying_date, buying_price):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, qty, buying_price
        FROM portfolio
        WHERE user_id = ? AND stock = ?
    """, (user_id, stock))
    row = cur.fetchone()

    if row:
        row_id, old_qty, old_price = row
        new_qty = old_qty + qty
        new_price = ((old_qty * old_price) + (qty * buying_price)) / new_qty

        cur.execute("""
            UPDATE portfolio
            SET qty = ?, buying_price = ?, buying_date = ?
            WHERE id = ?
        """, (new_qty, new_price, str(buying_date), row_id))
    else:
        cur.execute("""
            INSERT INTO portfolio (user_id, stock, ticker, qty, buying_date, buying_price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, stock, ticker, qty, str(buying_date), buying_price))

    conn.commit()
    conn.close()


# =====================================================
# DUMMY MARKET FUNCTIONS
# =====================================================
def get_dummy_balance(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT balance FROM dummy_accounts WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()

    return float(row[0]) if row else 100000.0


def update_dummy_balance(user_id, new_balance):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE dummy_accounts
        SET balance = ?
        WHERE user_id = ?
    """, (float(new_balance), user_id))

    conn.commit()
    conn.close()


def get_dummy_portfolio(user_id):
    conn = get_connection()
    query = """
        SELECT
            stock,
            ROUND(SUM(price * qty) / SUM(qty), 2) AS price,
            SUM(qty) AS qty
        FROM dummy_portfolio
        WHERE user_id = ?
        GROUP BY stock
        ORDER BY stock
    """
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df


def add_dummy_stock(user_id, stock, price, qty):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO dummy_portfolio (user_id, stock, price, qty)
        VALUES (?, ?, ?, ?)
    """, (user_id, stock, float(price), int(qty)))

    conn.commit()
    conn.close()


def sell_dummy_stock(user_id, stock, price, qty_to_sell):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT id, qty
            FROM dummy_portfolio
            WHERE user_id = ? AND stock = ?
            ORDER BY id ASC
        """, (user_id, stock))
        rows = cur.fetchall()

        if not rows:
            conn.close()
            return False, 0

        total_qty = sum(int(row[1]) for row in rows)

        if total_qty < qty_to_sell:
            conn.close()
            return False, 0

        remaining_to_sell = int(qty_to_sell)

        for row_id, row_qty in rows:
            row_qty = int(row_qty)

            if remaining_to_sell <= 0:
                break

            if row_qty <= remaining_to_sell:
                cur.execute("DELETE FROM dummy_portfolio WHERE id = ?", (row_id,))
                remaining_to_sell -= row_qty
            else:
                new_qty = row_qty - remaining_to_sell
                cur.execute(
                    "UPDATE dummy_portfolio SET qty = ? WHERE id = ?",
                    (new_qty, row_id)
                )
                remaining_to_sell = 0

        sell_value = float(price) * int(qty_to_sell)

        cur.execute(
            "SELECT balance FROM dummy_accounts WHERE user_id = ?",
            (user_id,)
        )
        bal_row = cur.fetchone()

        current_balance = float(bal_row[0]) if bal_row else 100000.0
        new_balance = current_balance + sell_value

        cur.execute("""
            UPDATE dummy_accounts
            SET balance = ?
            WHERE user_id = ?
        """, (new_balance, user_id))

        conn.commit()
        conn.close()
        return True, sell_value

    except Exception:
        conn.rollback()
        conn.close()
        raise


# =====================================================
# NET WORTH FUNCTIONS
# =====================================================
def get_networth_data(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            cash, investments, property_val, vehicles, gold,
            home_loan, home_rate, home_years,
            car_loan, car_rate, car_years,
            education_loan, education_rate, education_years,
            credit_card, credit_rate, credit_years,
            other_loan, other_rate, other_years
        FROM networth_data
        WHERE user_id = ?
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "cash": float(row[0]),
            "investments": float(row[1]),
            "property_val": float(row[2]),
            "vehicles": float(row[3]),
            "gold": float(row[4]),
            "home_loan": float(row[5]),
            "home_rate": float(row[6]),
            "home_years": int(row[7]),
            "car_loan": float(row[8]),
            "car_rate": float(row[9]),
            "car_years": int(row[10]),
            "education_loan": float(row[11]),
            "education_rate": float(row[12]),
            "education_years": int(row[13]),
            "credit_card": float(row[14]),
            "credit_rate": float(row[15]),
            "credit_years": int(row[16]),
            "other_loan": float(row[17]),
            "other_rate": float(row[18]),
            "other_years": int(row[19]),
        }

    return {
        "cash": 0.0,
        "investments": 0.0,
        "property_val": 0.0,
        "vehicles": 0.0,
        "gold": 0.0,
        "home_loan": 0.0,
        "home_rate": 0.0,
        "home_years": 0,
        "car_loan": 0.0,
        "car_rate": 0.0,
        "car_years": 0,
        "education_loan": 0.0,
        "education_rate": 0.0,
        "education_years": 0,
        "credit_card": 0.0,
        "credit_rate": 0.0,
        "credit_years": 0,
        "other_loan": 0.0,
        "other_rate": 0.0,
        "other_years": 0,
    }


def save_networth_data(
    user_id,
    cash, investments, property_val, vehicles, gold,
    home_loan, home_rate, home_years,
    car_loan, car_rate, car_years,
    education_loan, education_rate, education_years,
    credit_card, credit_rate, credit_years,
    other_loan, other_rate, other_years
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO networth_data (
            user_id,
            cash, investments, property_val, vehicles, gold,
            home_loan, home_rate, home_years,
            car_loan, car_rate, car_years,
            education_loan, education_rate, education_years,
            credit_card, credit_rate, credit_years,
            other_loan, other_rate, other_years,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        float(cash), float(investments), float(property_val), float(vehicles), float(gold),
        float(home_loan), float(home_rate), int(home_years),
        float(car_loan), float(car_rate), int(car_years),
        float(education_loan), float(education_rate), int(education_years),
        float(credit_card), float(credit_rate), int(credit_years),
        float(other_loan), float(other_rate), int(other_years),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# =====================================================
# CACHED MARKET DATA HELPERS
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history_data(ticker_symbol, start_date, end_date):
    data = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        threads=True
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_prices_batch(ticker_list):
    ticker_list = [t for t in ticker_list if t]
    if not ticker_list:
        return {}

    result = {}
    try:
        data = yf.download(
            ticker_list,
            period="5d",
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="ticker"
        )

        if data is None or data.empty:
            return result

        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if "Close" in data.columns and not data["Close"].dropna().empty:
                result[ticker] = float(data["Close"].dropna().iloc[-1])
            return result

        for ticker in ticker_list:
            try:
                ticker_df = data[ticker].copy()
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = ticker_df.columns.get_level_values(0)
                if "Close" in ticker_df.columns and not ticker_df["Close"].dropna().empty:
                    result[ticker] = float(ticker_df["Close"].dropna().iloc[-1])
            except Exception:
                continue

        return result
    except Exception:
        return result


@st.cache_data(ttl=86400, show_spinner=False)
def is_valid_market_day(ticker_symbol, selected_date_str):
    try:
        selected_date = pd.to_datetime(selected_date_str).date()
        next_date = selected_date + timedelta(days=1)

        data = yf.download(
            ticker_symbol,
            start=str(selected_date),
            end=str(next_date),
            progress=False,
            auto_adjust=True
        )

        if data is None or data.empty:
            return False, np.nan

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close_price = float(data["Close"].iloc[0])
        return True, close_price
    except Exception:
        return False, np.nan


@st.cache_data(ttl=5, show_spinner=False)
def fetch_live_intraday_data(ticker_symbol):
    data = yf.download(
        ticker_symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True,
        threads=True
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


# =====================================================
# HELPERS
# =====================================================
def is_market_open():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    if now.weekday() >= 5:
        return False

    current_time = now.time()
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()

    return market_open <= current_time <= market_close


def is_live_market_tradable(ticker_symbol):
    if not is_market_open():
        return False

    data = fetch_live_intraday_data(ticker_symbol)
    return not data.empty


def go_to_page(page_name):
    st.session_state.page = page_name
    st.query_params["page"] = page_name
    st.rerun()


def logout_user():
    token = cookies.get("session_token")

    if token:
        delete_user_session(token)

    cookies["session_token"] = ""
    cookies.save()

    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.page = "login"
    st.query_params.clear()
    st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_info_data(ticker_symbol):
    try:
        stock_obj = yf.Ticker(ticker_symbol)
        info = stock_obj.info

        return {
            "longName": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "website": info.get("website", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "currentPrice": info.get("currentPrice", "N/A"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "N/A"),
            "trailingPE": info.get("trailingPE", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "longBusinessSummary": info.get("longBusinessSummary", "N/A")
        }
    except Exception:
        return {
            "longName": "N/A",
            "sector": "N/A",
            "industry": "N/A",
            "country": "N/A",
            "website": "N/A",
            "marketCap": "N/A",
            "currentPrice": "N/A",
            "fiftyTwoWeekHigh": "N/A",
            "fiftyTwoWeekLow": "N/A",
            "trailingPE": "N/A",
            "dividendYield": "N/A",
            "longBusinessSummary": "N/A"
        }


def format_large_number(value):
    if value == "N/A" or pd.isna(value):
        return "N/A"

    try:
        value = float(value)
        if value >= 1_000_000_000_000:
            return f"₹ {value / 1_000_000_000_000:.2f} T"
        elif value >= 1_000_000_000:
            return f"₹ {value / 1_000_000_000:.2f} B"
        elif value >= 1_000_000:
            return f"₹ {value / 1_000_000:.2f} M"
        else:
            return f"₹ {value:,.2f}"
    except Exception:
        return str(value)


def format_percentage(value):
    if value == "N/A" or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)

# =====================================================
# INITIALIZE
# =====================================================
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "page" not in st.session_state:
    st.session_state.page = "login"

query_page = st.query_params.get("page", None)

if query_page:
    st.session_state.page = query_page


cookie_token = cookies.get("session_token")

if cookie_token and not st.session_state.logged_in:
    saved_user = get_user_from_session(cookie_token)

    if saved_user:
        st.session_state.logged_in = True
        st.session_state.current_user = saved_user

        if st.session_state.page == "login":
            st.session_state.page = "dashboard"
    else:
        cookies["session_token"] = ""
        cookies.save()

# =====================================================
# PAGE 0 : LOGIN / SIGNUP
# =====================================================
if not st.session_state.logged_in:
    st.title("🔐 AI Market Dashboard Login")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        st.subheader("Login")

        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_btn"):
            if not login_username or not login_password:
                st.warning("Please enter username and password")
            else:
                user = verify_user(login_username, login_password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.current_user = user
                    st.session_state.page = "dashboard"

                    token = create_user_session(user["id"])
                    cookies["session_token"] = token
                    cookies.save()

                    st.query_params["page"] = "dashboard"
                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with signup_tab:
        st.subheader("Create New Account")

        signup_name = st.text_input("Full Name", key="signup_name")
        signup_username = st.text_input("Choose Username", key="signup_username")
        signup_password = st.text_input("Choose Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Create Account", key="signup_btn"):
            if not signup_name or not signup_username or not signup_password or not signup_confirm:
                st.warning("Please fill all fields")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match")
            elif len(signup_password) < 4:
                st.warning("Password must be at least 4 characters")
            else:
                ok, msg = create_user(signup_name, signup_username, signup_password)
                if ok:
                    st.success(msg + " - now login")
                else:
                    st.error(msg)

    st.stop()


# =====================================================
# USER HEADER
# =====================================================
current_user = st.session_state.current_user
user_id = current_user["id"]

top1, top2, top3 = st.columns([5, 2, 1])
with top1:
    st.caption(f"Logged in as: **{current_user['full_name']}** (@{current_user['username']})")
with top2:
    st.caption(f"User ID: {user_id}")
with top3:
    if st.button("Logout"):
        logout_user()


# =====================================================
# PAGE 1 : MARKET DASHBOARD
# =====================================================
if st.session_state.page == "dashboard":

    st.title("📊 AI Market Direction Dashboard")
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        if st.button("💰 Calculate Your Net Worth", use_container_width=True):
            go_to_page("networth")

    with nav2:
        if st.button("🧪 Dummy Market For Practicing", use_container_width=True):
            go_to_page("dummy_market")

    with nav3:
        if st.button("📚 Stock Information", use_container_width=True):
            go_to_page("stock_info")

    st.subheader("📈 Stock AI Analysis")

    stock_name = st.selectbox(
        "Select Stock for AI Prediction",
        list(STOCKS.keys()),
        key="single_stock"
    )

    ticker = STOCKS[stock_name]

    algo = st.selectbox(
        "Select Algorithm",
        ("Random Forest", "Logistic Regression", "KNN"),
        key="algo_select"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    data = fetch_history_data(
        ticker_symbol=ticker,
        start_date="2015-01-01",
        end_date=today
    )

    if data.empty:
        st.error("Unable to fetch market data. Try another stock.")
        st.stop()

    data["Return"] = data["Close"].pct_change()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Return"].rolling(10).std()
    data["Direction"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

    data = data.dropna()

    if len(data) < 50:
        st.warning("Not enough data for prediction.")
        st.stop()

    features = ["Return", "MA10", "MA20", "Volatility"]
    X = data[features]
    y = data["Direction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    if algo == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif algo == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    else:
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)

    prediction = model.predict(X.iloc[-1:])[0]
    prob = model.predict_proba(X.iloc[-1:])[0][prediction]
    accuracy = accuracy_score(y_test, model.predict(X_test))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Market Direction")
        if prediction == 1:
            st.success("UP 🔼")
        else:
            st.error("DOWN 🔽")
        st.metric("Confidence", f"{round(prob * 100, 2)}%")

    with col2:
        st.subheader("Volatility")
        latest_vol = data["Volatility"].iloc[-1]
        avg_vol = data["Volatility"].mean()

        if latest_vol > avg_vol:
            st.warning("HIGH ⚠")
        else:
            st.info("LOW 📉")

        st.metric("Current Volatility", round(latest_vol, 5))

    with col3:
        st.subheader("Model Accuracy")
        st.metric("Accuracy", f"{round(accuracy * 100, 2)}%")

    st.subheader("🤖 AI Recommendation")

    if prediction == 1 and prob > 0.6:
        recommendation = "Buy 🟢"
    elif prediction == 0 and prob > 0.6:
        recommendation = "Sell 🔴"
    else:
        recommendation = "Hold 🟡"

    st.info(
        f"Recommendation: {recommendation} "
        f"(Confidence: {round(prob * 100, 2)}%)"
    )

    st.subheader("📈 Stock Price Chart")
    data_chart = data.reset_index()

    fig = px.line(
        data_chart,
        x="Date",
        y="Close",
        title=f"{stock_name} Price Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📂 Portfolio Tracker")

    with st.form("add_stock_form"):
        stock_name_port = st.selectbox(
            "Select Stock to Add",
            list(PORTFOLIO_STOCKS.keys()),
            key="portfolio_stock"
        )
        qty = st.number_input("Quantity Owned", 1, step=1, key="portfolio_qty")
        buying_date = st.date_input("Buying Date", key="portfolio_buying_date")
        submitted = st.form_submit_button("Add to Portfolio")

        if submitted:
            ticker_val = PORTFOLIO_STOCKS[stock_name_port]
            selected_date = pd.to_datetime(buying_date).date()
            today_date = datetime.now().date()

            if selected_date > today_date:
                st.error("Future dates are not allowed.")
            else:
                valid_day, buying_price = is_valid_market_day(ticker_val, str(selected_date))

                if not valid_day:
                    st.error("Selected date is a market holiday or closed trading day.")
                else:
                    add_or_update_portfolio(
                        user_id=user_id,
                        stock=stock_name_port,
                        ticker=ticker_val,
                        qty=int(qty),
                        buying_date=str(selected_date),
                        buying_price=float(buying_price)
                    )
                    st.success(f"Added/Updated {stock_name_port} in portfolio")
                    st.rerun()

    portfolio_df = get_user_portfolio(user_id)

    if not portfolio_df.empty:
        ticker_price_map = fetch_latest_prices_batch(portfolio_df["ticker"].dropna().unique().tolist())
        portfolio_df["Latest Price"] = portfolio_df["ticker"].map(ticker_price_map)

        portfolio_df["Profit ₹"] = (
            (portfolio_df["Latest Price"] - portfolio_df["buying_price"]) * portfolio_df["qty"]
        )
        portfolio_df["Profit %"] = (
            ((portfolio_df["Latest Price"] - portfolio_df["buying_price"]) / portfolio_df["buying_price"]) * 100
        )
        portfolio_df["Total Value"] = portfolio_df["Latest Price"] * portfolio_df["qty"]

        portfolio_df["Profit %"] = portfolio_df["Profit %"].round(2)
        portfolio_df["Profit ₹"] = portfolio_df["Profit ₹"].round(2)
        portfolio_df["Total Value"] = portfolio_df["Total Value"].round(2)

        st.dataframe(
            portfolio_df[
                [
                    "stock",
                    "qty",
                    "buying_date",
                    "buying_price",
                    "Latest Price",
                    "Profit ₹",
                    "Profit %",
                    "Total Value"
                ]
            ],
            use_container_width=True
        )

        st.write("💰 Total Portfolio Value: ₹", round(portfolio_df["Total Value"].sum(), 2))

        fig_port = px.pie(
            portfolio_df,
            names="stock",
            values="Total Value",
            title="Portfolio Distribution"
        )
        st.plotly_chart(fig_port, use_container_width=True)
    else:
        st.info("No portfolio data yet for this user")


# =====================================================
# PAGE 2 : NET WORTH CALCULATOR
# =====================================================
elif st.session_state.page == "networth":

    st.title("💰 Net Worth Calculator")

    if st.button("⬅ Back to Dashboard"):
        go_to_page("dashboard")

    saved_networth = get_networth_data(user_id)

    st.subheader("Assets – Everything you own")

    cash = st.number_input("Cash / Bank Balance (₹)", min_value=0.0, step=1000.0, value=saved_networth["cash"])
    investments = st.number_input("Investments (Stocks / Mutual Funds / Crypto) (₹)", min_value=0.0, step=1000.0, value=saved_networth["investments"])
    property_val = st.number_input("Property / Real Estate Value (₹)", min_value=0.0, step=1000.0, value=saved_networth["property_val"])
    vehicles = st.number_input("Vehicles Value (₹)", min_value=0.0, step=1000.0, value=saved_networth["vehicles"])
    gold = st.number_input("Gold / Valuable Items (₹)", min_value=0.0, step=1000.0, value=saved_networth["gold"])

    total_assets = cash + investments + property_val + vehicles + gold
    st.write("Total Assets: ₹", round(total_assets, 2))

    st.subheader("Liabilities – Everything you owe")

    st.markdown("### Home Loan")
    home_loan = st.number_input("Home Loan Amount (₹)", min_value=0.0, step=1000.0, value=saved_networth["home_loan"])
    c1, c2 = st.columns(2)
    with c1:
        home_rate = st.number_input("Home Loan Interest Rate (%)", min_value=0.0, step=0.1, key="home_rate", value=saved_networth["home_rate"])
    with c2:
        home_years = st.number_input("Home Loan Tenure (Years)", min_value=0, step=1, key="home_years", value=saved_networth["home_years"])
    home_interest = home_loan * home_rate * home_years / 100
    home_total_payable = home_loan + home_interest

    st.markdown("### Car Loan")
    car_loan = st.number_input("Car Loan Amount (₹)", min_value=0.0, step=1000.0, value=saved_networth["car_loan"])
    c1, c2 = st.columns(2)
    with c1:
        car_rate = st.number_input("Car Loan Interest Rate (%)", min_value=0.0, step=0.1, key="car_rate", value=saved_networth["car_rate"])
    with c2:
        car_years = st.number_input("Car Loan Tenure (Years)", min_value=0, step=1, key="car_years", value=saved_networth["car_years"])
    car_interest = car_loan * car_rate * car_years / 100
    car_total_payable = car_loan + car_interest

    st.markdown("### Education Loan")
    education_loan = st.number_input("Education Loan Amount (₹)", min_value=0.0, step=1000.0, value=saved_networth["education_loan"])
    c1, c2 = st.columns(2)
    with c1:
        education_rate = st.number_input("Education Loan Interest Rate (%)", min_value=0.0, step=0.1, key="edu_rate", value=saved_networth["education_rate"])
    with c2:
        education_years = st.number_input("Education Loan Tenure (Years)", min_value=0, step=1, key="edu_years", value=saved_networth["education_years"])
    education_interest = education_loan * education_rate * education_years / 100
    education_total_payable = education_loan + education_interest

    st.markdown("### Credit Card Debt")
    credit_card = st.number_input("Credit Card Debt (₹)", min_value=0.0, step=1000.0, value=saved_networth["credit_card"])
    c1, c2 = st.columns(2)
    with c1:
        credit_rate = st.number_input("Credit Card Interest Rate (%)", min_value=0.0, step=0.1, key="cc_rate", value=saved_networth["credit_rate"])
    with c2:
        credit_years = st.number_input("Credit Card Duration (Years)", min_value=0, step=1, key="cc_years", value=saved_networth["credit_years"])
    credit_interest = credit_card * credit_rate * credit_years / 100
    credit_total_payable = credit_card + credit_interest

    st.markdown("### Other Liabilities")
    other_loan = st.number_input("Other Loan / Debt (₹)", min_value=0.0, step=1000.0, value=saved_networth["other_loan"])
    c1, c2 = st.columns(2)
    with c1:
        other_rate = st.number_input("Other Loan Interest Rate (%)", min_value=0.0, step=0.1, key="other_rate", value=saved_networth["other_rate"])
    with c2:
        other_years = st.number_input("Other Loan Tenure (Years)", min_value=0, step=1, key="other_years", value=saved_networth["other_years"])
    other_interest = other_loan * other_rate * other_years / 100
    other_total_payable = other_loan + other_interest

    total_liabilities = (
        home_total_payable
        + car_total_payable
        + education_total_payable
        + credit_total_payable
        + other_total_payable
    )

    st.write("Total Liabilities: ₹", round(total_liabilities, 2))

    net_worth = total_assets - total_liabilities

    st.subheader("📌 Final Result")
    if net_worth > 0:
        st.success(f"Your Net Worth is: ₹ {round(net_worth, 2)}")
    elif net_worth < 0:
        st.error(f"Your Net Worth is: ₹ {round(net_worth, 2)}")
    else:
        st.info("Your Net Worth is: ₹ 0.0")

    st.subheader("📊 Assets vs Liabilities")

    chart_df = pd.DataFrame({
        "Category": ["Assets", "Liabilities"],
        "Amount": [total_assets, total_liabilities]
    })

    fig_networth = px.bar(
        chart_df,
        x="Category",
        y="Amount",
        title="Assets vs Liabilities"
    )
    st.plotly_chart(fig_networth, use_container_width=True)

    if st.button("💾 Save Net Worth Data"):
        save_networth_data(
            user_id=user_id,
            cash=cash,
            investments=investments,
            property_val=property_val,
            vehicles=vehicles,
            gold=gold,
            home_loan=home_loan,
            home_rate=home_rate,
            home_years=home_years,
            car_loan=car_loan,
            car_rate=car_rate,
            car_years=car_years,
            education_loan=education_loan,
            education_rate=education_rate,
            education_years=education_years,
            credit_card=credit_card,
            credit_rate=credit_rate,
            credit_years=credit_years,
            other_loan=other_loan,
            other_rate=other_rate,
            other_years=other_years
        )
        st.success("Net Worth data saved successfully")


# =====================================================
# PAGE 3 : DUMMY MARKET
# =====================================================
elif st.session_state.page == "dummy_market":

    st.title("🧪 Dummy Market (Practice Trading)")

    # refresh every 5 seconds
    st_autorefresh(interval=5000, key="dummy_market_refresh_10s")

    if is_live_market_tradable("RELIANCE.NS"):
        st.success("🟢 Market is Open")
    else:
        st.error("🔴 Market is Closed / Holiday")

    if st.button("⬅ Back to Dashboard"):
        go_to_page("dashboard")

    current_balance = get_dummy_balance(user_id)
    st.metric("Virtual Balance", f"₹ {round(current_balance, 2)}")

    stock = st.selectbox("Select Stock", list(PRACTICE_STOCKS.keys()), key="dummy_stock")
    ticker = PRACTICE_STOCKS[stock]

    data = fetch_live_intraday_data(ticker)

    if data.empty:
        st.warning("Live price unavailable because market is closed / holiday.")
        st.stop()

    price = float(data["Close"].iloc[-1])

    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.metric("Current Price", f"₹ {round(price, 2)}")
    with top_col2:
        previous_price = float(data["Close"].iloc[-2]) if len(data) > 1 else price
        price_change = price - previous_price
        st.metric("Live Change", f"₹ {round(price_change, 2)}")

    qty = st.number_input("Quantity", min_value=1, step=1, key="dummy_qty")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Buy"):
            if not is_live_market_tradable(ticker):
                st.error("Market is closed. Buying is not allowed on holidays, weekends, or outside market hours.")
            else:
                cost = price * qty
                balance = get_dummy_balance(user_id)

                if balance >= cost:
                    update_dummy_balance(user_id, balance - cost)
                    add_dummy_stock(user_id, stock, price, qty)
                    fetch_live_intraday_data.clear()
                    st.success("Stock Purchased")
                    st.rerun()
                else:
                    st.error("Not enough balance")

    with col2:
        if st.button("🔴 Sell"):
            if not is_live_market_tradable(ticker):
                st.error("Market is closed. Selling is not allowed on holidays, weekends, or outside market hours.")
            else:
                sold, value = sell_dummy_stock(user_id, stock, price, qty)
                if sold:
                    fetch_live_intraday_data.clear()
                    st.success(f"{qty} share(s) sold for ₹ {round(value, 2)}")
                    st.rerun()
                else:
                    st.error("Not enough stock to sell")

    st.subheader("📊 Practice Portfolio")

    df = get_dummy_portfolio(user_id)

    if not df.empty:
        ticker_map = {name: symbol for name, symbol in PRACTICE_STOCKS.items()}

        current_prices = []
        live_changes = []

        for s in df["stock"]:
            ticker_symbol = ticker_map[s]
            live_data = fetch_live_intraday_data(ticker_symbol)

            if not live_data.empty:
                current_price = float(live_data["Close"].iloc[-1])
                if len(live_data) > 1:
                    change_val = current_price - float(live_data["Close"].iloc[-2])
                else:
                    change_val = 0.0
            else:
                current_price = float(df[df["stock"] == s]["price"].iloc[0])
                change_val = 0.0

            current_prices.append(current_price)
            live_changes.append(change_val)

        df["Current Price"] = current_prices
        df["Live Change ₹"] = live_changes
        df["Investment"] = df["price"] * df["qty"]
        df["Market Value"] = df["Current Price"] * df["qty"]
        df["Profit ₹"] = df["Market Value"] - df["Investment"]
        df["Profit %"] = (df["Profit ₹"] / df["Investment"]) * 100

        df["Live Change ₹"] = df["Live Change ₹"].round(2)
        df["Investment"] = df["Investment"].round(2)
        df["Market Value"] = df["Market Value"].round(2)
        df["Profit ₹"] = df["Profit ₹"].round(2)
        df["Profit %"] = df["Profit %"].round(2)

        total_investment = df["Investment"].sum()
        total_market_value = df["Market Value"].sum()
        total_profit = df["Profit ₹"].sum()
        total_profit_percent = (total_profit / total_investment) * 100 if total_investment > 0 else 0

        met1, met2, met3 = st.columns(3)
        with met1:
            st.metric("Total Investment", f"₹ {round(total_investment, 2)}")
        with met2:
            st.metric("Portfolio Market Value", f"₹ {round(total_market_value, 2)}")
        with met3:
            st.metric("Live Total Profit", f"₹ {round(total_profit, 2)}", f"{round(total_profit_percent, 2)}%")

        st.dataframe(
            df[
                [
                    "stock",
                    "price",
                    "qty",
                    "Current Price",
                    "Live Change ₹",
                    "Investment",
                    "Market Value",
                    "Profit ₹",
                    "Profit %"
                ]
            ],
            use_container_width=True
        )

    else:
        st.info("No stocks purchased yet")

    st.subheader("📈 Stock Price Chart")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Line", "Bar", "Candlestick"],
        key="dummy_chart_type"
    )

    chart_data = data.copy()

    if chart_data.index.tz is not None:
        chart_data.index = chart_data.index.tz_convert("Asia/Kolkata")
    else:
        chart_data.index = chart_data.index.tz_localize("UTC").tz_convert("Asia/Kolkata")

    chart_data = chart_data.reset_index()

    if "Datetime" in chart_data.columns:
        time_col = "Datetime"
    elif "Date" in chart_data.columns:
        time_col = "Date"
    else:
        time_col = chart_data.columns[0]

    chart_data["Time"] = pd.to_datetime(chart_data[time_col]).dt.strftime("%H:%M:%S")

    if chart_type == "Line":
        fig = px.line(
            chart_data,
            x="Time",
            y="Close",
            title=f"{stock} Live Price Chart"
        )

    elif chart_type == "Bar":
        fig = px.bar(
            chart_data,
            x="Time",
            y="Close",
            title=f"{stock} Live Price Chart"
        )

    else:
        fig = go.Figure(data=[go.Candlestick(
            x=chart_data["Time"],
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"]
        )])

        fig.update_layout(
            title=f"{stock} Live Candlestick Chart",
            xaxis_title="Time (IST)",
            yaxis_title="Price"
        )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# PAGE 4 : STOCK INFORMATION
# =====================================================
elif st.session_state.page == "stock_info":

    st.title("📚 Stock Information")

    if st.button("⬅ Back to Dashboard"):
        go_to_page("dashboard")

    st.subheader("Search and View Complete Historical Stock Information")

    # ---------------- SEARCH BOX ----------------
    search_text = st.text_input("Search Stock", placeholder="Type stock name...").strip().lower()

    filtered_stock_names = [
        stock_name for stock_name in STOCKS.keys()
        if search_text in stock_name.lower()
    ]

    if not filtered_stock_names:
        st.warning("No matching stocks found.")
        st.stop()

    info_stock_name = st.selectbox(
        "Select Stock",
        filtered_stock_names,
        key="info_stock_select"
    )

    info_ticker = STOCKS[info_stock_name]

    # ---------------- DATE RANGE SELECT ----------------
    period_option = st.selectbox(
        "Select Historical Range",
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        key="info_period_select"
    )

    today_date = datetime.now().date()

    if period_option == "1 Year":
        start_date = str(today_date - timedelta(days=365))
    elif period_option == "3 Years":
        start_date = str(today_date - timedelta(days=365 * 3))
    elif period_option == "5 Years":
        start_date = str(today_date - timedelta(days=365 * 5))
    elif period_option == "10 Years":
        start_date = str(today_date - timedelta(days=365 * 10))
    else:
        start_date = "2000-01-01"

    end_date = str(today_date)

    # ---------------- FETCH DATA ----------------
    info_data = fetch_history_data(
        ticker_symbol=info_ticker,
        start_date=start_date,
        end_date=end_date
    )

    company_info = fetch_stock_info_data(info_ticker)

    if info_data.empty:
        st.error("Unable to fetch historical data for the selected stock.")
        st.stop()

    # ---------------- TECHNICAL CALCULATIONS ----------------
    info_data = info_data.copy()
    info_data["MA20"] = info_data["Close"].rolling(20).mean()
    info_data["MA50"] = info_data["Close"].rolling(50).mean()
    info_data["Daily Return %"] = info_data["Close"].pct_change() * 100
    info_data["Volatility"] = info_data["Daily Return %"].rolling(20).std()

    latest_close = float(info_data["Close"].iloc[-1])
    first_close = float(info_data["Close"].iloc[0])
    highest_price = float(info_data["High"].max())
    lowest_price = float(info_data["Low"].min())
    total_return = ((latest_close - first_close) / first_close) * 100 if first_close != 0 else 0
    avg_daily_return = float(info_data["Daily Return %"].dropna().mean()) if not info_data["Daily Return %"].dropna().empty else 0
    latest_volatility = float(info_data["Volatility"].dropna().iloc[-1]) if not info_data["Volatility"].dropna().empty else 0

    # ---------------- COMPANY INFO ----------------
    st.subheader("🏢 Company Information")

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"**Company Name:** {company_info['longName']}")
        st.write(f"**Sector:** {company_info['sector']}")
        st.write(f"**Industry:** {company_info['industry']}")
        st.write(f"**Country:** {company_info['country']}")
        st.write(f"**Website:** {company_info['website']}")

    with c2:
        st.write(f"**Market Cap:** {format_large_number(company_info['marketCap'])}")
        st.write(f"**Current Price:** {company_info['currentPrice']}")
        st.write(f"**52 Week High:** {company_info['fiftyTwoWeekHigh']}")
        st.write(f"**52 Week Low:** {company_info['fiftyTwoWeekLow']}")
        st.write(f"**P/E Ratio:** {company_info['trailingPE']}")
        st.write(f"**Dividend Yield:** {format_percentage(company_info['dividendYield'])}")

    st.write("**Business Summary:**")
    st.info(company_info["longBusinessSummary"])

    # ---------------- SUMMARY METRICS ----------------
    st.subheader("📊 Historical Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Latest Close", f"₹ {round(latest_close, 2)}")
    with m2:
        st.metric("Highest Price", f"₹ {round(highest_price, 2)}")
    with m3:
        st.metric("Lowest Price", f"₹ {round(lowest_price, 2)}")
    with m4:
        st.metric("Total Return", f"{round(total_return, 2)}%")

    m5, m6, m7 = st.columns(3)
    with m5:
        st.metric("Avg Daily Return", f"{round(avg_daily_return, 2)}%")
    with m6:
        st.metric("Volatility", f"{round(latest_volatility, 2)}")
    with m7:
        st.metric("Available Records", len(info_data))

    # ---------------- MOVING AVERAGE CHART ----------------
    st.subheader("📈 Historical Chart with Moving Averages")

    chart_type_info = st.selectbox(
        "Select Chart Type",
        ["Line", "Candlestick"],
        key="info_chart_type"
    )

    info_chart_df = info_data.reset_index()

    if chart_type_info == "Line":
        fig_info = go.Figure()

        fig_info.add_trace(go.Scatter(
            x=info_chart_df["Date"],
            y=info_chart_df["Close"],
            mode="lines",
            name="Close Price"
        ))

        fig_info.add_trace(go.Scatter(
            x=info_chart_df["Date"],
            y=info_chart_df["MA20"],
            mode="lines",
            name="MA20"
        ))

        fig_info.add_trace(go.Scatter(
            x=info_chart_df["Date"],
            y=info_chart_df["MA50"],
            mode="lines",
            name="MA50"
        ))

        fig_info.update_layout(
            title=f"{info_stock_name} Historical Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price"
        )

    else:
        fig_info = go.Figure(data=[go.Candlestick(
            x=info_chart_df["Date"],
            open=info_chart_df["Open"],
            high=info_chart_df["High"],
            low=info_chart_df["Low"],
            close=info_chart_df["Close"],
            name="Candlestick"
        )])

        fig_info.add_trace(go.Scatter(
            x=info_chart_df["Date"],
            y=info_chart_df["MA20"],
            mode="lines",
            name="MA20"
        ))

        fig_info.add_trace(go.Scatter(
            x=info_chart_df["Date"],
            y=info_chart_df["MA50"],
            mode="lines",
            name="MA50"
        ))

        fig_info.update_layout(
            title=f"{info_stock_name} Historical Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price"
        )

    st.plotly_chart(fig_info, use_container_width=True)

    # ---------------- DAILY RETURNS CHART ----------------
    st.subheader("📉 Daily Returns")

    returns_df = info_chart_df.dropna(subset=["Daily Return %"]).copy()

    fig_returns = px.line(
        returns_df,
        x="Date",
        y="Daily Return %",
        title=f"{info_stock_name} Daily Return Percentage"
    )
    st.plotly_chart(fig_returns, use_container_width=True)

    # ---------------- VOLATILITY CHART ----------------
    st.subheader("📌 Volatility Trend")

    volatility_df = info_chart_df.dropna(subset=["Volatility"]).copy()

    fig_vol = px.line(
        volatility_df,
        x="Date",
        y="Volatility",
        title=f"{info_stock_name} Rolling Volatility"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ---------------- HISTORICAL DATA TABLE ----------------
    st.subheader("📋 Full Historical Data")

    display_df = info_chart_df.copy()
    display_df = display_df.round(2)

    st.dataframe(
        display_df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "MA20",
                "MA50",
                "Daily Return %",
                "Volatility"
            ]
        ],
        use_container_width=True,
        height=500
    )

    # ---------------- DOWNLOAD OPTION ----------------
    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Historical Data as CSV",
        data=csv_data,
        file_name=f"{info_stock_name.lower().replace(' ', '_')}_historical_data.csv",
        mime="text/csv"
    )
