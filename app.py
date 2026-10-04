import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

try:
    import ta
except ImportError:
    ta = None

try:
    from hyperopt import fmin, tpe, hp, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="HOSE Quant: SMA + OBV + MPT Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sleek financial dashboard look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-bull { background-color: #DCFCE7; color: #166534; }
    .badge-bear { background-color: #FEE2E2; color: #991B1B; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. CORE QUANTITATIVE FUNCTIONS
# ==========================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính On-Balance Volume (OBV) thuần túy bằng NumPy & Pandas."""
    if ta is not None:
        try:
            return ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        except Exception:
            pass
    # Fallback thuần vector hóa
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    direction[0] = 0.0
    return pd.Series((direction * volume).cumsum(), index=close.index, name="OBV")


def find_position_sma(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tạo tín hiệu giao dịch theo SMA Crossover."""
    position = pd.Series(0.0, index=df.index, name="position")
    ma_short = int(paras.get("ma_short", 20))
    ma_long = int(paras.get("ma_long", 200))

    if ma_short >= ma_long:
        return position

    ma_s = df["Close"].rolling(window=ma_short).mean()
    ma_l = df["Close"].rolling(window=ma_long).mean()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_obv(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tạo tín hiệu giao dịch theo OBV Breakout."""
    position = pd.Series(0.0, index=df.index, name="position")
    obv_window = int(paras.get("obv_window", 20))

    obv = calculate_obv(df["Close"], df["Volume"])
    obv_ma = obv.rolling(window=obv_window).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_combined_and(df: pd.DataFrame, sma_paras: dict, obv_paras: dict) -> pd.Series:
    """Kết hợp tín hiệu AND: cả 2 chỉ báo cùng kích hoạt."""
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    position.loc[(sma == 1.0) & (obv == 1.0)] = 1.0
    position.loc[(sma == -1.0) & (obv == -1.0)] = -1.0
    return position


def find_position_combined_or(df: pd.DataFrame, sma_paras: dict, obv_paras: dict) -> pd.Series:
    """Kết hợp tín hiệu OR: ít nhất 1 chỉ báo kích hoạt, mâu thuẫn -> trung lập (0)."""
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    buy = (sma == 1.0) | (obv == 1.0)
    sell = (sma == -1.0) | (obv == -1.0)

    conflict = buy & sell
    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


def events_to_holding(events: pd.Series) -> pd.Series:
    """Chuyển đổi các sự kiện mua (1), bán (-1) thành trạng thái nắm giữ (0 hoặc 1)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, signal in enumerate(events):
        if signal == 1.0:
            current = 1.0
        elif signal == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df: pd.DataFrame, events: pd.Series, commission: float = 0.0):
    """
    Tính lợi nhuận chiến lược với cơ chế dịch 1 phiên (shift 1) 
    để loại bỏ hoàn toàn Look-ahead Bias.
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)

    # Shift 1: Tín hiệu phiên t chỉ có hiệu lực giao dịch tại phiên t+1
    executed_holding = holding.shift(1).fillna(0.0)
    strat_ret = executed_holding * asset_ret

    # Chi phí giao dịch theo thay đổi vị thế (turnover)
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding


def performance_stats(returns: pd.Series, trading_days: int = 252) -> dict:
    """Tính các chỉ số đo lường hiệu suất tài chính."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan,
            "Win Rate [%]": np.nan
        }

    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0

    years = len(r) / trading_days
    annual_return = (
        equity.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    annual_vol = r.std() * np.sqrt(trading_days)
    sharpe = (
        r.mean() / r.std() * np.sqrt(trading_days)
        if r.std() > 0
        else np.nan
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    active_trades = r[r != 0]
    win_rate = (len(active_trades[active_trades > 0]) / len(active_trades) * 100) if len(active_trades) > 0 else 0.0

    return {
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0,
        "Win Rate [%]": win_rate
    }


# ==========================================
# 2. HYPEROPT PARAMETER OPTIMIZATION (TRAIN ONLY)
# ==========================================

def score_sma(paras: dict, df_train: pd.DataFrame, commission: float = 0.0) -> float:
    paras = {
        "ma_short": int(paras["ma_short"]),
        "ma_long": int(paras["ma_long"])
    }
    if paras["ma_short"] >= paras["ma_long"]:
        return 999999.0

    events = find_position_sma(df_train, paras)
    ret, _ = strategy_returns(df_train, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]

    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def score_obv(paras: dict, df_train: pd.DataFrame, commission: float = 0.0) -> float:
    paras = {"obv_window": int(paras["obv_window"])}
    events = find_position_obv(df_train, paras)
    ret, _ = strategy_returns(df_train, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]

    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def optimize_one_stock_hyperopt(df_train: pd.DataFrame, max_evals: int = 30, commission: float = 0.0):
    if not HAS_HYPEROPT:
        # Fallback nếu máy chủ chưa cài hyperopt
        return {"ma_short": 50, "ma_long": 200}, {"obv_window": 20}

    space_sma = {
        "ma_short": hp.quniform("ma_short", 25, 150, 5),
        "ma_long": hp.quniform("ma_long", 200, 400, 5)
    }
    trials_sma = Trials()
    best_sma_raw = fmin(
        fn=lambda p: score_sma(p, df_train, commission),
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_sma,
        verbose=False
    )
    sma_best = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }

    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 100, 5)
    }
    trials_obv = Trials()
    best_obv_raw = fmin(
        fn=lambda p: score_obv(p, df_train, commission),
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_obv,
        verbose=False
    )
    obv_best = {
        "obv_window": int(best_obv_raw["obv_window"])
    }
    return sma_best, obv_best


# ==========================================
# 3. MODERN PORTFOLIO THEORY (MPT)
# ==========================================

def portfolio_annual_return(weights: np.ndarray, returns: pd.DataFrame, trading_days: int = 252) -> float:
    mean_daily = returns.mean().values
    return float(weights @ mean_daily * trading_days)


def portfolio_annual_volatility(weights: np.ndarray, returns: pd.DataFrame, trading_days: int = 252) -> float:
    cov_annual = returns.cov().values * trading_days
    variance = float(weights.T @ cov_annual @ weights)
    return np.sqrt(max(variance, 0.0))


def negative_sharpe(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    p_return = portfolio_annual_return(weights, returns, trading_days)
    p_vol = portfolio_annual_volatility(weights, returns, trading_days)
    if p_vol <= 1e-8:
        return 1e9
    return -(p_return - risk_free_rate) / p_vol


def optimize_mpt(returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> np.ndarray:
    """Tối ưu hóa danh mục MPT (Max Sharpe Ratio, Long-Only, Không Short)."""
    n = returns.shape[1]
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        result = minimize(
            negative_sharpe,
            x0=x0,
            args=(returns, risk_free_rate, trading_days),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if result.success and not np.isnan(result.x).any():
            weights = np.maximum(result.x, 0.0)
            return weights / weights.sum()
    except Exception:
        pass
    # Fallback sang Equal Weight nếu tối ưu không hội tụ
    return x0


def portfolio_returns(return_matrix: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return return_matrix.mul(weights, axis=1).sum(axis=1)


# ==========================================
# 4. DATA LOADING & PREPARATION
# ==========================================

@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes_or_path):
    """Đọc và chuẩn hóa dữ liệu giao dịch HOSE."""
    if hasattr(file_bytes_or_path, "seek"):
        file_bytes_or_path.seek(0)

    if isinstance(file_bytes_or_path, str):
        df_full = pd.read_csv(file_bytes_or_path, encoding="utf-8-sig", low_memory=False)
    else:
        df_full = pd.read_csv(file_bytes_or_path, encoding="utf-8-sig", low_memory=False)

    df_full.columns = df_full.columns.str.strip().str.lower()
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

    df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
    df_full["ticker"] = df_full["ticker"].astype(str).str.strip().str.upper()
    df_full = df_full.dropna(subset=["date", "ticker", "close", "volume"])
    return df_full


def prepare_stock_data(df_full: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df_full[df_full["ticker"] == ticker.upper()].copy()
    if df.empty:
        raise ValueError(f"Không tìm thấy mã {ticker} trong dữ liệu.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].set_index("date")
    return df


# ==========================================
# 5. STREAMLIT APP USER INTERFACE
# ==========================================

def main():
    st.markdown('<div class="main-title">📈 HOSE Quant Backtesting Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Kiểm định chiến lược giao dịch kết hợp <b>SMA (Trend) + OBV (Volume)</b> '
        'và phân bổ danh mục <b>Equal Weight vs Modern Portfolio Theory (MPT)</b> trên sàn HOSE.'
        '</div>', 
        unsafe_allow_html=True
    )

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.header("⚙️ Thiết Lập Tham Số")

        # 1. Nguồn Dữ Liệu
        st.subheader("1. Nguồn dữ liệu")
        default_file = "HOSE_2020_2023_in.csv"
        alternative_default = "HOSE_2020_2023_in(1).csv"
        found_path = None
        for p in [default_file, alternative_default, os.path.join(".", default_file)]:
            if os.path.exists(p):
                found_path = p
                break

        uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu", type=["csv"])
        data_source = None
        if uploaded_file is not None:
            data_source = uploaded_file
            st.success("✅ Đang dùng file tải lên")
        elif found_path:
            data_source = found_path
            st.info(f"📁 Dùng dữ liệu mẫu: `{found_path}`")
        else:
            st.warning("⚠️ Chưa tìm thấy file dữ liệu CSV mặc định. Vui lòng tải file lên.")

        if not data_source:
            st.stop()

        try:
            df_full = load_csv_data(data_source)
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")
            st.stop()

        all_tickers = sorted(df_full["ticker"].dropna().unique().tolist())
        default_selected = [t for t in ["ACB", "FPT", "HPG"] if t in all_tickers]
        if len(default_selected) < 3:
            default_selected = all_tickers[:3]

        # 2. Chọn Cổ Phiếu
        st.subheader("2. Chọn cổ phiếu danh mục")
        selected_tickers = st.multiselect(
            "Cổ phiếu (tối thiểu 2 mã)",
            options=all_tickers,
            default=default_selected,
            help="Theo notebook gốc, khuyến nghị chọn 3 cổ phiếu đại diện cho 3 ngành khác nhau (VD: Ngân hàng, Công nghệ, Thép)."
        )

        if len(selected_tickers) < 2:
            st.error("Vui lòng chọn ít nhất 2 cổ phiếu để phân bổ danh mục!")
            st.stop()

        # 3. Phân chia Train / Test
        st.subheader("3. Phân chia Train & Test")
        min_date = df_full["date"].min().date()
        max_date = df_full["date"].max().date()

        # Giới hạn giá trị mặc định luôn nằm an toàn trong khoảng [min_date, max_date]
        target_tr_start = pd.to_datetime("2020-01-01").date()
        target_tr_end   = pd.to_datetime("2021-12-31").date()
        target_te_start = pd.to_datetime("2022-01-01").date()
        target_te_end   = pd.to_datetime("2022-12-31").date()

        d_train_start = max(min_date, min(target_tr_start, max_date))
        d_train_end   = max(min_date, min(target_tr_end, max_date))
        d_test_start  = max(min_date, min(target_te_start, max_date))
        d_test_end    = max(min_date, min(target_te_end, max_date))

        col_tr1, col_tr2 = st.columns(2)
        with col_tr1:
            train_start = st.date_input("Train Start", d_train_start)
        with col_tr2:
            train_end = st.date_input("Train End", d_train_end)

        col_te1, col_te2 = st.columns(2)
        with col_te1:
            test_start = st.date_input("Test Start", d_test_start)
        with col_te2:
            test_end = st.date_input("Test End", d_test_end)

        # 4. Giả định giao dịch
        st.subheader("4. Giả định giao dịch")
        initial_capital = st.number_input("Vốn khởi đầu (VND)", min_value=1_000, value=100_000_000, step=10_000_000, format="%d")
        commission = st.slider("Phí giao dịch mỗi chiều (%)", min_value=0.0, max_value=0.5, value=0.0, step=0.05) / 100.0
        risk_free_rate = st.slider("Lãi suất phi rủi ro Rf (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.5) / 100.0

        portfolio_mode = st.radio(
            "Cách kết hợp tín hiệu cho Danh mục:",
            options=["OR", "AND"],
            index=0,
            help="OR: Mua nếu SMA hoặc OBV phát tín hiệu. AND: Mua chỉ khi cả SMA và OBV cùng đồng thuận."
        )

        # 5. Phương pháp tham số
        st.subheader("5. Tối ưu tham số (Train)")
        opt_mode = st.radio(
            "Chế độ chọn tham số:",
            options=["Tham số tối ưu chuẩn (Khuyên dùng)", "Chạy Hyperopt TPE", "Nhập thủ công"],
            index=0
        )

        hyperopt_evals = 30
        if opt_mode == "Chạy Hyperopt TPE":
            if not HAS_HYPEROPT:
                st.warning("⚠️ Thư viện `hyperopt` chưa được cài đặt. Hệ thống sẽ tự động dùng tham số mẫu.")
            else:
                hyperopt_evals = st.slider("Số vòng lặp Hyperopt (max_evals)", 15, 60, 30, step=5)

    # ------------------ PREPARE DATA ------------------
    stock_data = {}
    train_data = {}
    test_data = {}

    for ticker in selected_tickers:
        try:
            df_stock = prepare_stock_data(df_full, ticker)
            tr = df_stock.loc[(df_stock.index >= pd.to_datetime(train_start)) & (df_stock.index <= pd.to_datetime(train_end))].copy()
            te = df_stock.loc[(df_stock.index >= pd.to_datetime(test_start)) & (df_stock.index <= pd.to_datetime(test_end))].copy()

            if tr.empty or te.empty:
                st.error(f"Mã {ticker} không đủ dữ liệu trong giai đoạn Train hoặc Test đã chọn.")
                st.stop()

            stock_data[ticker] = df_stock
            train_data[ticker] = tr
            test_data[ticker] = te
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu mã {ticker}: {e}")
            st.stop()

    # Notebook standard baseline parameters (found via Hyperopt in research)
    standard_params = {
        "ACB": {"SMA": {"ma_short": 60, "ma_long": 270}, "OBV": {"obv_window": 5}},
        "FPT": {"SMA": {"ma_short": 70, "ma_long": 205}, "OBV": {"obv_window": 65}},
        "HPG": {"SMA": {"ma_short": 25, "ma_long": 220}, "OBV": {"obv_window": 40}}
    }

    best_params = {}

    if opt_mode == "Tham số tối ưu chuẩn (Khuyên dùng)":
        for ticker in selected_tickers:
            if ticker in standard_params:
                best_params[ticker] = standard_params[ticker]
            else:
                best_params[ticker] = {
                    "SMA": {"ma_short": 50, "ma_long": 200},
                    "OBV": {"obv_window": 20}
                }

    elif opt_mode == "Chạy Hyperopt TPE" and HAS_HYPEROPT:
        with st.spinner("⏳ Đang chạy Hyperopt tối ưu tham số trên dữ liệu TRAIN (2020-2021)..."):
            progress_bar = st.progress(0)
            total = len(selected_tickers)
            for idx, ticker in enumerate(selected_tickers):
                sma_b, obv_b = optimize_one_stock_hyperopt(train_data[ticker], max_evals=hyperopt_evals, commission=commission)
                best_params[ticker] = {"SMA": sma_b, "OBV": obv_b}
                progress_bar.progress((idx + 1) / total)
            st.success("✅ Hoàn tất tối ưu tham số Hyperopt!")

    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Cấu hình thủ công từng mã:**")
        for ticker in selected_tickers:
            with st.sidebar.expander(f"Mã {ticker}", expanded=False):
                s_ma = st.number_input(f"[{ticker}] SMA Ngắn", 5, 150, 50, key=f"s_{ticker}")
                l_ma = st.number_input(f"[{ticker}] SMA Dài", 100, 400, 200, key=f"l_{ticker}")
                obv_w = st.number_input(f"[{ticker}] OBV MA Window", 3, 150, 20, key=f"o_{ticker}")
                best_params[ticker] = {
                    "SMA": {"ma_short": int(s_ma), "ma_long": int(l_ma)},
                    "OBV": {"obv_window": int(obv_w)}
                }

    # ------------------ BACKTEST COMPUTATION ------------------
    def run_evaluations(df, sma_p, obv_p):
        ev_sma = find_position_sma(df, sma_p)
        r_sma, _ = strategy_returns(df, ev_sma, commission)

        ev_obv = find_position_obv(df, obv_p)
        r_obv, _ = strategy_returns(df, ev_obv, commission)

        ev_and = find_position_combined_and(df, sma_p, obv_p)
        r_and, _ = strategy_returns(df, ev_and, commission)

        ev_or = find_position_combined_or(df, sma_p, obv_p)
        r_or, _ = strategy_returns(df, ev_or, commission)

        # Benchmark Buy & Hold
        r_bh = df["Close"].pct_change().fillna(0.0)

        strats = {
            "SMA": r_sma,
            "OBV": r_obv,
            "SMA + OBV AND": r_and,
            "SMA + OBV OR": r_or,
            "Buy & Hold": r_bh
        }
        stats = pd.DataFrame({k: performance_stats(v) for k, v in strats.items()}).T
        return strats, stats

    train_results = {}
    test_results = {}

    for ticker in selected_tickers:
        sma_p = best_params[ticker]["SMA"]
        obv_p = best_params[ticker]["OBV"]

        tr_strats, tr_stats = run_evaluations(train_data[ticker], sma_p, obv_p)
        te_strats, te_stats = run_evaluations(test_data[ticker], sma_p, obv_p)

        train_results[ticker] = {"returns": tr_strats, "stats": tr_stats}
        test_results[ticker] = {"returns": te_strats, "stats": te_stats}

    # Strategy name for portfolio
    strat_key = f"SMA + OBV {portfolio_mode.upper()}"

    train_return_matrix = pd.concat(
        {t: train_results[t]["returns"][strat_key] for t in selected_tickers},
        axis=1
    ).dropna()

    test_return_matrix = pd.concat(
        {t: test_results[t]["returns"][strat_key] for t in selected_tickers},
        axis=1
    ).dropna()

    # Equal Weights
    n_assets = len(selected_tickers)
    equal_weights = np.repeat(1.0 / n_assets, n_assets)

    # MPT Weights (Estimated strictly on Train)
    mpt_weights = optimize_mpt(train_return_matrix, risk_free_rate=risk_free_rate)

    # Calculate Portfolio returns
    ew_train_ret = portfolio_returns(train_return_matrix, equal_weights)
    mpt_train_ret = portfolio_returns(train_return_matrix, mpt_weights)

    ew_test_ret = portfolio_returns(test_return_matrix, equal_weights)
    mpt_test_ret = portfolio_returns(test_return_matrix, mpt_weights)

    portfolio_train_stats = pd.DataFrame({
        "Equal Weight (1/N)": performance_stats(ew_train_ret),
        "MPT (Max Sharpe Train)": performance_stats(mpt_train_ret)
    }).T

    portfolio_test_stats = pd.DataFrame({
        "Equal Weight (1/N)": performance_stats(ew_test_ret),
        "MPT (Max Sharpe Train)": performance_stats(mpt_test_ret)
    }).T

    # ==========================================
    # 6. APP MAIN PRESENTATION TABS
    # ==========================================

    tabs = st.tabs([
        "💼 Phân Bổ Danh Mục (EW vs MPT)",
        "📊 Chi Tiết Từng Cổ Phiếu",
        "⚙️ Tham Số & Dữ Liệu",
        "📑 Báo Cáo Nghiên Cứu",
        "📖 Phương Pháp Luận"
    ])

    # ---------------- TAB 1: PORTFOLIO ALLOCATION ----------------
    with tabs[0]:
        st.subheader("🎯 Đánh Giá Danh Mục Ngoài Mẫu (Out-of-Sample: TEST 2022)")
        st.markdown(
            "Cả hai mô hình phân bổ (**Equal Weight** và **MPT**) đều được xác định tỷ trọng **DUY NHẤT** "
            "từ dữ liệu quá khứ **Train (2020-2021)**, sau đó áp dụng cố định sang **Test (2022)** để kiểm định tính bền vững."
        )

        # Highlight KPI metrics on Test
        col1, col2, col3, col4 = st.columns(4)
        ew_tot_ret = portfolio_test_stats.loc["Equal Weight (1/N)", "Total Return [%]"]
        mpt_tot_ret = portfolio_test_stats.loc["MPT (Max Sharpe Train)", "Total Return [%]"]
        ew_sharpe = portfolio_test_stats.loc["Equal Weight (1/N)", "Sharpe Ratio"]
        mpt_sharpe = portfolio_test_stats.loc["MPT (Max Sharpe Train)", "Sharpe Ratio"]

        with col1:
            st.metric("Total Return (Equal Weight)", f"{ew_tot_ret:.2f}%")
        with col2:
            st.metric("Total Return (MPT)", f"{mpt_tot_ret:.2f}%", delta=f"{mpt_tot_ret - ew_tot_ret:.2f}% so với EW")
        with col3:
            st.metric("Sharpe Ratio (Equal Weight)", f"{ew_sharpe:.3f}" if not pd.isna(ew_sharpe) else "N/A")
        with col4:
            st.metric("Sharpe Ratio (MPT)", f"{mpt_sharpe:.3f}" if not pd.isna(mpt_sharpe) else "N/A")

        # Interactive Portfolio Equity Curve Plot (Test 2022)
        ew_equity_test = initial_capital * (1.0 + ew_test_ret).cumprod()
        mpt_equity_test = initial_capital * (1.0 + mpt_test_ret).cumprod()

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=ew_equity_test.index, y=ew_equity_test,
            mode='lines', name='Equal Weight (1/N)',
            line=dict(color='#2563EB', width=2.5)
        ))
        fig_port.add_trace(go.Scatter(
            x=mpt_equity_test.index, y=mpt_equity_test,
            mode='lines', name='MPT (Max Sharpe Train)',
            line=dict(color='#DC2626', width=2.5)
        ))

        fig_port.update_layout(
            title=f"Đường Tăng Trưởng Vốn (Equity Curve) — TEST 2022 (Chiến lược: {strat_key})",
            xaxis_title="Thời Gian",
            yaxis_title="Giá Trị Danh Mục (VND)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=480
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # Drawdown Chart (Underwater plot)
        ew_dd = (ew_equity_test / ew_equity_test.cummax() - 1.0) * 100.0
        mpt_dd = (mpt_equity_test / mpt_equity_test.cummax() - 1.0) * 100.0

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=ew_dd.index, y=ew_dd,
            fill='tozeroy', mode='lines', name='Drawdown Equal Weight (%)',
            line=dict(color='#3B82F6', width=1.5)
        ))
        fig_dd.add_trace(go.Scatter(
            x=mpt_dd.index, y=mpt_dd,
            fill='tozeroy', mode='lines', name='Drawdown MPT (%)',
            line=dict(color='#EF4444', width=1.5)
        ))
        fig_dd.update_layout(
            title="Biểu Đồ Sụt Giảm Tối Đa (Drawdown Under-Water) — Giai Đoạn TEST 2022",
            xaxis_title="Thời Gian",
            yaxis_title="Sụt Giảm (%)",
            template="plotly_white",
            hovermode="x unified",
            height=300
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Performance Comparison Tables
        col_tb1, col_tb2 = st.columns(2)
        with col_tb1:
            st.markdown("#### 📉 Bảng Hiệu Suất Ngoài Mẫu: TEST (2022)")
            st.dataframe(portfolio_test_stats.style.format("{:.2f}", na_rep="N/A").highlight_max(subset=["Total Return [%]", "Sharpe Ratio"], color="#DCFCE7"), use_container_width=True)

        with col_tb2:
            st.markdown("#### 📈 Bảng Hiệu Suất Trong Mẫu: TRAIN (2020-2021)")
            st.dataframe(portfolio_train_stats.style.format("{:.2f}", na_rep="N/A"), use_container_width=True)

        # Weights Comparison
        st.markdown("---")
        st.subheader("⚖️ So Sánh Phân Bổ Trọng Số: Equal Weight vs MPT")
        col_w1, col_w2 = st.columns([1, 1.4])

        weight_df = pd.DataFrame({
            "Equal Weight": equal_weights * 100.0,
            "MPT (Train)": mpt_weights * 100.0
        }, index=selected_tickers)

        with col_w1:
            st.markdown("**Bảng phân bổ tỷ trọng (%):**")
            st.dataframe(weight_df.style.format("{:.2f}%"), use_container_width=True)

        with col_w2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=selected_tickers, y=weight_df["Equal Weight"],
                name="Equal Weight (1/N)", marker_color="#3B82F6"
            ))
            fig_bar.add_trace(go.Bar(
                x=selected_tickers, y=weight_df["MPT (Train)"],
                name="MPT (Max Sharpe)", marker_color="#EF4444"
            ))
            fig_bar.update_layout(
                title="Cơ Cấu Tỷ Trọng Danh Mục",
                yaxis_title="Tỷ Trọng (%)",
                barmode='group',
                template="plotly_white",
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- TAB 2: SINGLE-STOCK ANALYSIS ----------------
    with tabs[1]:
        st.subheader("🔍 Phân Tích Kỹ Thuật & Tín Hiệu Giao Dịch Từng Mã")
        stock_to_inspect = st.selectbox("Chọn cổ phiếu để xem chi tiết:", selected_tickers)

        st_df = stock_data[stock_to_inspect].copy()
        s_best = best_params[stock_to_inspect]["SMA"]
        o_best = best_params[stock_to_inspect]["OBV"]

        col_st1, col_st2, col_st3 = st.columns(3)
        col_st1.metric(f"SMA Ngắn ({stock_to_inspect})", f"{s_best['ma_short']} phiên")
        col_st2.metric(f"SMA Dài ({stock_to_inspect})", f"{s_best['ma_long']} phiên")
        col_st3.metric(f"OBV Moving Average ({stock_to_inspect})", f"{o_best['obv_window']} phiên")

        # Visualizing Price with SMA lines and Buy/Sell markers
        st_df["SMA_S"] = st_df["Close"].rolling(s_best["ma_short"]).mean()
        st_df["SMA_L"] = st_df["Close"].rolling(s_best["ma_long"]).mean()
        st_df["OBV"] = calculate_obv(st_df["Close"], st_df["Volume"])
        st_df["OBV_MA"] = st_df["OBV"].rolling(o_best["obv_window"]).mean()

        events_strat = find_position_combined_or(st_df, s_best, o_best) if portfolio_mode == "OR" else find_position_combined_and(st_df, s_best, o_best)
        buy_points = st_df[events_strat == 1.0]
        sell_points = st_df[events_strat == -1.0]

        # Multi-panel chart: Price & SMA top, OBV bottom
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                                 subplot_titles=(f"Đồ Thị Giá & Điểm Vào Ra Tín Hiệu ({stock_to_inspect})", f"Chỉ Báo Khối Lượng OBV & OBV-MA ({stock_to_inspect})"))

        fig_tech.add_trace(go.Scatter(x=st_df.index, y=st_df["Close"], name="Giá Đóng Cửa", line=dict(color="#1E293B", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=st_df.index, y=st_df["SMA_S"], name=f"SMA Ngắn ({s_best['ma_short']})", line=dict(color="#06B6D4", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=st_df.index, y=st_df["SMA_L"], name=f"SMA Dài ({s_best['ma_long']})", line=dict(color="#8B5CF6", width=1.5)), row=1, col=1)

        if not buy_points.empty:
            fig_tech.add_trace(go.Scatter(
                x=buy_points.index, y=buy_points["Close"], mode="markers", name="Tín hiệu MUA",
                marker=dict(symbol="triangle-up", size=10, color="#16A34A")
            ), row=1, col=1)

        if not sell_points.empty:
            fig_tech.add_trace(go.Scatter(
                x=sell_points.index, y=sell_points["Close"], mode="markers", name="Tín hiệu BÁN",
                marker=dict(symbol="triangle-down", size=10, color="#DC2626")
            ), row=1, col=1)

        # OBV row
        fig_tech.add_trace(go.Scatter(x=st_df.index, y=st_df["OBV"], name="OBV", line=dict(color="#F59E0B", width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=st_df.index, y=st_df["OBV_MA"], name=f"OBV MA ({o_best['obv_window']})", line=dict(color="#64748B", width=1.2, dash="dash")), row=2, col=1)

        fig_tech.update_layout(height=600, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_tech, use_container_width=True)

        # Cumulative returns comparison for this stock
        st.markdown(f"#### 🚀 Tăng Trưởng Lũy Kế Các Chiến Lược ({stock_to_inspect})")
        r_dict = test_results[stock_to_inspect]["returns"]
        fig_cum = go.Figure()
        colors = {"SMA": "#06B6D4", "OBV": "#F59E0B", "SMA + OBV AND": "#8B5CF6", "SMA + OBV OR": "#10B981", "Buy & Hold": "#94A3B8"}
        for s_name, s_ret in r_dict.items():
            cum = (1.0 + s_ret).cumprod()
            fig_cum.add_trace(go.Scatter(x=cum.index, y=cum, name=s_name, line=dict(color=colors.get(s_name, "#000000"), width=2)))

        fig_cum.update_layout(title=f"Lợi Nhuận Tích Lũy Trên TEST (2022) — {stock_to_inspect}", yaxis_title="Hệ số tăng trưởng vốn (x lần)", template="plotly_white", height=380)
        st.plotly_chart(fig_cum, use_container_width=True)

        # Side by side stats
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**Bảng Thống Kê TRAIN (2020-2021) — {stock_to_inspect}:**")
            st.dataframe(train_results[stock_to_inspect]["stats"].style.format("{:.2f}", na_rep="N/A"), use_container_width=True)
        with col_s2:
            st.markdown(f"**Bảng Thống Kê TEST (2022) — {stock_to_inspect}:**")
            st.dataframe(test_results[stock_to_inspect]["stats"].style.format("{:.2f}", na_rep="N/A"), use_container_width=True)

    # ---------------- TAB 3: PARAMETERS & RAW DATA ----------------
    with tabs[2]:
        st.subheader("📋 Bảng Tham Số Tối Ưu Hóa & Dữ Liệu Lịch Sử")
        st.markdown("Bảng tổng hợp tham số được xác định trên tập **Train (2020-2021)**:")

        params_summary = pd.DataFrame({
            t: {
                "SMA Short (phiên)": best_params[t]["SMA"]["ma_short"],
                "SMA Long (phiên)": best_params[t]["SMA"]["ma_long"],
                "OBV Window (phiên)": best_params[t]["OBV"]["obv_window"]
            }
            for t in selected_tickers
        }).T
        st.dataframe(params_summary, use_container_width=True)

        st.markdown("#### 📂 Dữ Liệu Giá Gốc (Preview 100 phiên đầu)")
        st.dataframe(df_full[df_full["ticker"].isin(selected_tickers)].head(100), use_container_width=True)

    # ---------------- TAB 4: RESEARCH REPORT & EXPORT ----------------
    with tabs[3]:
        st.subheader("📑 Báo Cáo Nghiên Cứu Định Lượng & Kết Luận")
        st.markdown("""
        ### 1. Tổng Kết Kết Quả Nghiên Cứu Thực Nghiệm:
        - **Bối Cảnh Thị Trường**:
            - **Giai đoạn Train (2020 - 2021)**: Thị trường HOSE bùng nổ (Bull Market mạnh mẽ hậu Covid), dòng tiền cá nhân F0 đổ vào lớn. Các chiến lược theo xu hướng (Trend Following SMA) và dòng tiền (OBV) đều đạt lợi nhuận rất cao (HPG tăng hơn 200%, ACB và FPT tăng >80%).
            - **Giai đoạn Test (2022)**: Thị trường bước vào giai đoạn Downtrend khốc liệt do thắt chặt chính sách tiền tệ và sự kiện trái phiếu. Hầu hết các cổ phiếu HOSE giảm từ 30% đến 60%.
        - **Hiệu Quả Chiến Lược Từng Mã**:
            - Khi kết hợp **SMA + OBV OR**, chiến lược kích hoạt vị thế linh hoạt hơn, nắm bắt tốt các nhịp tăng trong năm 2020-2021.
            - Khi kết hợp **SMA + OBV AND**, điều kiện vào lệnh quá khắt khe (đòi hỏi cả 2 chỉ báo cùng lúc đảo chiều), dẫn tới số lượng giao dịch rất ít, thường giữ tiền mặt.
        - **So Sánh Danh Mục: Equal Weight vs MPT (Markowitz)**:
            - **Trên Train**: MPT tối ưu hóa Sharpe cao hơn Equal Weight nhờ dồn tỷ trọng vào các mã có tỷ suất sinh lời/rủi ro cao nhất trong quá khứ.
            - **Trên Test (Out-of-sample)**: **Equal Weight (1/N)** thường thể hiện sự bền bỉ và phòng vệ tốt hơn hoặc tương đương MPT. Điều này phản ánh rõ hiện tượng **Estimation Risk / Overfitting**: các trọng số tối ưu hóa trên một chế độ thị trường (Bull Market) thường mất đi tính tối ưu khi thị trường đổi chiều (Bear Market).
        """)

        # Data export
        st.markdown("---")
        st.subheader("📥 Xuất Báo Cáo & Dữ Liệu")

        export_df = pd.concat([
            portfolio_test_stats.assign(Giai_Đoạn="TEST 2022"),
            portfolio_train_stats.assign(Giai_Đoạn="TRAIN 2020-2021")
        ])

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=True)

        st.download_button(
            label="💾 Tải Xuất Thống Kê Hiệu Suất (CSV)",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name="hose_quant_portfolio_performance.csv",
            mime="text/csv"
        )

    # ---------------- TAB 5: METHODOLOGY ----------------
    with tabs[4]:
        st.subheader("📖 Phương Pháp Luận & Quy Chuẩn Thực Nghiệm")
        st.markdown("""
        #### 1. Chỉ Báo Kỹ Thuật Được Sử Dụng
        - **Đường Trung Bình Động Giản Đơn (SMA - Simple Moving Average)**:
            - Xác định xu hướng giá trung và dài hạn.
            - Tín hiệu Mua (Buy = 1): Khi SMA ngắn cắt lên trên SMA dài (Golden Cross).
            - Tín hiệu Bán (Sell = -1): Khi SMA ngắn cắt xuống dưới SMA dài (Death Cross).
        - **Khối Lượng Cân Bằng (OBV - On-Balance Volume)**:
            - Đo lường áp lực tích lũy hoặc phân phối của dòng tiền thông qua khối lượng khớp lệnh.
            - Lập đường trung bình động $OBV\_MA$ với chu kỳ cửa sổ $W$.
            - Tín hiệu Mua (Buy = 1): Khi $OBV > OBV\_MA$ và phiên trước $OBV \le OBV\_MA$.
            - Tín hiệu Bán (Sell = -1): Khi $OBV < OBV\_MA$ và phiên trước $OBV \ge OBV\_MA$.

        #### 2. Cơ Chế Khử Nhiễu & Tránh Thiên Lệch Nhìn Về Tương Lai (Look-Ahead Bias)
        - **Dịch 1 Phiên (Lag Shift = 1)**: Tín hiệu giao dịch được tạo ra tại cuối ngày $t$ chỉ được áp dụng để tính toán lợi nhuận bắt đầu từ ngày $t+1$. Giả định giao dịch tại giá đóng cửa phiên kế tiếp, đảm bảo tính khả thi thực tế.
        - **Quản lý trạng thái vị thế (Long-Cash)**: Vị thế giữ ở mức 1 (Long) cho tới khi xuất hiện tín hiệu Bán (-1) để chuyển hoàn toàn về 0 (Tiền mặt). Không sử dụng bán khống (No Shorting) phù hợp quy định chứng khoán cơ sở HOSE.

        #### 3. Mô Hình Danh Mục Đầu Tư Hiện Đại (Markowitz MPT)
        - Tối đa hóa tỷ số Sharpe (Sharpe Ratio) trên tập dữ liệu huấn luyện **Train**:
        $$\\max_{w} \\quad \\frac{w^T \\mu - R_f}{\\sqrt{w^T \\Sigma w}}$$
        - Ràng buộc:
            - $\\sum_{i=1}^N w_i = 1$ (Tổng trọng số bằng 100%)
            - $0 \\le w_i \\le 1$ (Long-only, không sử dụng margin hay bán khống)
        - Giữ nguyên bộ trọng số $w$ này sang giai đoạn **Test** ngoài mẫu để kiểm nghiệm khách quan.
        """)


if __name__ == "__main__":
    main()
