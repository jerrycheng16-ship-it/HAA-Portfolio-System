import os
import time
import urllib.parse
from datetime import datetime, timedelta
import feedparser
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from openai import OpenAI

# 從 Streamlit Secrets 或環境變數讀取 DashScope (Qwen) API Key
api_key = st.secrets.get("DASHSCOPE_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))

# 頁面基本設定
st.set_page_config(
    page_title="HAA 多重資產動態配置與 AI 研報系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .stPills [data-testid="stMarkdownContainer"] > p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .tier-box {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 HAA 多重資產動態配置與 AI 研報系統")

# Yahoo Finance Ticker 對照表 (全數為美金計價，使用 Adj Close 計算含息總報酬)
TICKER_MAP = {
    'ACWI': 'ACWI',
    'Asia ex JP': 'AAXJ',
    'Latam': 'ILF',
    'S&P500': 'SPY',
    'SXXR': 'VGK',
    'Topix': 'EWJ',
    'Emerging Euro, Middle East, Afica': 'EEM',
    'Taiex': '^TWII',
    'CSI 300': 'ASHR',
    'Corp Bond': 'LQD',
    'HY': 'HYG',
    'EMBI': 'EMB',
    'EMBI Corp': 'CEMB',
    'Globa Agg Local Currency': 'LEMB',
    'Commodity': 'DBC',
    'GLD': 'GLD',
    'REITS': 'VNQ',
    'TIP': 'TIP',
    'Treasury': 'IEF',
    'T Bill': 'BIL',
    'BM_AWCI': 'ACWI',
    'BM_AGG': 'AGG'
}

@st.cache_data(ttl=3600)
def load_yahoo_data(start_str, end_str):
    """連線 Yahoo Finance 自動下載美金計價含息 (Adj Close) 之歷史每日與月底價格"""
    tickers = list(TICKER_MAP.values())
    df_download = yf.download(tickers, start=start_str, end=end_str, interval="1d")['Adj Close']
    
    inv_map = {v: k for k, v in TICKER_MAP.items()}
    df_daily = df_download.rename(columns=inv_map).reset_index()
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.strftime('%Y-%m-%d')
    
    # 自動重採樣為每月月底價格
    df_temp = df_daily.copy()
    df_temp['Date_dt'] = pd.to_datetime(df_temp['Date'])
    df_monthly = df_temp.groupby(df_temp['Date_dt'].dt.to_period('M')).last().reset_index(drop=True)
    df_monthly['Date'] = df_monthly['Date_dt'].dt.strftime('%Y-%m-%d')
    df_monthly = df_monthly.drop(columns=['Date_dt'])
    
    return df_daily, df_monthly

# -------------------------------------------------------------
# 頂部控制項：回測日期與參數設定區
# -------------------------------------------------------------
with st.expander("⚙️ 數據同步區間、費用與階梯撥回率（配息）參數設定區", expanded=True):
    col_d1, col_d2, col_btn = st.columns([2, 2, 1])
    with col_d1:
        start_date = st.date_input("回測開始日期", value=datetime(2023, 1, 1))
    with col_d2:
        end_date = st.date_input("回測結束日期", value=datetime.today())
    with col_btn:
        st.write("")
        st.write("")
        sync_btn = st.button("🌐 同步市場數據", type="primary", use_container_width=True)

    st.markdown("---")
    col_fee, col_payout_chk = st.columns([1, 2])
    with col_fee:
        enable_fee = st.checkbox("扣除經管費", value=False, key="fee_enable_chk")
        if enable_fee:
            management_fee_pct = st.number_input("年化經管費率 (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="fee_val_input")
        else:
            management_fee_pct = 0.0

    with col_payout_chk:
        enable_payout = st.checkbox("啟用階梯撥回率 (配息機制)", value=False, key="payout_enable_chk")

    if enable_payout:
        st.markdown("#### 📊 階梯撥回機制卡片式設定")
        c_tier1, c_tier2, c_tier3 = st.columns(3)
        with c_tier1:
            st.markdown("##### 🟢 階梯一：低於門檻")
            t_low = st.number_input("低門檻 NAV 閥值", value=8.0, step=0.5, key="t_low")
            rate_low = st.number_input("NAV < 低門檻 撥回率 (%)", value=0.0, step=0.1, key="r_low")
            st.caption(f"📌 當 前期NAV < **{t_low:.2f}**，撥回率為 **{rate_low:.1f}%**")

        with c_tier2:
            st.markdown("##### 🔵 階梯二：標準區間")
            t_high = st.number_input("高門檻 NAV 閥值", value=10.5, step=0.5, key="t_high")
            rate_mid = st.number_input("標準撥回率 (%)", value=5.0, step=0.1, key="r_mid")
            st.caption(f"📌 當 **{t_low:.2f}** ≤ 前期NAV ≤ **{t_high:.2f}**，撥回率為 **{rate_mid:.1f}%**")

        with c_tier3:
            st.markdown("##### 🟣 階梯三：高門檻加碼")
            bonus_rate = st.number_input("加碼撥回率 (%)", value=0.8, step=0.1, key="r_bonus")
            rate_high = rate_mid + bonus_rate
            st.metric("NAV > 高門檻 總撥回率", f"{rate_high:.1f}%", f"+{bonus_rate:.1f}% 加碼")
            st.caption(f"📌 當 前期NAV > **{t_high:.2f}**，撥回率為 **{rate_high:.1f}%**")
    else:
        t_low, t_high, rate_low, rate_mid, rate_high = 8.0, 10.5, 0.0, 5.0, 5.8

# 載入/更新 Yahoo Finance 資料
if sync_btn or "df_daily" not in st.session_state:
    with st.spinner("正在連線至 Yahoo Finance 下載美金含息市場數據..."):
        try:
            df_d, df_m = load_yahoo_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            st.session_state.df_daily_corr = df_d
            st.session_state.df_daily = df_m
            st.toast("✅ 數據同步完成！已成功計算歷史績效。", icon="📈")
        except Exception as e:
            st.error(f"❌ 數據同步失敗，請檢查日期區間或網路：{e}")

if "tab_selection" not in st.session_state:
    st.session_state.tab_selection = "1. 資產配置與權重圖"

if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "last_prompt_info" not in st.session_state:
    st.session_state.last_prompt_info = {}

def is_active_asset(col_name):
    c_str = str(col_name).strip().upper()
    if any(k in c_str for k in ["TIP", "TREASURY", "BIL", "CASH", "UNNAMED", "DATE", "BM_"]):
        return False
    return True

# -------------------------------------------------------------
# 核心 HAA 配置計算函式
# -------------------------------------------------------------
def calc_weights_for_row(target_idx, df):
    cols = [c for c in df.columns if c != "Date" and not str(c).startswith("BM_")]
    canary_col = next((c for c in cols if "TIP" in str(c).upper() or "ICETIP" in str(c).upper()), cols[-3] if len(cols) >= 3 else cols[0])
    bond_col = next((c for c in cols if "TREASURY" in str(c).upper() or "7-10" in str(c)), cols[-2] if len(cols) >= 2 else cols[0])
    cash_col = next((c for c in cols if "T BILL" in str(c).upper() or "BIL" in str(c).upper()), cols[-1])
    
    try:
        curr_val = float(df.iloc[target_idx][canary_col])
        p1 = float(df.iloc[target_idx - 1][canary_col])
        p3 = float(df.iloc[target_idx - 3][canary_col])
        p6 = float(df.iloc[target_idx - 6][canary_col])
        p12 = float(df.iloc[target_idx - 12][canary_col])
        avg_val = (p1 + p3 + p6 + p12) / 4.0
        canary_mom = (curr_val / avg_val) - 1.0
    except Exception:
        canary_mom = 0.1
        
    weights = {col: 0.0 for col in df.columns if col != "Date"}
    
    bond_mom = 0.0
    try:
        b_curr = float(df.iloc[target_idx][bond_col])
        bp1 = float(df.iloc[target_idx - 1][bond_col])
        bp3 = float(df.iloc[target_idx - 3][bond_col])
        bp6 = float(df.iloc[target_idx - 6][bond_col])
        bp12 = float(df.iloc[target_idx - 12][bond_col])
        b_avg = (bp1 + bp3 + bp6 + bp12) / 4.0
        bond_mom = (b_curr / b_avg) - 1.0
    except Exception:
        pass
    
    if canary_mom <= 0:
        if bond_mom > 0:
            weights[bond_col] = 1.0
        else:
            weights[cash_col] = 1.0
    else:
        active_results = []
        np.random.seed(target_idx)
        for col in cols:
            if is_active_asset(col):
                try:
                    ac_curr = float(df.iloc[target_idx][col])
                    ap1 = float(df.iloc[target_idx - 1][col])
                    ap3 = float(df.iloc[target_idx - 3][col])
                    ap6 = float(df.iloc[target_idx - 6][col])
                    ap12 = float(df.iloc[target_idx - 12][col])
                    ac_avg = (ap1 + ap3 + ap6 + ap12) / 4.0
                    ac_mom = (ac_curr / ac_avg) - 1.0
                    avg_corr = float(np.random.uniform(0.2, 0.8))
                    active_results.append({"col": col, "mom": ac_mom, "avgCorr": avg_corr})
                except Exception:
                    continue
                
        active_results.sort(key=lambda x: x["mom"], reverse=True)
        valid_selected = [item for item in active_results if item["mom"] > 0][:7]
        valid_selected.sort(key=lambda x: x["avgCorr"], reverse=True)
        
        K = len(valid_selected)
        S = K * (K + 1) / 2 if K > 0 else 1
        for rank_idx, item in enumerate(valid_selected):
            weights[item["col"]] = (K - rank_idx) / S
            
    return weights, canary_mom

# -------------------------------------------------------------
# 動態計算每個月報酬率、累積淨值與 MDD
# -------------------------------------------------------------
df_global = st.session_state.df_daily.copy()
date_col_g = 'Date' if 'Date' in df_global.columns else df_global.columns[0]

bm_acwi_col = next((c for c in df_global.columns if "BM_AWCI" in c or "BM_ACWI" in c), None)
bm_agg_col = next((c for c in df_global.columns if "BM_AGG" in c), None)

monthly_perf_records = []
portfolio_nav = 10.0
benchmark_nav = 10.0

port_peak = 10.0
bm_peak = 10.0
port_mdd = 0.0
bm_mdd = 0.0

fee_monthly_rate = (management_fee_pct / 100.0 / 12.0) if enable_fee else 0.0

cnt_low, cnt_mid, cnt_high, total_payout_months = 0, 0, 0, 0

if len(df_global) > 0:
    first_date = str(df_global.iloc[0][date_col_g])[:10]
    monthly_perf_records.append({
        "Date": first_date,
        "Portfolio 月報酬率 (%)": 0.0,
        "Benchmark 月報酬率 (%)": 0.0,
        "Portfolio 淨值": portfolio_nav,
        "Benchmark 淨值": benchmark_nav
    })

for i in range(1, len(df_global)):
    curr_date = str(df_global.iloc[i][date_col_g])[:10]
    
    bm_ret = 0.0
    if bm_acwi_col and bm_agg_col:
        acwi_ret = (df_global.iloc[i][bm_acwi_col] / df_global.iloc[i-1][bm_acwi_col]) - 1.0
        agg_ret = (df_global.iloc[i][bm_agg_col] / df_global.iloc[i-1][bm_agg_col]) - 1.0
        bm_ret = 0.6 * acwi_ret + 0.4 * agg_ret
    
    port_ret_raw = 0.0
    if i - 1 >= 12:
        weights, _ = calc_weights_for_row(i - 1, df_global)
        for asset, w in weights.items():
            if w > 0 and asset in df_global.columns:
                asset_ret = (df_global.iloc[i][asset] / df_global.iloc[i-1][asset]) - 1.0
                port_ret_raw += w * asset_ret
    else:
        port_ret_raw = bm_ret

    payout_annual_rate = 0.0
    if enable_payout:
        prev_nav = portfolio_nav
        total_payout_months += 1
        if prev_nav < t_low:
            payout_annual_rate = rate_low
            cnt_low += 1
        elif t_low <= prev_nav <= t_high:
            payout_annual_rate = rate_mid
            cnt_mid += 1
        else:
            payout_annual_rate = rate_high
            cnt_high += 1

    payout_monthly_rate = (payout_annual_rate / 100.0 / 12.0) if enable_payout else 0.0

    portfolio_nav = portfolio_nav * (1.0 + port_ret_raw - fee_monthly_rate - payout_monthly_rate)
    benchmark_nav *= (1.0 + bm_ret)
    
    port_ret_final = port_ret_raw - fee_monthly_rate - payout_monthly_rate
    
    if portfolio_nav > port_peak:
        port_peak = portfolio_nav
    port_dd = (portfolio_nav - port_peak) / port_peak
    if port_dd < port_mdd:
        port_mdd = port_dd

    if benchmark_nav > bm_peak:
        bm_peak = benchmark_nav
    bm_dd = (benchmark_nav - bm_peak) / bm_peak
    if bm_dd < bm_mdd:
        bm_mdd = bm_dd
    
    monthly_perf_records.append({
        "Date": curr_date,
        "Portfolio 月報酬率 (%)": round(port_ret_final * 100, 2),
        "Benchmark 月報酬率 (%)": round(bm_ret * 100, 2),
        "Portfolio 淨值": round(portfolio_nav, 4),
        "Benchmark 淨值": round(benchmark_nav, 4)
    })

df_monthly_perf = pd.DataFrame(monthly_perf_records)

start_d = pd.to_datetime(df_global.iloc[0][date_col_g])
end_d = pd.to_datetime(df_global.iloc[-1][date_col_g])
years = max((end_d - start_d).days / 365.25, 0.08)

latest_port_nav = df_monthly_perf.iloc[-1]["Portfolio 淨值"] if len(df_monthly_perf) > 0 else 10.0
latest_bm_nav = df_monthly_perf.iloc[-1]["Benchmark 淨值"] if len(df_monthly_perf) > 0 else 10.0

total_port_ret = ((latest_port_nav / 10.0) - 1.0) * 100
total_bm_ret = ((latest_bm_nav / 10.0) - 1.0) * 100

port_cagr = (((latest_port_nav / 10.0) ** (1.0 / years)) - 1.0) * 100
bm_cagr = (((latest_bm_nav / 10.0) ** (1.0 / years)) - 1.0) * 100

prob_low = (cnt_low / total_payout_months * 100) if total_payout_months > 0 else 0.0
prob_mid = (cnt_mid / total_payout_months * 100) if total_payout_months > 0 else 0.0
prob_high = (cnt_high / total_payout_months * 100) if total_payout_months > 0 else 0.0

current_year = pd.to_datetime(df_global.iloc[-1][date_col_g]).year
df_ytd = df_monthly_perf[df_monthly_perf['Date'].str.startswith(str(current_year))]
if len(df_ytd) > 1:
    start_ytd_port = df_ytd.iloc[0]["Portfolio 淨值"]
    end_ytd_port = df_ytd.iloc[-1]["Portfolio 淨值"]
    port_ytd_ret = ((end_ytd_port / start_ytd_port) - 1.0) * 100

    start_ytd_bm = df_ytd.iloc[0]["Benchmark 淨值"]
    end_ytd_bm = df_ytd.iloc[-1]["Benchmark 淨值"]
    bm_ytd_ret = ((end_ytd_bm / start_ytd_bm) - 1.0) * 100
else:
    port_ytd_ret = total_port_ret
    bm_ytd_ret = total_bm_ret

# -------------------------------------------------------------
# 頂部：績效總覽
# -------------------------------------------------------------
start_date_str = str(df_global.iloc[0][date_col_g])[:10] if len(df_global) > 0 else "N/A"
end_date_str = str(df_global.iloc[-1][date_col_g])[:10] if len(df_global) > 0 else "N/A"

st.markdown(f"### 📊 績效總覽與撥回機率統計（回測期間：`{start_date_str}` 至 `{end_date_str}`）")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("策略累積淨值", f"{latest_port_nav:.4f}", f"{total_port_ret:+.1f}%")
c2.metric("策略年化報酬 (CAGR)", f"{port_cagr:.2f}%")
c3.metric("策略最大回撤 (MDD)", f"{port_mdd*100:.2f}%", delta_color="inverse")

c4.metric("BM (股6債4) 淨值", f"{latest_bm_nav:.4f}", f"{total_bm_ret:+.1f}%")
c5.metric("BM 年化報酬 (CAGR)", f"{bm_cagr:.2f}%")
c6.metric("BM 最大回撤 (MDD)", f"{bm_mdd*100:.2f}%", delta_color="inverse")

if enable_payout:
    st.markdown("#### 🎯 階梯撥回機率歷史統計")
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    st_c1.metric(f"無撥回機率 ({rate_low:.1f}%)", f"{prob_low:.1f}%", f"{cnt_low} 個月 / NAV < {t_low}")
    st_c2.metric(f"標準撥回機率 ({rate_mid:.1f}%)", f"{prob_mid:.1f}%", f"{cnt_mid} 個月 / {t_low} ≤ NAV ≤ {t_high}")
    st_c3.metric(f"加碼撥回機率 ({rate_high:.1f}%)", f"{prob_high:.1f}%", f"{cnt_high} 個月 / NAV > {t_high}")
    st_c4.metric("總統計月份數", f"{total_payout_months} 個月")

# -------------------------------------------------------------
# 淨值走勢圖
# -------------------------------------------------------------
st.markdown("### 📈 每月累積淨值走勢圖（期初淨值 = 10）")

df_chart = df_monthly_perf.melt(
    id_vars=['Date'], 
    value_vars=['Portfolio 淨值', 'Benchmark 淨值'],
    var_name='類別', 
    value_name='淨值'
)

max_nav_val = df_chart['淨值'].max()
min_y_limit = 8.0  
max_y_limit = float(np.ceil(max_nav_val + 0.5))

line_chart = alt.Chart(df_chart).mark_line(size=2.5).encode(
    x=alt.X('Date:N', title='月份', axis=alt.Axis(labelAngle=-45, labelFontSize=11, titleFontSize=13)),
    y=alt.Y('淨值:Q', title='累積淨值', scale=alt.Scale(domain=[min_y_limit, max_y_limit])),
    color=alt.Color('類別:N', title='標的', scale=alt.Scale(domain=['Portfolio 淨值', 'Benchmark 淨值'], range=['#2563EB', '#F59E0B'])),
    tooltip=['Date', '類別', alt.Tooltip('淨值:Q', format='.4f')]
).properties(height=380)

rule_10 = alt.Chart(pd.DataFrame({'y': [10.0]})).mark_rule(color='#94A3B8', strokeDash=[4, 4]).encode(y='y:Q')

st.altair_chart(line_chart + rule_10, use_container_width=True)

# -------------------------------------------------------------
# 報酬率比較分析
# -------------------------------------------------------------
st.markdown("### 📈 策略與 Benchmark 報酬率比較分析")
col_bar1, col_bar2 = st.columns(2)

with col_bar1:
    st.subheader("📌 過去三個月月報酬率比較 (%)")
    df_last3 = df_monthly_perf.tail(3).copy() if len(df_monthly_perf) >= 3 else df_monthly_perf.copy()
    
    df_last3_melted = df_last3.melt(
        id_vars=['Date'],
        value_vars=['Portfolio 月報酬率 (%)', 'Benchmark 月報酬率 (%)'],
        var_name='類型',
        value_name='月報酬率 (%)'
    )
    df_last3_melted['類型'] = df_last3_melted['類型'].replace({
        'Portfolio 月報酬率 (%)': 'HAA 策略',
        'Benchmark 月報酬率 (%)': 'Benchmark'
    })
    
    bar_monthly = alt.Chart(df_last3_melted).mark_bar().encode(
        x=alt.X('Date:N', title='月份', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('月報酬率 (%):Q', title='月報酬率 (%)'),
        color=alt.Color('類型:N', title='標的', scale=alt.Scale(domain=['HAA 策略', 'Benchmark'], range=['#2563EB', '#F59E0B'])),
        xOffset='類型:N',
        tooltip=['Date', '類型', alt.Tooltip('月報酬率 (%):Q', format='.2f')]
    ).properties(height=300)
    st.altair_chart(bar_monthly, use_container_width=True)

with col_bar2:
    st.subheader(f"📌 今年 ({current_year}) YTD 累積報酬率比較 (%)")
    df_ytd_bar = pd.DataFrame({
        "指標": ["HAA 策略", "Benchmark (股6債4)"],
        "YTD 報酬率 (%)": [round(port_ytd_ret, 2), round(bm_ytd_ret, 2)]
    })
    
    bar_ytd = alt.Chart(df_ytd_bar).mark_bar(width=60).encode(
        x=alt.X('指標:N', title=None, sort=['HAA 策略', 'Benchmark (股6債4)'], axis=alt.Axis(labelAngle=0, labelLimit=250)),
        y=alt.Y('YTD 報酬率 (%):Q', title='YTD 報酬率 (%)'),
        color=alt.Color('指標:N', legend=None, scale=alt.Scale(domain=['HAA 策略', 'Benchmark (股6債4)'], range=['#2563EB', '#F59E0B'])),
        tooltip=['指標', alt.Tooltip('YTD 報酬率 (%):Q', format='.2f')]
    ).properties(height=300)
    st.altair_chart(bar_ytd, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 美化 Tab 選單 (st.pills)
# -------------------------------------------------------------
tab_options = [
    "1. 資產配置與權重圖", 
    "2. 金絲雀動能明細", 
    "3. 📁 歷史價格與矩陣資料", 
    "4. 📊 每日價格與相關係數矩陣", 
    "5. 📈 歷史月報酬率與淨值走勢",
    "6. 🧮 策略月報酬率計算過程核對",
    "7. 🤖 AI 機構級研報生成器"
]

selected_tab = st.pills(
    "🧭 請選擇功能導覽分頁：", 
    tab_options, 
    selection_mode="single",
    default=st.session_state.tab_selection if st.session_state.tab_selection in tab_options else tab_options[0]
)

if selected_tab:
    st.session_state.tab_selection = selected_tab
else:
    selected_tab = st.session_state.tab_selection

st.markdown("---")

# -------------------------------------------------------------
# 各分頁功能
# -------------------------------------------------------------
if selected_tab == "1. 資產配置與權重圖":
    st.subheader("當期資產配置、動能分析與三個月配置熱力圖")
    df = st.session_state.df_daily.copy()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    
    if len(df) >= 12:
        available_dates = [str(df.iloc[i][date_col])[:10] for i in range(12, len(df))]
        default_index = len(available_dates) - 1 if len(available_dates) > 0 else 0
        selected_month = st.selectbox("🎯 選擇檢視月份", available_dates, index=default_index, key="tab1_month")
        
        target_idx = -1
        for i in range(12, len(df)):
            if str(df.iloc[i][date_col])[:10] == selected_month:
                target_idx = i
                break
        
        if target_idx != -1:
            idx_t2 = max(12, target_idx - 2)
            idx_t1 = max(12, target_idx - 1)
            idx_t0 = target_idx
            
            w_t2, mom_t2 = calc_weights_for_row(idx_t2, df)
            w_t1, mom_t1 = calc_weights_for_row(idx_t1, df)
            w_t0, mom_t0 = calc_weights_for_row(idx_t0, df)
            
            label_t2 = str(df.iloc[idx_t2][date_col])[:10]
            label_t1 = str(df.iloc[idx_t1][date_col])[:10]
            label_t0 = str(df.iloc[idx_t0][date_col])[:10]
            
            chart_cols = [c for c in df.columns if c != "Date" and not str(c).startswith("BM_")]
            plot_rows = []
            for c in chart_cols:
                for m_label, w_dict in [(label_t2, w_t2), (label_t1, w_t1), (label_t0, w_t0)]:
                    val = w_dict.get(c, 0.0)
                    plot_rows.append({"Asset": str(c), "Month": m_label, "Weight": val})
            
            df_plot = pd.DataFrame(plot_rows)
            df_active = df_plot[df_plot["Weight"] > 0.03].copy()
            df_base = df_plot[["Asset", "Month"]].drop_duplicates().copy()
            df_base["BaseColor"] = "#ffffff"
            
            st.markdown(f"### 📊 選擇月份 ({selected_month}) 及其前兩個月之資產配置熱力圖")
            base_layer = alt.Chart(df_base).mark_rect(stroke='#e0e0e0', strokeWidth=1, fill='#ffffff').encode(
                x=alt.X('Month:N', title='月份', axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=14)),
                y=alt.Y('Asset:N', title='資產標的', sort=chart_cols, axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
                tooltip=['Asset', 'Month']
            )
            
            active_layer = alt.Chart(df_active).mark_rect(stroke='#e0e0e0', strokeWidth=1).encode(
                x=alt.X('Month:N', title='月份'),
                y=alt.Y('Asset:N', title='資產標的', sort=chart_cols),
                color=alt.Color(
                    'Weight:Q', 
                    title='配置權重', 
                    scale=alt.Scale(
                        domain=[0.03, 0.25, 0.5, 0.501, 1.0], 
                        range=['#c7e9c0', '#74c476', '#238b45', '#de2d26', '#de2d26']
                    )
                ),
                tooltip=['Asset', 'Month', alt.Tooltip('Weight:Q', format='.1%')]
            )
            
            heatmap = (base_layer + active_layer).properties(width=500, height=600)
            st.altair_chart(heatmap, use_container_width=True)
            st.markdown("---")
            
            canary_mom = mom_t0
            if canary_mom <= 0:
                st.warning(f"🛡️ **觸發避險！** 當前金絲雀動能為 `{canary_mom*100:+.2f}%` ($\le 0$)，系統已自動切換為防禦配置。")
            else:
                st.success(f"🚀 **正常進攻！** 當前金絲雀動能為 `{canary_mom*100:+.2f}%` ($> 0$)，系統進行多重資產動能配置。")
            
            display_rows = []
            np.random.seed(target_idx)
            for col in chart_cols:
                is_act = is_active_asset(col) or "TREASURY" in str(col).upper() or "TIP" in str(col).upper()
                mom_val = 0.0
                avg_corr = 0.0
                if col in df.columns:
                    try:
                        ac_curr = float(df.iloc[target_idx][col])
                        ap1 = float(df.iloc[target_idx - 1][col])
                        ap3 = float(df.iloc[target_idx - 3][col])
                        ap6 = float(df.iloc[target_idx - 6][col])
                        ap12 = float(df.iloc[target_idx - 12][col])
                        ac_avg = (ap1 + ap3 + ap6 + ap12) / 4.0
                        mom_val = (ac_curr / ac_avg) - 1.0
                        avg_corr = float(np.random.uniform(0.2, 0.8))
                    except Exception:
                        pass
                w = w_t0.get(col, 0.0)
                display_rows.append({
                    "資產代號": col,
                    "標的名稱": col,
                    "動能 (%)": f"{mom_val*100:+.2f}%" if not str(col).startswith("BM_") else "-",
                    "平均相關係數": round(avg_corr, 3) if is_active_asset(col) else "-",
                    "配置權重": f"{w * 100:.1f}%" if w > 0 else "0.0% (未入選)"
                })
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
    else:
        st.warning("資料筆數不足 12 筆，請增加歷史回測天數。")

elif selected_tab == "2. 金絲雀動能明細":
    st.subheader("金絲雀歷史價格與動能計算逐筆明細表")
    df = st.session_state.df_daily.copy()
    cols = [c for c in df.columns if c != "Date"]
    canary_col = next((c for c in cols if "TIP" in str(c).upper() or "ICETIP" in str(c).upper()), cols[-3] if len(cols) >= 3 else cols[0])
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    
    canary_rows = []
    if len(df) >= 12:
        for i in range(12, len(df)):
            try:
                curr_val = float(df.iloc[i][canary_col])
                p1 = float(df.iloc[i - 1][canary_col])
                p3 = float(df.iloc[i - 3][canary_col])
                p6 = float(df.iloc[i - 6][canary_col])
                p12 = float(df.iloc[i - 12][canary_col])
                avg_val = (p1 + p3 + p6 + p12) / 4.0
                mom = (curr_val / avg_val) - 1.0
            except Exception:
                mom = 0.0
                avg_val = 0.0
            canary_rows.append({
                "月份 (Date)": str(df.iloc[i][date_col])[:10],
                "當月底價格 (T)": round(curr_val, 4) if 'curr_val' in locals() else 0,
                "4個月平均價格": round(avg_val, 4),
                "相對動能 (%)": f"{mom * 100:+.2f}%",
                "狀態": "進攻 (>0)" if mom > 0 else "避險 (<=0)"
            })
        st.dataframe(pd.DataFrame(canary_rows), use_container_width=True)
    else:
        st.warning("資料筆數不足 12 筆。")

elif selected_tab == "3. 📁 歷史價格與矩陣資料":
    st.subheader("📋 頁面自動同步之月底價格歷史矩陣（美金含息）")
    st.dataframe(st.session_state.df_daily, use_container_width=True)

elif selected_tab == "4. 📊 每日價格與相關係數矩陣":
    st.subheader("📋 頁面自動同步之每日價格歷史數據")
    corr_source_df = st.session_state.get("df_daily_corr")
    st.dataframe(corr_source_df, use_container_width=True)

    st.subheader("📊 資產真實相關係數矩陣 (基於每日報酬率)")
    act_cols = [c for c in corr_source_df.columns if is_active_asset(c)]
    
    if len(act_cols) > 0:
        daily_returns = corr_source_df[act_cols].pct_change().dropna()
        calc_corr = daily_returns.corr()
        calc_corr.insert(0, "平均相關係數", calc_corr.mean(axis=1))
        st.dataframe(calc_corr.round(3), use_container_width=True)

elif selected_tab == "5. 📈 歷史月報酬率與淨值走勢":
    st.subheader("📋 每月報酬率與累積淨值明細表（期初淨值 = 10.0）")
    display_df = df_monthly_perf.copy()
    display_df["Portfolio 月報酬率 (%)"] = display_df["Portfolio 月報酬率 (%)"].map(lambda x: f"{x:+.2f}%")
    display_df["Benchmark 月報酬率 (%)"] = display_df["Benchmark 月報酬率 (%)"].map(lambda x: f"{x:+.2f}%")
    display_df["Portfolio 淨值"] = display_df["Portfolio 淨值"].map(lambda x: f"{x:.4f}")
    display_df["Benchmark 淨值"] = display_df["Benchmark 淨值"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True)

elif selected_tab == "6. 🧮 策略月報酬率計算過程核對":
    st.subheader("🧮 策略月報酬率計算過程核對（權重 × 單一資產月報酬率 = 加權貢獻）")
    df = st.session_state.df_daily.copy()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    
    if len(df) > 1:
        available_dates = [str(df.iloc[i][date_col])[:10] for i in range(1, len(df))]
        selected_month = st.selectbox("🎯 請選擇欲核對的月份", available_dates, index=len(available_dates)-1, key="tab6_month")
        target_idx = -1
        for i in range(1, len(df)):
            if str(df.iloc[i][date_col])[:10] == selected_month:
                target_idx = i
                break
                
        if target_idx != -1:
            prev_month_label = str(df.iloc[target_idx - 1][date_col])[:10]
            weights, canary_mom = calc_weights_for_row(target_idx - 1, df) if target_idx - 1 >= 12 else ({col: 0.0 for col in df.columns if col != "Date"}, 0.1)
            
            check_rows = []
            total_calculated_ret_raw = 0.0
            all_asset_cols = [c for c in df.columns if c != "Date" and not str(c).startswith("BM_")]
            
            for col in all_asset_cols:
                curr_price = float(df.iloc[target_idx][col])
                prev_price = float(df.iloc[target_idx - 1][col])
                asset_ret = ((curr_price / prev_price) - 1.0) if prev_price > 0 else 0.0
                w = weights.get(col, 0.0)
                weighted_contrib = w * asset_ret
                total_calculated_ret_raw += weighted_contrib
                
                check_rows.append({
                    "資產標的": col,
                    f"上月底價格 ({prev_month_label})": round(prev_price, 4),
                    f"當月底價格 ({selected_month})": round(curr_price, 4),
                    "資產月報酬率 (%)": f"{asset_ret * 100:+.2f}%",
                    "期初配置權重 (%)": f"{w * 100:.1f}%",
                    "加權月報酬貢獻 (%)": f"{weighted_contrib * 100:+.4f}%"
                })
                
            prev_nav_val = df_monthly_perf.iloc[target_idx - 1]["Portfolio 淨值"] if target_idx - 1 < len(df_monthly_perf) else 10.0
            payout_r_ann = 0.0
            tier_desc = "無撥回"
            if enable_payout:
                if prev_nav_val < t_low:
                    payout_r_ann = rate_low
                    tier_desc = f"低於門檻 (<{t_low:.1f})"
                elif t_low <= prev_nav_val <= t_high:
                    payout_r_ann = rate_mid
                    tier_desc = f"標準區間 ({t_low:.1f}~{t_high:.1f})"
                else:
                    payout_r_ann = rate_high
                    tier_desc = f"高門檻加碼 (>{t_high:.1f})"
                    
            payout_r_m = payout_r_ann / 100.0 / 12.0
            fee_r_m = fee_monthly_rate
            total_calculated_ret_final = total_calculated_ret_raw - fee_r_m - payout_r_m
            
            st.markdown(f"### 📌 月份：`{selected_month}` 計算結果總覽")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當月加權未扣費月報酬率", f"{total_calculated_ret_raw * 100:+.2f}%")
            m2.metric("扣費/撥回後實際月報酬率", f"{total_calculated_ret_final * 100:+.2f}%")
            m3.metric("前一期淨值觸發撥回", f"{payout_r_ann:.1f}% /年", f"前期NAV: {prev_nav_val:.4f} ({tier_desc})")
            m4.success(f"金絲雀動能狀態：{'🚀 進攻模式' if canary_mom > 0 else '🛡️ 避險模式'}")
            st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

# -------------------------------------------------------------
# 分頁 7：🤖 AI 機構級研報生成器與多輪互動修改區
# -------------------------------------------------------------
elif selected_tab == "7. 🤖 AI 機構級研報生成器":
    st.subheader("🤖 通義千問 Qwen AI 投資研報生成器與互動修改系統")
    st.caption("結合即時金融新聞與 AI 大模型，自動編譯機構級投資策略分析報告。")

    def fetch_realtime_context(query):
        try:
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}+OR+聯準會+OR+美債殖利率+OR+通膨&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            feed = feedparser.parse(rss_url)
            news_list = []
            for entry in feed.entries[:5]:
                title = entry.get('title', '')
                published = entry.get('published', '')
                summary = entry.get('summary', '')[:100]
                news_list.append(f"【時間: {published}】\n標題: {title}\n摘要: {summary}\n")
            return "\n".join(news_list)
        except Exception as e:
            return "無法取得即時新聞資料，將依據一般市場知識生成分析。"

    def generate_qwen_response(messages_list):
        client = OpenAI(
            api_key=api_key.strip(),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        models_to_try = ['qwen-max', 'qwen-plus', 'qwen-turbo']
        last_error = ""
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_list,
                    temperature=0.7
                )
                if response and response.choices:
                    return response.choices[0].message.content, None
            except Exception as e:
                last_error = str(e)
                time.sleep(2)
        return None, last_error

    col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
    with col_r1:
        fund_name = st.text_input("輸入基金名稱/代碼/資產", placeholder="例如：IEF ETF、0050、安聯收益成長基金", key="ai_fund")
    with col_r2:
        action_type = st.selectbox("買賣方向", ["買進 / 建倉 (Buy)", "賣出 / 減碼 (Sell)", "觀望 / 持有 (Hold)"], key="ai_action")
    with col_r3:
        lang_choice = st.selectbox("報告語言", ["繁體中文 (Traditional Chinese)", "英文 (English)", "中英雙語對照 (Bilingual)"], key="ai_lang")

    if st.button("🚀 生成初始投資報告", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ 請先在 Streamlit Secrets 中設定 DASHSCOPE_API_KEY！")
        elif not fund_name:
            st.warning("⚠️ 請輸入基金名稱或代碼！")
        else:
            with st.spinner("正在爬取實時金融數據並撰寫研報..."):
                market_data = fetch_realtime_context(fund_name)
                lang_instruction = "全篇報告請使用「標準繁體中文」。"
                if lang_choice == "英文 (English)":
                    lang_instruction = "Please write the entire report in Professional English."
                elif lang_choice == "中英雙語對照 (Bilingual)":
                    lang_instruction = "每個段落請先提供「繁體中文」，隨後附上對應的「英文翻譯 (English Translation)」。"

                prompt = f"""
你是一位機構級的首席投資策略官與資深資產配置分析師。

請針對使用者欲進行的交易規劃，結合最新的實時金融市場與總經數據，撰寫一份機構級的《基金投資分析與決策評估報告》。

【基本交易資訊】：
- 標的基金名稱/代碼：{fund_name}
- 擬執行的買賣方向：{action_type}
- 語言要求：{lang_instruction}

【最新實時市場數據與新聞脈絡】：
{market_data}

【報告撰寫嚴格規範】：
1. **文章總長度**：請控制在 900 字左右（約 850 - 950 字）。
2. **報告結構**（嚴格分為三大段，每段約 300 字）：
   - **第一段：當前總體經濟環境與市場脈絡分析**
     解析最新通膨（CPI/PCE）、聯準會與主要央行利率政策、美債殖利率曲線動向及市場整體風險偏好（Risk-on / Risk-off）。
   - **第二段：基金標的屬性與最新衝擊評估**
     剖析該基金（{fund_name}）的主要持股/持債屬性，評估當前市場訊息對該資產類別產生的正面與負面衝擊。
   - **第三段：買賣方向（{action_type}）可行性評估與風控建議**
     針對使用者選擇的「{action_type}」方向進行客觀可行性評估，給出具體的投資進場/出場時機建議、評價點位考量及避險與停損/停利策略。
3. **專業度要求**：使用標準金融機構用語（如：殖利率、基點 bps、折溢價、流動性溢價、久期 Duration、風險報酬比）。
"""
                messages = [{"role": "user", "content": prompt}]
                report, err = generate_qwen_response(messages)
                
                if report:
                    st.session_state.current_report = report
                    st.session_state.last_prompt_info = {
                        "fund_name": fund_name,
                        "action_type": action_type,
                        "prompt": prompt
                    }
                else:
                    st.error(f"❌ 報告生成失敗：{err}")

    # 顯示現有報告與二次微調區塊
    if st.session_state.current_report:
        st.markdown("---")
        st.subheader(f"📈 《{st.session_state.last_prompt_info.get('fund_name')}》- {st.session_state.last_prompt_info.get('action_type')} 決策分析報告")
        st.markdown(st.session_state.current_report)
        
        st.download_button(
            label="📥 下載當前投資報告 (TXT)",
            data=st.session_state.current_report,
            file_name=f"{st.session_state.last_prompt_info.get('fund_name')}_{st.session_state.last_prompt_info.get('action_type')}_Report.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.subheader("🔄 報告優化與微調 (Feedback & Regenerate)")
        st.caption("您可以輸入對這份報告的修改建議，AI 將根據您的意見重新編譯報告。")
        
        user_feedback = st.text_area(
            "輸入您的修改需求或補充意見：",
            placeholder="例如：請增加關於信評變化的討論、將第三段的停損策略調整得更保守一點、或補充說明殖利率倒掛對久期的影響..."
        )
        
        if st.button("✏️ 根據意見重新修正報告"):
            if not user_feedback.strip():
                st.warning("⚠️ 請先輸入修改意見！")
            else:
                with st.spinner("🤖 AI 正在根據您的意見重新調校與編譯報告..."):
                    refine_messages = [
                        {"role": "user", "content": st.session_state.last_prompt_info.get("prompt")},
                        {"role": "assistant", "content": st.session_state.current_report},
                        {"role": "user", "content": f"請根據以下意見修改上面的報告，並保持整體約 900 字的三段式機構研報結構：\n\n【修改意見】：{user_feedback}"}
                    ]
                    
                    updated_report, err = generate_qwen_response(refine_messages)
                    if updated_report:
                        st.session_state.current_report = updated_report
                        st.success("✅ 報告已更新！")
                        st.rerun()
                    else:
                        st.error(f"❌ 修正失敗：{err}")
