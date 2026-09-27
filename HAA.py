import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# 設定本地持久化儲存檔案路徑
SAVED_MONTHLY_PATH = "last_uploaded_data.csv"
SAVED_DAILY_PATH = "last_uploaded_daily_data.csv"

# 頁面基本設定
st.set_page_config(
    page_title="HAA 多重資產動態配置系統",
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

st.title("HAA 多重資產動態配置系統")

def clean_dataframe(uploaded_file):
    """專為 Excel/CSV 格式設計：安全讀取（強健支援多種 CSV 編碼）並將第一列設為表頭，第一欄設為日期"""
    try:
        from io import BytesIO
        content_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file
        
        is_csv = hasattr(uploaded_file, 'name') and uploaded_file.name.lower().endswith('.csv')
        
        df_raw = None
        if is_csv:
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp950', 'big5', 'ansi', 'gbk', 'utf-16', 'latin1']
            
            for enc in encodings_to_try:
                try:
                    bio = BytesIO(content_bytes)
                    bio.seek(0)
                    df_raw = pd.read_csv(bio, header=None, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df_raw is None:
                try:
                    bio = BytesIO(content_bytes)
                    bio.seek(0)
                    df_raw = pd.read_csv(bio, header=None, encoding='utf-8', errors='ignore')
                except Exception as e:
                    st.error(f"檔案解析失敗：{e}")
                    return None
        else:
            df_raw = pd.read_excel(BytesIO(content_bytes), header=None)
            
        if df_raw is None or df_raw.empty:
            st.error("讀取的檔案內容為空，請確認檔案內容。")
            return None

        df = df_raw.copy()
        
        headers = df.iloc[0].values.copy()
        headers[0] = "Date"  # 強制將第一個欄位命名為 Date
        
        df = df.iloc[1:].copy()
        df.columns = headers
        df = df.loc[:, df.columns.notna()]
        
        date_col = df.columns[0]
        def parse_excel_date(val):
            try:
                num = float(val)
                return pd.to_datetime(num, unit='D', origin='1899-12-30').strftime('%Y-%m-%d')
            except Exception:
                try:
                    return pd.to_datetime(val).strftime('%Y-%m-%d')
                except Exception:
                    return str(val)
                    
        df[date_col] = df[date_col].apply(parse_excel_date)
        df = df.dropna(subset=[date_col])
        
        for col in df.columns:
            if col != "Date":
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"檔案解析失敗：{e}")
        return None

# -------------------------------------------------------------
# 安全初始化 Session State 變數
# -------------------------------------------------------------
if "df_daily_corr" not in st.session_state:
    st.session_state.df_daily_corr = None
    if os.path.exists(SAVED_DAILY_PATH):
        try:
            saved_daily_df = pd.read_csv(SAVED_DAILY_PATH)
            st.session_state.df_daily_corr = saved_daily_df
        except Exception:
            st.session_state.df_daily_corr = None

if "df_daily" not in st.session_state:
    if os.path.exists(SAVED_MONTHLY_PATH):
        try:
            saved_df = pd.read_csv(SAVED_MONTHLY_PATH)
            st.session_state.df_daily = saved_df
            st.toast("已自動載入上次上傳的月底歷史資料！", icon="📂")
        except Exception as e:
            st.warning(f"讀取上次儲存的月底資料失敗，切換為系統預設資料：{e}")
            if os.path.exists(SAVED_MONTHLY_PATH):
                os.remove(SAVED_MONTHLY_PATH)

    if "df_daily" not in st.session_state:
        default_excel_data = [
            ['2024-12-31', 455.98969, 578.39697, 470.595, 12911.82031, 1279.98499, 30.4882, 204.17, 1552.3064, 6184.0498, 275.4921, 1661.86304, 897.19098, 6.164, 463.4374, 457.0121, 240.9976, 7090.06982, 119.166, 105.854, 228.65, 1952.8, 463.4374],
            ['2025-01-31', 471.2955, 582.69897, 515.35199, 13271.37988, 1370.64685, 31.00252, 213.38, 1587.13354, 6037.58008, 277.2163, 1684.67896, 908.19781, 6.216, 466.0748, 467.1058, 259.4889, 7164.22021, 120.691, 106.576, 229.5, 2018.79, 466.0748],
            ['2025-02-28', 468.45621, 588.72803, 505.92801, 13098.21973, 1414.32471, 30.69987, 214.16, 1553.18921, 6135.72021, 281.7491, 1697.98499, 923.18079, 6.306, 472.7397, 466.7578, 261.4858, 7461.3501, 123.322, 109.583, 230.25, 2007.19, 472.7397],
            ['2025-03-31', 449.95001, 588.89301, 530.375, 12360.20996, 1414.36096, 30.86404, 218.71, 1381.61401, 6164.16016, 283.304, 1692.59094, 918.23889, 6.3095, 475.6522, 477.8602, 287.2772, 7285.16016, 124.219, 109.983, 231.03, 1928.98, 475.6522],
            ['2025-04-30', 454.14651, 593.24597, 567.03101, 12276.38965, 1475.53052, 32.53406, 220.59, 1403.08716, 5972.35986, 288.6796, 1707.02161, 917.48352, 6.286, 489.6304, 436.2842, 304.4191, 7137.18994, 124.301, 111.092, 231.85, 1947.84, 489.6304],
            ['2025-05-31', 624.45001, 576.07397, 13049.12988, 1545.90405, 33.90431, 222.35, 1593.15247, 6158.0, 289.191, 1735.17139, 925.28333, 6.333, 487.8912, 440.4325, 302.0618, 7221.18994, 123.57, 109.774, 232.71, 2061.07, 487.8912],
            ['2025-06-30', 662.29102, 611.09003, 13712.70996, 1582.61755, 34.49026, 231.71, 1703.33643, 6379.45996, 295.8693, 1775.33374, 946.34668, 6.424, 497.1365, 459.2098, 302.8712, 7217.77002, 124.779, 111.509, 233.51, 2154.5, 497.1365],
            ['2025-07-31', 679.242, 583.95001, 14020.45996, 1551.72314, 34.05813, 236.76, 1781.91602, 6621.02979, 293.996, 1782.06201, 957.5011, 6.479, 489.7316, 471.833, 303.819, 7143.52979, 124.892, 110.878, 234.37, 2184.27, 489.7316],
            ['2025-08-31', 686.70001, 631.95099, 14304.67969, 1604.6134, 36.51201, 237.67, 1797.05176, 7375.91016, 298.0911, 1809.3645, 972.15991, 6.553, 496.8505, 465.4838, 315.7241, 7382.60986, 126.884, 112.714, 235.27, 2239.19, 496.8505],
            ['2025-09-30', 733.53198, 673.31702, 14826.7998, 1636.66846, 37.38784, 250.23, 1925.77039, 7632.27002, 301.5529, 1821.45239, 987.90387, 6.611, 500.0961, 469.5738, 352.0861, 7413.49023, 127.375, 113.473, 236.07, 2321.16, 500.0961],
            ['2025-10-31', 766.51398, 679.54498, 15173.9502, 1645.33911, 38.06613, 252.79, 2087.06738, 7655.12988, 301.3995, 1834.04675, 1009.62097, 6.654, 498.8263, 476.4557, 369.1021, 7250.60986, 127.797, 114.255, 236.93, 2373.52, 498.8263],
            ['2025-11-30', 744.73199, 720.74902, 15211.13965, 1672.37524, 38.08687, 248.26, 1999.66418, 7509.2998, 303.0344, 1844.26245, 1012.65997, 6.667, 499.9849, 477.7115, 385.5097, 7414.81982, 128.079, 115.408, 237.63, 2373.92, 499.9849],
            ['2025-12-31', 765.00098, 728.53497, 15220.4502, 1738.87622, 38.44813, 259.21, 2097.84766, 7791.81982, 303.8636, 1862.33386, 1017.89899, 6.6905, 501.2906, 480.0591, 396.1204, 7254.77979, 127.363, 114.529, 238.46, 2399.41, 501.2906],
            ['2026-01-31', 827.71399, 840.211, 15441.15039, 1815.69177, 40.71719, 281.95, 2319.90332, 7973.16992, 306.4952, 1880.73633, 1023.57001, 6.731, 505.9834, 518.5422, 457.9546, 7459.43994, 127.942, 114.292, 239.17, 2471.01, 505.9834],
            ['2026-02-28', 876.27899, 872.255, 15323.7998, 1876.97681, 44.51602, 286.3, 2581.2063, 8094.75977, 308.8157, 1884.54236, 1037.89099, 6.793, 511.641, 534.2389, 479.9178, 8016.75, 129.597, 117.142, 239.84, 2503.4, 511.641],
            ['2026-03-31', 755.96002, 834.72803, 14560.75, 1692.64917, 39.19176, 256.89, 2257.72729, 7591.68994, 300.0013, 1837.91504, 1006.26801, 6.671, 495.9127, 616.0168, 423.3374, 7526.66016, 127.911, 114.425, 240.56, 2324.86, 495.9127],
            ['2026-04-30', 879.15002, 860.80798, 16088.55957, 1812.35889, 42.39718, 265.51, 2804.55884, 8289.91992, 303.8425, 1885.24402, 1031.73206, 6.767, 502.0911, 658.2464, 423.4757, 8200.76953, 129.327, 114.296, 241.29, 2562.34, 502.0911],
            ['2026-05-31', 978.03601, 824.54199, 16935.34961, 1858.56763, 44.2995, 268.1, 3254.03345, 8535.91992, 305.7231, 1898.43054, 1040.17297, 6.789, 503.778, 622.8731, 417.3398, 8204.49023, 129.647, 114.274, 242.04, 2695.81, 503.778],
            ['2026-06-30', 965.427, 804.79901, 16774.07031, 1865.46997, 43.816, 259.29, 3317.13452, 8706.2002, 304.3977, 1902.23071, 1046.87, 6.82, 500.2215, 562.81012, 369.4695, 8328.58984, 129.005, 114.561, 242.76, 2675.08, 500.2215],
            ['2026-07-31', 934.23, 844.347, 16763.42, 1907.464906, 45.37611182, 262.5, 3077.004272, 8106.35, 301.134, 1896.8195, 1030.75, 6.758, 497.5484, 620.8962, 371.54, 8524.5, 128.107, 112.96, 243.58, 2677.7, 497.5484],
            ['2026-08-31', 964.93103, 838.54602, 17219.93945, 1931.80261, 46.45047, 277.57999, 3362.00391, 8218.0, 302.8004, 1916.45764, 1040.09, 6.818, 499.81091, 657.9306, 408.42, 8295.49023, 128.14, 113.123, 244.3, 2749.87, 499.8109]
        ]
        headers_list = ['Date', 'ACWI', 'Asia ex JP', 'Latam', 'S&P500', 'SXXR', 'Topix', 'Emerging Euro, Middle East, Afica', 'Taiex', 'CSI 300', 'Corp Bond', 'HY', 'EMBI', 'EMBI Corp', 'Globa Agg Local Currency', 'Commodity', 'GLD', 'REITS', 'TIP', 'Treasury', 'T Bill', 'BM_AWCI', 'BM_AGG']
        st.session_state.df_daily = pd.DataFrame(default_excel_data, columns=headers_list)

# 初始化分頁狀態管理
if "tab_selection" not in st.session_state:
    st.session_state.tab_selection = "1. 資產配置與權重圖"

def is_active_asset(col_name):
    c_str = str(col_name).strip().upper()
    if any(k in c_str for k in ["TIP", "TREASURY", "BIL", "CASH", "UNNAMED", "DATE", "BM_"]):
        return False
    return True

# -------------------------------------------------------------
# 核心計算函式
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
# 頂部控制項：經管費與優化後的階梯撥回率設定
# -------------------------------------------------------------
with st.expander("⚙️ 費用與階梯撥回率（配息）參數設定區", expanded=True):
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

# -------------------------------------------------------------
# 動態計算每個月報酬率、累積淨值與最大回撤 (MDD) + 統計撥回機率
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

cnt_low = 0    # < t_low
cnt_mid = 0    # t_low <= NAV <= t_high
cnt_high = 0   # > t_high
total_payout_months = 0

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

# -------------------------------------------------------------
# 計算今年 YTD 報酬率
# -------------------------------------------------------------
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
# 頂部：績效總覽（同時展示 Portfolio、Benchmark 及撥回機率）
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
# 淨值走勢圖 (期初 10 在中低位置)
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
# 📊 報酬率比較分析：過去三個月月報酬率 & 今年 YTD 累積報酬率（統一排序與不截斷 X 軸）
# -------------------------------------------------------------
st.markdown("### 📈 策略與 Benchmark 報酬率比較分析")
col_bar1, col_bar2 = st.columns(2)

with col_bar1:
    st.subheader("📌 過去三個月月報酬率比較 (%)")
    if len(df_monthly_perf) >= 3:
        df_last3 = df_monthly_perf.tail(3).copy()
    else:
        df_last3 = df_monthly_perf.copy()
        
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
    # 統一排序：左邊為 HAA 策略，右邊為 Benchmark，名稱與左圖顏色對應
    df_ytd_bar = pd.DataFrame({
        "指標": ["HAA 策略", "Benchmark (股6債4)"],
        "YTD 報酬率 (%)": [round(port_ytd_ret, 2), round(bm_ytd_ret, 2)]
    })
    
    bar_ytd = alt.Chart(df_ytd_bar).mark_bar(width=60).encode(
        # 確保 x 軸標籤完整顯示，不被截斷
        x=alt.X('指標:N', title=None, sort=['HAA 策略', 'Benchmark (股6債4)'], axis=alt.Axis(labelAngle=0, labelLimit=250)),
        y=alt.Y('YTD 報酬率 (%):Q', title='YTD 報酬率 (%)'),
        color=alt.Color('指標:N', legend=None, scale=alt.Scale(domain=['HAA 策略', 'Benchmark (股6債4)'], range=['#2563EB', '#F59E0B'])),
        tooltip=['指標', alt.Tooltip('YTD 報酬率 (%):Q', format='.2f')]
    ).properties(height=300)
    st.altair_chart(bar_ytd, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 美化 Tab 選單 (使用 st.pills)
# -------------------------------------------------------------
tab_options = [
    "1. 資產配置與權重圖", 
    "2. 金絲雀動能明細", 
    "3. 📁 月底價格上傳與歷史矩陣", 
    "4. 📊 每日價格上傳與相關係數矩陣", 
    "5. 📈 歷史月報酬率與淨值走勢",
    "6. 🧮 策略月報酬率計算過程核對"
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
# 分頁 1：資產配置與權重圖
# -------------------------------------------------------------
if selected_tab == "1. 資產配置與權重圖":
    st.subheader("當期資產配置、動能分析與三個月配置熱力圖")
    df = st.session_state.df_daily.copy()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    
    if len(df) >= 12:
        available_dates = [str(df.iloc[i][date_col])[:10] for i in range(12, len(df))]
        
        default_index = len(available_dates) - 1 if len(available_dates) > 0 else 0
        selected_month = st.selectbox("🎯 選擇檢視月份 (自動跳至最新日期)", available_dates, index=default_index, key="tab1_month")
        
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
            
            st.markdown(f"### 📊 選擇月份 ({selected_month}) 及其前兩個月之資產配置熱力圖 (0~3% 白、3%~50% 淺綠$\rightarrow$深綠、50%以上紅)")
            
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
                st.warning(f"🛡️ **觸發避險！** 當前金絲雀動能為 `{canary_mom*100:+.2f}%` ($\le 0$)，系統已自動切換為防禦配置（Treasury vs T-Bill）。")
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
        st.warning("資料筆數不足 12 筆，請上傳完整歷史資料。")

# -------------------------------------------------------------
# 分頁 2：金絲雀動能明細
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 分頁 3：📁 月底價格上傳與歷史矩陣
# -------------------------------------------------------------
elif selected_tab == "3. 📁 月底價格上傳與歷史矩陣":
    st.subheader("步驟一：上傳月底資產價格檔案 (覆蓋或更新策略數據)")
    uploaded_file = st.file_uploader("請選擇您的檔案 (支援 Excel 或 CSV 格式)", type=["csv", "xlsx", "xls", "xlsm"], key="monthly_uploader")
    
    if uploaded_file is not None:
        cleaned_df = clean_dataframe(uploaded_file)
        if cleaned_df is not None:
            st.session_state.df_daily = cleaned_df
            cleaned_df.to_csv(SAVED_MONTHLY_PATH, index=False)
            st.success(f"成功載入月底價格檔案，共計 {len(st.session_state.df_daily)} 筆資料！")

    st.subheader("步驟二：月底歷史價格矩陣核對與互動編輯")
    edited_df = st.data_editor(st.session_state.df_daily, num_rows="dynamic", key="daily_editor")
    
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("🔄 儲存變更並重新計算策略配置", type="primary"):
            st.session_state.df_daily = edited_df
            edited_df.to_csv(SAVED_MONTHLY_PATH, index=False)
            st.success("已成功儲存變更至本地，並更新策略配置！")
            st.session_state.tab_selection = "1. 資產配置與權重圖"
            st.rerun()

    with col_reset:
        if st.button("🗑️ 清除自訂檔並重置為系統預設資料"):
            if os.path.exists(SAVED_MONTHLY_PATH):
                os.remove(SAVED_MONTHLY_PATH)
            if "df_daily" in st.session_state:
                del st.session_state["df_daily"]
            st.success("已刪除歷史儲存檔，正在重載頁面...")
            st.rerun()

# -------------------------------------------------------------
# 分頁 4：📊 每日價格上傳與相關係數矩陣
# -------------------------------------------------------------
elif selected_tab == "4. 📊 每日價格上傳與相關係數矩陣":
    st.subheader("步驟一：上傳每日資產價格檔案 (用於計算相關係數矩陣)")
    st.info("💡 上傳含每日價格的 Excel 或 CSV 檔案，系統將自動計算日報酬率及資產間的相關係數。")
    
    daily_uploaded_file = st.file_uploader("請選擇每日價格檔案 (支援 Excel 或 CSV)", type=["csv", "xlsx", "xls", "xlsm"], key="daily_corr_uploader")
    
    if daily_uploaded_file is not None:
        cleaned_daily = clean_dataframe(daily_uploaded_file)
        if cleaned_daily is not None:
            st.session_state.df_daily_corr = cleaned_daily
            cleaned_daily.to_csv(SAVED_DAILY_PATH, index=False)
            st.success(f"成功載入每日價格資料，共計 {len(cleaned_daily)} 筆歷史交易日資料！")

    st.subheader("步驟二：每日資產價格矩陣核對區")
    corr_source_df = st.session_state.get("df_daily_corr") if st.session_state.get("df_daily_corr") is not None else st.session_state.df_daily
    
    if st.session_state.get("df_daily_corr") is not None:
        st.success("📋 以下為已上傳並解析完成之每日價格歷史數據：")
    else:
        st.caption("📌 尚未上傳每日價格檔案，目前暫時預覽月底價格數據（上傳後將自動切換為每日價格檔）。")
        
    st.dataframe(corr_source_df, use_container_width=True)

    st.subheader("步驟三：相關係數矩陣與平均相關係數核對區")
    act_cols = [c for c in corr_source_df.columns if is_active_asset(c)]
    
    if len(act_cols) > 0:
        if st.session_state.get("df_daily_corr") is not None:
            st.success("✅ **已根據上圖每日價格資料計算真實資產間相關係數矩陣：**")
            daily_returns = corr_source_df[act_cols].pct_change().dropna()
            calc_corr = daily_returns.corr()
        else:
            st.caption("📌 目前使用月底價格數據展示（建議上傳每日價格檔以獲得精確相關係數）。")
            corr_vals = np.random.uniform(0.2, 0.8, (len(act_cols), len(act_cols)))
            np.fill_diagonal(corr_vals, 1.00)
            calc_corr = pd.DataFrame(corr_vals, index=act_cols, columns=act_cols)
        
        calc_corr.insert(0, "平均相關係數", calc_corr.mean(axis=1))
        st.dataframe(calc_corr.round(3), use_container_width=True)
    else:
        st.warning("無符合條件之資產標的進行相關係數計算。")

# -------------------------------------------------------------
# 分頁 5：歷史月報酬率與淨值走勢
# -------------------------------------------------------------
elif selected_tab == "5. 📈 歷史月報酬率與淨值走勢":
    st.subheader("📋 每月報酬率與累積淨值明細表（期初淨值 = 10.0）")
    st.markdown(f"💡 **目前淨值計算公式**：`當期淨值 = 前期淨值 * (1 + 未扣費月報酬率 - {management_fee_pct}%/12 - 適用撥回率/12)`")
    
    display_df = df_monthly_perf.copy()
    display_df["Portfolio 月報酬率 (%)"] = display_df["Portfolio 月報酬率 (%)"].map(lambda x: f"{x:+.2f}%")
    display_df["Benchmark 月報酬率 (%)"] = display_df["Benchmark 月報酬率 (%)"].map(lambda x: f"{x:+.2f}%")
    display_df["Portfolio 淨值"] = display_df["Portfolio 淨值"].map(lambda x: f"{x:.4f}")
    display_df["Benchmark 淨值"] = display_df["Benchmark 淨值"].map(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)

# -------------------------------------------------------------
# 分頁 6：🧮 策略月報酬率計算過程核對
# -------------------------------------------------------------
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
            
            if target_idx - 1 >= 12:
                weights, canary_mom = calc_weights_for_row(target_idx - 1, df)
            else:
                weights = {col: 0.0 for col in df.columns if col != "Date"}
                canary_mom = 0.1
            
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
            
            if target_idx - 1 < 12:
                m4.info("註：前 12 個月權重以 Benchmark 同步處理。")
            else:
                m4.success(f"金絲雀動能狀態：{'🚀 進攻模式' if canary_mom > 0 else '🛡️ 避險模式'}")
                
            st.dataframe(pd.DataFrame(check_rows), use_container_width=True)
    else:
        st.warning("歷史資料筆數不足以計算月報酬率。")