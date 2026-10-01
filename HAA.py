import os
import glob
import time
import re
import json
import datetime
from datetime import timezone, timedelta
import urllib.parse
import urllib.request
import io
import feedparser
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------
# 1. 頁面配置與 Secrets 安全讀取
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 機構級金融研報與總經決策系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 讀取 DashScope 與 FRED 的 API Key
dashscope_key = st.secrets.get("DASHSCOPE_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
fred_api_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))

st.title("📈 AI 機構級金融市場研報與總經決策系統")
st.caption("自動彙整實時總經新聞、Yahoo / FRED 雙資料源動態連動圖表、每日研報/月報，以及資產交易決策評估。")

# ---------------------------------------------------------
# 2. 側邊欄選單
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 功能選單")
    
    if dashscope_key:
        st.success("🔒 DashScope API Key 已載入")
    else:
        st.error("❌ 未讀取到 DASHSCOPE_API_KEY")

    if fred_api_key:
        st.success("🔒 FRED API Key 已載入")
    else:
        st.warning("⚠️ 未讀取到 FRED_API_KEY（總經數據將改用備用管道）")

    st.markdown("---")
    app_mode = st.radio(
        "請選擇功能模組：",
        ["📰 每日要聞與總經月報", "📊 全球總體經濟數據 (Yahoo & FRED)", "🎯 基金 / ETF 交易決策評估"]
    )

# ---------------------------------------------------------
# 3. 工具函數 (Qwen API & FRED API / Yahoo 雙資料源抓取)
# ---------------------------------------------------------
def call_qwen_api(messages_list):
    if not dashscope_key:
        return None, "請先於 Streamlit Secrets 設定 DASHSCOPE_API_KEY！"
        
    client = OpenAI(
        api_key=dashscope_key.strip(),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    models_to_try = ['qwen-max', 'qwen-plus', 'qwen-turbo']
    last_error = ""
    
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages_list,
                temperature=0.3
            )
            if response and response.choices:
                return response.choices[0].message.content, None
        except Exception as e:
            last_error = str(e)
            time.sleep(1.5)
            
    return None, f"模型呼叫失敗，錯誤原因: {last_error}"

def clean_html(raw_html):
    """清除 RSS 中的 HTML 標籤"""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

# 常用總經與市場標的代碼表
DEFAULT_INDICATORS = {
    "美國 10 年期公債殖利率 (%)": "^TNX",
    "S&P 500 指數": "^GSPC",
    "美國 核心 PCE 物價指數": "PCEPILFE",
    "美國 實質 GDP (Real GDP)": "GDPC1",
    "美國 CPI 消費者物價指數": "CPIAUCSL",
    "黃金期貨 (Gold)": "GC=F",
    "原油期貨 (WTI Crude)": "CL=F",
    "美元指數 (DXY)": "DX-Y.NYB",
    "VIX 恐慌指數": "^VIX",
    "台灣加權指數": "^TWII"
}

def search_symbol_by_llm(keyword):
    prompt = f"""
你是一個精通全球金融市場（Yahoo Finance 與 FRED 數據庫）的總經專家。
使用者輸入的自然語言關鍵字為："{keyword}"

請提供 3 個最精準的資料代碼。
注意：
1. 若屬總經指標（如 GDP, CPI, PCE, 失業率），請務必提供 FRED 正確 Series ID（例如 GDP 提供 GDPC1、CPI 提供 CPIAUCSL）。
2. 若屬市場指數或資產，提供 Yahoo Ticker（如 ^TNX, ^GSPC）。

請嚴格以 JSON 陣列格式輸出，不要加任何多餘說明：
[
  {{"code": "GDPC1", "name": "US Real GDP", "source": "FRED"}},
  {{"code": "^TNX", "name": "US 10-Year Treasury Yield", "source": "Yahoo"}},
  {{"code": "PCEPILFE", "name": "US Core PCE Index", "source": "FRED"}}
]
"""
    res, err = call_qwen_api([{"role": "user", "content": prompt}])
    if res:
        try:
            match = re.search(r'\[.*\]', res, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            return []
    return []

@st.cache_data(ttl=3600)
def fetch_smart_data(symbol):
    """結合 FRED 官方 REST API 與 Yahoo Finance 的智慧抓取函數（強制使用 period="max" 抓取完整歷史）"""
    raw_code = str(symbol).strip().replace("$", "").replace('"', "").replace("'", "")
    
    fred_mapping = {
        "GDP": "GDPC1",
        "CPI": "CPIAUCSL",
        "PCE": "PCEPILFE",
        "UNRATE": "UNRATE"
    }
    clean_code = fred_mapping.get(raw_code.upper(), raw_code)

    # 1. 優先嘗試透過 FRED 官方 API 抓取
    if fred_api_key and not clean_code.startswith("^") and len(clean_code) <= 10:
        fred_url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={clean_code}&api_key={fred_api_key}&file_type=json"
        )
        try:
            req = urllib.request.Request(
                fred_url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            observations = data.get("observations", [])
            if observations:
                dates = []
                values = []
                for obs in observations:
                    if obs["value"] != ".":
                        dates.append(obs["date"])
                        values.append(float(obs["value"]))
                
                df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    symbol: values
                })
                df = df.dropna().sort_values('Date')
                if len(df) > 2:
                    return df
        except Exception:
            pass

    # 2. 備用：嘗試 FRED 官方公開 CSV 捷徑
    fred_csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={clean_code.replace('^', '')}"
    try:
        req = urllib.request.Request(
            fred_csv_url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            csv_data = response.read()
            
        df = pd.read_csv(io.BytesIO(csv_data))
        if not df.empty and 'DATE' in df.columns:
            df['Date'] = pd.to_datetime(df['DATE'], errors='coerce')
            val_col = [c for c in df.columns if c != 'DATE'][0]
            df[symbol] = pd.to_numeric(df[val_col], errors='coerce')
            df = df[['Date', symbol]].dropna().sort_values('Date')
            if len(df) > 2:
                return df
    except Exception:
        pass

    # 3. 嘗試 Yahoo Finance (yfinance) - 使用 period="max" 確保獲取完整歷史
    try:
        ticker = yf.Ticker(clean_code)
        df = ticker.history(period="max", auto_adjust=True)
        if not df.empty and len(df) > 2:
            df = df.reset_index()
            val_col = 'Close' if 'Close' in df.columns else df.columns[1]
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df[symbol] = pd.to_numeric(df[val_col], errors='coerce')
            df = df[['Date', symbol]].dropna().sort_values('Date')
            return df
    except Exception:
        pass

    return pd.DataFrame()

# ---------------------------------------------------------
# 模組一：每日金融市場要聞 & 歷史查詢 & 月報彙整
# ---------------------------------------------------------
if app_mode == "📰 每日要聞與總經月報":
    st.header("📰 全球金融市場要聞與總經月報系統")
    
    tab1, tab2, tab3 = st.tabs(["🚀 即時生成今日要聞", "📅 瀏覽歷史每日研報", "🗓️ 自動彙整總經月報"])
    
    tz_taiwan = timezone(timedelta(hours=8))
    today_dt = datetime.datetime.now(tz_taiwan)
    today_str = today_dt.strftime("%Y-%m-%d")

    with tab1:
        st.subheader("📡 即時抓取路透社與 Yahoo 財經新聞摘要並編譯研報")
        st.info(f"📅 基準日期（台灣時間）：{today_dt.strftime('%Y 年 %m 月 %d 日')}")
        
        if st.button("🚀 即時編譯今日研報", type="primary"):
            with st.spinner("正在專注擷取路透社 (Reuters) 與 Yahoo 財經最新新聞與市場數據..."):
                rss_urls = [
                    "https://news.google.com/rss/search?q=site:cn.reuters.com+OR+site:reuters.com&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
                    "https://news.google.com/rss/search?q=site:tw.stock.yahoo.com+OR+site:finance.yahoo.com+通膨+OR+聯準會+OR+美股+OR+美債+OR+殖利率&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
                    "https://news.google.com/rss/search?q=site:reuters.com+OR+site:finance.yahoo.com+Fed+OR+Yield+OR+CPI&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                ]
                
                raw_news = []
                for url in rss_urls:
                    try:
                        feed = feedparser.parse(url)
                        for entry in feed.entries[:8]:
                            title = clean_html(entry.get('title', ''))
                            published = entry.get('published', '')
                            summary = clean_html(entry.get('summary', ''))[:250]
                            raw_news.append(f"【時間: {published}】\n標題: {title}\n摘要: {summary}\n")
                    except Exception as e:
                        st.warning(f"⚠️ RSS 讀取異常: {e}")

            if not raw_news:
                st.error("❌ 無法讀取新聞資料，請稍後再試。")
            else:
                prompt = f"""
你是一位機構級高級總體經濟分析師與資深財經主編。
今天是 {today_dt.strftime('%Y 年 %m 月 %d 日')}。

以下是今日從「路透社 (Reuters)」與「Yahoo 財經」擷取的最新即時新聞資料：
=== 今日新聞原始資料 ===
{"".join(raw_news)}
========================

【任務要求 - 嚴格遵照機構研報高規格排版格式】：
請根據上述原始新聞資料，為機構投資人撰寫《每日金融市場要聞與機構深度研報》。

輸出格式必須嚴格對齊以下結構與樣式：
1. **頂部標題宣告**：
   日期：{today_dt.strftime('%Y年%m月%d日')}（路透中文網 & Yahoo 奇摩財經 深度總結）
   
   # 每日金融市場要聞與機構深度研報（{today_dt.strftime('%Y年%m月%d日')}）

2. **引言段落**：
   簡明扼要點出今日全球金融市場的核心轉折、關鍵利率表現、央行政策預期、股市風險與地緣政治影響。

3. **四大核心板塊（必須嚴格使用以下標題與條列格式）**：
   - **一、全球金融市場焦點與數據總覽**
   - **二、總體經濟、央行政策與債券市場**
   - **三、科技產業與企業財務動態**
   - **四、外匯、大宗商品與信用市場**

4. **條列細節規範**：
   - 每個板塊底下包含 3 個條列項目（使用 `*`）。
   - 每個條列開頭必須採用 **粗體前綴名稱加冒號**。
   - 內文中所有關鍵數字、百分比、企業名稱、重要指標均需**粗體標示**。
"""
                with st.spinner("🤖 Qwen 首席分析師正在進行深度研報撰寫與脈絡梳理..."):
                    report_content, err = call_qwen_api([{"role": "user", "content": prompt}])
                    if report_content:
                        st.session_state.today_report = report_content
                        os.makedirs("daily_reports", exist_ok=True)
                        with open(f"daily_reports/{today_str}.md", "w", encoding="utf-8") as f:
                            f.write(report_content)
                        st.success("✅ 深度研報生成完畢並已同步存檔！")
                    else:
                        st.error(f"❌ 生成失敗: {err}")

        if "today_report" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state.today_report)

    with tab2:
        st.subheader("📅 歷史每日研報查詢")
        report_files = sorted(glob.glob("daily_reports/*.md"), reverse=True)
        if not report_files:
            st.info("💡 目前資料夾中尚無歷史研報檔。")
        else:
            selected_file = st.selectbox("選擇欲檢視的日期：", report_files, format_func=lambda x: os.path.basename(x).replace(".md", ""))
            if selected_file:
                with open(selected_file, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown("---")
                st.markdown(content)
                st.download_button("📥 下載此研報 (.md)", content, file_name=os.path.basename(selected_file))

    with tab3:
        st.subheader("🗓️ 月度總經趨勢彙整系統")
        all_files = glob.glob("daily_reports/*.md")
        available_months = sorted(list(set([os.path.basename(f)[:7] for f in all_files])), reverse=True)
        if not available_months:
            st.warning("⚠️ 尚無每日研報數據。")
        else:
            target_month = st.selectbox("選擇欲彙整的月份：", available_months)
            if st.button("🚀 生成機構級月報", type="primary"):
                month_files = sorted(glob.glob(f"daily_reports/{target_month}-*.md"))
                monthly_combined_text = ""
                for filepath in month_files:
                    date_str = os.path.basename(filepath).replace(".md", "")
                    with open(filepath, "r", encoding="utf-8") as f:
                        monthly_combined_text += f"\n\n=== {date_str} 數據與論述 ===\n" + f.read()[:800]
                
                monthly_prompt = f"""
你是一位機構首席經濟學家。請針對 {target_month} 月份每日金融市場紀錄進行融會貫通，撰寫一份高規格、論述詳盡且可讀性強的《{target_month} 全球金融市場總經趨勢月報》。

=== 全月資料紀錄 ===
{monthly_combined_text}
===================

【任務要求】：
請進行全月核心主軸歸納與深度趨勢分析，段落前標題與重點數據請加粗標示，絕不可出現 "XX" 等佔位符符號。
"""
                with st.spinner("🤖 Qwen 正進行數據彙整與深度月報撰寫..."):
                    monthly_report, err = call_qwen_api([{"role": "user", "content": monthly_prompt}])
                    if monthly_report:
                        st.session_state[f"monthly_{target_month}"] = monthly_report
                        st.success("✅ 月報彙整成功！")
                    else:
                        st.error(f"❌ 生成失敗: {err}")

            if f"monthly_{target_month}" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state[f"monthly_{target_month}"])
                st.download_button(
                    "📥 下載總經月報 (.md)",
                    st.session_state[f"monthly_{target_month}"],
                    file_name=f"Monthly_Report_{target_month}.md"
                )

# ---------------------------------------------------------
# 模組二：全球總體經濟數據 (Yahoo & FRED)
# ---------------------------------------------------------
elif app_mode == "📊 全球總體經濟數據 (Yahoo & FRED)":
    st.header("📊 全球總體經濟與市場數據庫 (Yahoo & FRED 智慧雙資料源)")
    st.caption("自動連線 Yahoo Finance 與 FRED API 資料庫，支援獨立資料轉換、異質頻率完美對齊與動態時間區間篩選！")

    if "custom_indicators" not in st.session_state:
        st.session_state.custom_indicators = DEFAULT_INDICATORS.copy()

    if "selected_indicators_list" not in st.session_state:
        st.session_state.selected_indicators_list = ["美國 10 年期公債殖利率 (%)", "S&P 500 指數"]

    if "indicator_transforms" not in st.session_state:
        st.session_state.indicator_transforms = {}

    with st.expander("🔍 智慧搜尋數據代碼 / 新增自訂指標", expanded=True):
        col_sch1, col_sch2 = st.columns([3, 1])
        with col_sch1:
            search_query = st.text_input("輸入想尋找的數據名稱（中英文皆可）：", placeholder="例如：US GDP, 美國CPI, 台積電, S&P500, 比特幣, 黃金")
        with col_sch2:
            st.write(" ")
            st.write(" ")
            do_search = st.button("🔎 搜尋數據代碼", type="primary")

        if do_search and search_query:
            with st.spinner(f"正在搜尋與 '{search_query}' 最匹配的代碼..."):
                search_results = search_symbol_by_llm(search_query)
                if search_results:
                    st.session_state.search_results = search_results

        if "search_results" in st.session_state and st.session_state.search_results:
            st.markdown("##### 🎯 匹配到的數據代碼建議：")
            res_options = {
                f"{item['name']} (Code: {item['code']} - Source: {item.get('source', 'Auto')})": item 
                for item in st.session_state.search_results
            }
            selected_match_label = st.selectbox("選擇欲加入的數據指標：", list(res_options.keys()))
            selected_item = res_options[selected_match_label]

            col_add1, col_add2 = st.columns([3, 1])
            with col_add1:
                final_name = st.text_input("圖表顯示名稱：", value=selected_item['name'])
            with col_add2:
                st.write(" ")
                st.write(" ")
                if st.button("➕ 加入指標對比"):
                    st.session_state.custom_indicators[final_name] = selected_item['code']
                    if final_name not in st.session_state.selected_indicators_list:
                        st.session_state.selected_indicators_list.append(final_name)
                    st.success(f"✅ 成功將【{final_name}】加入圖表對比！")
                    st.rerun()

    st.markdown("---")
    
    valid_options = list(st.session_state.custom_indicators.keys())
    st.session_state.selected_indicators_list = [k for k in st.session_state.selected_indicators_list if k in valid_options]

    # 第一步：選擇欲比較的指標
    selected_indicators = st.multiselect(
        "選擇欲比較的市場/總經指標：",
        valid_options,
        default=st.session_state.selected_indicators_list
    )
    st.session_state.selected_indicators_list = selected_indicators

    # 第二步：針對每一個已選指標，獨立設定其數據處理方式
    transform_options = ["原始水準 (Level/Raw)", "年增率 (YoY %)", "月/日增額 (Diff)"]
    
    if selected_indicators:
        st.markdown("##### ⚙️ 針對個別指標設定資料處理方式：")
        transform_cols = st.columns(min(len(selected_indicators), 3))
        
        for idx, ind in enumerate(selected_indicators):
            col_target = transform_cols[idx % len(transform_cols)]
            with col_target:
                current_val = st.session_state.indicator_transforms.get(ind, "原始水準 (Level/Raw)")
                chosen_transform = st.selectbox(
                    f"【{ind}】處理方式",
                    transform_options,
                    index=transform_options.index(current_val) if current_val in transform_options else 0,
                    key=f"trans_{ind}"
                )
                st.session_state.indicator_transforms[ind] = chosen_transform

    st.markdown("---")
    
    # 第三步：時間區間與圖表顯示設定
    col_opt1, col_opt2, col_opt3 = st.columns([2, 1, 1])
    with col_opt1:
        time_range = st.selectbox(
            "📅 選擇檢視的時間區間：",
            ["近 1 年", "近 3 年", "近 5 年", "近 10 年", "近 15 年", "全部歷史 (Max)"],
            index=3  # 預設選近 10 年
        )
    with col_opt2:
        use_secondary_y = st.checkbox("開啟雙 Y 軸顯示", value=True)
    with col_opt3:
        chart_height = st.slider("圖表高度：", min_value=400, max_value=800, value=500)

    # 金絲雀觸發避險背景設定
    with st.expander("🦅 金絲雀避險背景色塊設定 (Canary Trigger)", expanded=False):
        enable_canary = st.checkbox("啟用金絲雀動態避險背景色塊", value=False)
        tip_col_choice = None
        bm_col_choice = None
        if enable_canary and len(selected_indicators) >= 2:
            cc1, cc2 = st.columns(2)
            with cc1:
                tip_col_choice = st.selectbox("選擇 TIP 動能指標", selected_indicators, key="canary_tip")
            with cc2:
                bm_col_choice = st.selectbox("選擇 Benchmark (BM) 指標", selected_indicators, key="canary_bm")
            st.caption("💡 規則：當 TIP 指標 < 0 時，次月開始觸發避險；觸發期間若 BM 上漲顯示綠色，下跌顯示紅色。")

    if not selected_indicators:
        st.warning("⚠️ 請至少選擇一項指標進行繪圖與編輯！")
    else:
        dfs_to_merge = []
        with st.spinner("智慧連線 Yahoo / FRED API 擷取並完整對齊歷史數據中..."):
            for ind_name in selected_indicators:
                code_item = st.session_state.custom_indicators.get(ind_name, "")
                code = code_item if isinstance(code_item, str) else code_item.get("code", "")
                calc_mode = st.session_state.indicator_transforms.get(ind_name, "原始水準 (Level/Raw)")
                
                if code:
                    s_df = fetch_smart_data(code)
                    if not s_df.empty:
                        s_df['Date'] = pd.to_datetime(s_df['Date'])
                        s_df = s_df.sort_values('Date').set_index('Date')
                        series = s_df.iloc[:, 0]
                        
                        # 依照該指標各自選擇的處理方式計算
                        if calc_mode == "年增率 (YoY %)":
                            if len(series) > 1:
                                dates_idx = pd.Series(series.index)
                                avg_diff_days = (dates_idx.diff().dt.days).median()
                                if avg_diff_days > 60 and avg_diff_days <= 120:
                                    shift_n = 4   # 季資料 (Quarterly) 跨 4 期
                                elif avg_diff_days > 300:
                                    shift_n = 1   # 年資料 (Annual) 跨 1 期
                                else:
                                    shift_n = 12  # 月資料 (Monthly) 跨 12 期
                            else:
                                shift_n = 12
                            
                            processed = series.pct_change(shift_n) * 100
                        elif calc_mode == "月/日增額 (Diff)":
                            processed = series.diff()
                        else:
                            processed = series
                            
                        # 統一透過月底 (ME) 取最後有效值進行異質頻率對齊
                        res_series = processed.resample('ME').last().dropna()
                        res_df = res_series.to_frame(name=ind_name).reset_index()
                        dfs_to_merge.append(res_df)

        if dfs_to_merge:
            # 先根據選擇的時間區間計算出絕對的起始日期
            all_max_date = max([df['Date'].max() for df in dfs_to_merge])
            
            if time_range == "近 1 年":
                start_date = all_max_date - pd.DateOffset(years=1)
            elif time_range == "近 3 年":
                start_date = all_max_date - pd.DateOffset(years=3)
            elif time_range == "近 5 年":
                start_date = all_max_date - pd.DateOffset(years=5)
            elif time_range == "近 10 年":
                start_date = all_max_date - pd.DateOffset(years=10)
            elif time_range == "近 15 年":
                start_date = all_max_date - pd.DateOffset(years=15)
            else:
                start_date = pd.Timestamp.min

            # 針對個別資料框先做時間篩選，再進行 outer join 合併
            filtered_dfs = []
            for df in dfs_to_merge:
                filtered_df = df[df['Date'] >= start_date]
                filtered_dfs.append(filtered_df)

            combined_df = filtered_dfs[0]
            for next_df in filtered_dfs[1:]:
                combined_df = pd.merge(combined_df, next_df, on='Date', how='outer')
            
            combined_df = combined_df.sort_values('Date')
            
            # 對各個選定指標欄位各自獨立進行前向補值，填平中間因結算日微調產生的 None 斷層
            for ind in selected_indicators:
                if ind in combined_df.columns:
                    combined_df[ind] = combined_df[ind].ffill()
            
            # 轉換顯示格式
            combined_df['日期 (YYYY-MM-DD)'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
            display_cols = ['日期 (YYYY-MM-DD)'] + [c for c in combined_df.columns if c not in ['Date', '日期 (YYYY-MM-DD)']]
            display_df = combined_df[display_cols].copy()

            st.subheader("✏️ 數據線上編輯器 (修改數值、點擊 ＋ Add row 手動補充最新資料)")
            edited_df = st.data_editor(display_df, num_rows="dynamic", key="macro_editor")

            if not edited_df.empty:
                chart_df = edited_df.copy()
                chart_df['日期 (YYYY-MM-DD)'] = pd.to_datetime(chart_df['日期 (YYYY-MM-DD)'], errors='coerce')
                chart_df = chart_df.dropna(subset=['日期 (YYYY-MM-DD)']).sort_values('日期 (YYYY-MM-DD)')
                chart_df.set_index('日期 (YYYY-MM-DD)', inplace=True)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
                
                value_cols = [c for c in chart_df.columns if c != '日期 (YYYY-MM-DD)']
                for idx, col in enumerate(value_cols):
                    is_secondary = (idx > 0 and use_secondary_y)
                    chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
                    fig.add_trace(
                        go.Scatter(x=chart_df.index, y=chart_df[col], name=str(col), mode='lines+markers', line=dict(width=2.5, color=colors[idx % len(colors)]), connectgaps=True),
                        secondary_y=is_secondary
                    )

                # 【關鍵修正】：如果啟用了金絲雀避險背景色塊，改為直接判斷 TIP 指標數值本身 < 0，並自次月開始觸發
                if enable_canary and tip_col_choice:
                    if tip_col_choice in chart_df.columns and bm_col_choice in chart_df.columns:
                        temp_canary = chart_df.copy()
                        temp_canary['tip_neg'] = temp_canary[tip_col_choice] < 0  # 修正：直接判斷指標數值 < 0
                        temp_canary['canary_active'] = temp_canary['tip_neg'].shift(1).fillna(False)  # 次月開始生效
                        temp_canary['bm_return'] = temp_canary[bm_col_choice].pct_change(1)

                        dates_list = temp_canary.index.tolist()
                        for i in range(1, len(dates_list)):
                            if temp_canary['canary_active'].iloc[i]:
                                bm_ret = temp_canary['bm_return'].iloc[i]
                                is_up = bm_ret >= 0 if pd.notna(bm_ret) else True
                                color = "rgba(0, 255, 0, 0.15)" if is_up else "rgba(255, 0, 0, 0.15)"
                                
                                fig.add_vrect(
                                    x0=dates_list[i-1],
                                    x1=dates_list[i],
                                    fillcolor=color,
                                    opacity=0.6,
                                    layer="below",
                                    line_width=0,
                                )

                fig.update_layout(
                    title=f"全球市場與總經數據互動對比圖 ({time_range})",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_dark",
                    height=chart_height
                )
                fig.update_xaxes(title_text="日期")
                fig.update_yaxes(title_text="主指標 (Left Axis)", secondary_y=False)
                if use_secondary_y and len(value_cols) > 1:
                    fig.update_yaxes(title_text="對比指標 (Right Axis)", secondary_y=True)

                st.subheader("📈 市場趨勢雙 Y 軸動態圖表")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ 無法連線讀取數據，請確認指標代碼或網路連線。")

# ---------------------------------------------------------
# 模組三：基金 / ETF 交易決策評估
# ---------------------------------------------------------
elif app_mode == "🎯 基金 / ETF 交易決策評估":
    st.header("🎯 基金 / ETF 投資決策與評估報告生成器")
    
    if "fund_report" not in st.session_state:
        st.session_state.fund_report = None
    if "fund_prompt_info" not in st.session_state:
        st.session_state.fund_prompt_info = {}

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fund_name = st.text_input("輸入基金 / ETF / 股票標的", placeholder="例如：1301.TW、0050、IEF ETF、元大美債20年")
    with col2:
        action_type = st.selectbox("擬執行交易方向", ["買進 / 建倉 (Buy)", "賣出 / 減碼 (Sell)", "觀望 / 持有 (Hold)"])
    with col3:
        lang_choice = st.selectbox("報告語言風格", ["繁體中文 (Traditional Chinese)", "英文 (English)", "中英雙語對照 (Bilingual)"])

    if st.button("🚀 生成個案投資評估報告", type="primary", use_container_width=True):
        if not fund_name.strip():
            st.warning("⚠️ 請輸入標的名稱或代碼！")
        else:
            with st.spinner(f"正在抓取 {fund_name} 最新資料與基本面..."):
                encoded_query = urllib.parse.quote(fund_name)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}+OR+聯準會+OR+美債殖利率+OR+通膨&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                feed = feedparser.parse(rss_url)
                
                news_list = []
                for entry in feed.entries[:6]:
                    title = clean_html(entry.get('title', ''))
                    published = entry.get('published', '')
                    summary = clean_html(entry.get('summary', ''))[:200]
                    news_list.append(f"【時間: {published}】\n標題: {title}\n摘要: {summary}\n")
                
                market_data = "\n".join(news_list) if news_list else "暫無具體即時新聞，將基於資產常規屬性分析。"

            lang_instruction = "全篇報告請使用「標準繁體中文」。"
            if lang_choice == "英文 (English)":
                lang_instruction = "Please write the entire report in Professional English."
            elif lang_choice == "中英雙語對照 (Bilingual)":
                lang_instruction = "每個段落請先提供「繁體中文」，隨後附上對應的「英文翻譯 (English Translation)」。"

            prompt = f"""
你是一位機構級資深基金分析師與首席投資策略官。請針對標的【{fund_name}】，撰寫一份包含**分拆獨立表格基本檔案**與**深度決策評估**的專業機構報告。

【基本交易資訊】：
- 標的輸入：{fund_name}
- 擬執行交易方向：{action_type}
- 語言要求：{lang_instruction}

【即時市場新聞與數據】：
{market_data}

【撰寫格式與結構規範（請將各資料分拆為獨立 Markdown 表格，絕對不要混在一張表內）】：

### 📌 零、標的基本檔案與配置概況 (Basic Profile)

#### 1. 基金 / ETF 基本資訊
| 項目 | 內容/數值 |
| :--- | :--- |
| **基金/ETF 中文全稱** | (正確中文名稱) |
| **基金/ETF 英文全稱** | (正確英文全稱) |
| **交易所 / 股票代碼** | (Ticker / Code) |
| **追蹤指數 / 標的屬性** | (Benchmark Index / Asset Class) |
| **基金規模 (AUM)** | (最新預估規模) |
| **經理費 / 總內扣費用 (TER)**| (Expense Ratio) |

#### 2. 歷史績效表現 (Performance Track Record)
| 期間 | MTD | YTD | 1M | 3M | 6M | 1Yr | 3Yr (年化) | 5Yr (年化) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **總報酬率 (%)** | (數據/估計) | (數據/估計) | (數據/估計) | (數據/估計) | (數據/估計) | (數據/估計) | (數據/估計) | (數據/估計) |

#### 3. 前十大持股 (Top 10 Holdings)
| 排序 | 持股 / 標的名稱 | 估計權重 (%) |
| :--- | :--- | :--- |
| 1 | (持股名稱 1) | (權重 1%) |
| ... | ... | ... |

#### 4. 主要產業與國家配置分布 (Sectors & Geographic Allocation)
| 主要產業 (Sectors) | 占比 (%) | 主要國家/地區 (Geographic) | 占比 (%) |
| :--- | :--- | :--- | :--- |
| (產業 1) | (%) | (國家 1) | (%) |

#### 5. 關鍵風險與固定收益專屬指標
| 專屬風險指標 | 內容 / 數值 | 說明 |
| :--- | :--- | :--- |
| **修正存續期間 (Modified Duration)** | (例如：6.8 年) | （對利率變動之價格敏感度） |
| **30 天 SEC 殖利率 / 到期殖利率 (Yield)** | (例如：4.85%) | （最新年化收益率） |

---

### 一、當前總體經濟環境與市場脈絡分析
### 二、標的屬性與最新市場衝擊評估 ({fund_name})
### 三、買賣方向 ({action_type}) 可行性評估與風控/停損策略
"""
            with st.spinner("🤖 Qwen 分析師正在編製獨立結構表格與撰寫評估報告..."):
                report, err = call_qwen_api([{"role": "user", "content": prompt}])
                if report:
                    st.session_state.fund_report = report
                    st.session_state.fund_prompt_info = {
                        "fund_name": fund_name,
                        "action_type": action_type,
                        "prompt": prompt
                    }
                    st.success("✅ 獨立表格化基本檔案與決策報告生成完畢！")
                else:
                    st.error(f"❌ 生成失敗: {err}")

    if st.session_state.fund_report:
        st.markdown("---")
        st.subheader(f"📈 《{st.session_state.fund_prompt_info.get('fund_name')}》- 標的表格檔案與決策評估報告")
        st.markdown(st.session_state.fund_report)
        
        st.download_button(
            "📥 下載完整評估報告 (.txt)",
            st.session_state.fund_report,
            file_name=f"{st.session_state.fund_prompt_info.get('fund_name')}_Report.txt"
        )
