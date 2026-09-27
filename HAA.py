@st.cache_data(ttl=3600)
def load_yahoo_data(start_str, end_str):
    """連線 Yahoo Finance 自動下載美金計價含息 (Adj Close / Adjusted Close) 之歷史每日與月底價格"""
    tickers = list(TICKER_MAP.values())
    
    # 使用 auto_adjust=True 讓 Yahoo Finance 自動將歷史股利與債息調整回算至 Close 欄位
    df_download = yf.download(tickers, start=start_str, end=end_str, interval="1d", auto_adjust=True)
    
    # 針對 yfinance 新版 MultiIndex 結構進行相容性處理
    if isinstance(df_download.columns, pd.MultiIndex):
        if 'Close' in df_download.columns.levels[0]:
            df_download = df_download['Close']
        elif 'Adj Close' in df_download.columns.levels[0]:
            df_download = df_download['Adj Close']
    else:
        if 'Close' in df_download.columns:
            df_download = df_download[['Close']]
        elif 'Adj Close' in df_download.columns:
            df_download = df_download[['Adj Close']]
            
    inv_map = {v: k for k, v in TICKER_MAP.items()}
    df_daily = df_download.rename(columns=inv_map).reset_index()
    
    # 確保 Date 欄位格式乾淨
    date_col_name = df_daily.columns[0]
    df_daily[date_col_name] = pd.to_datetime(df_daily[date_col_name]).dt.strftime('%Y-%m-%d')
    if date_col_name != 'Date':
        df_daily = df_daily.rename(columns={date_col_name: 'Date'})
    
    # 自動重採樣為每月月底價格
    df_temp = df_daily.copy()
    df_temp['Date_dt'] = pd.to_datetime(df_temp['Date'])
    df_monthly = df_temp.groupby(df_temp['Date_dt'].dt.to_period('M')).last().reset_index(drop=True)
    df_monthly['Date'] = df_monthly['Date_dt'].dt.strftime('%Y-%m-%d')
    df_monthly = df_monthly.drop(columns=['Date_dt'])
    
    return df_daily, df_monthly
