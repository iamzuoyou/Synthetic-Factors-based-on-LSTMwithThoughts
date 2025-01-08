import time
import pandas as pd
import chinadata.ca_data as ts

pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")

#查询目前所有上市交易的股票列表
df_stkinfolis = pro.stock_basic(fields='ts_code,symbol,name,area,industry,list_date,market,exchange,curr_type,list_status,list_date,delist_date')
df_stkinfolis.to_csv("股票基本信息.csv")
df_stkinfolis = df_stkinfolis[df_stkinfolis["exchange"]=="SSE"]  #仅使用上交所数据


#爬取各股票的技术因子
df_outputparams = pd.read_excel("Tushare-股票技术面因子(专业版)-输出参数list.xlsx")
df_factor = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s因子数据"%(ts_code_i))
    df_factor_i = pro.stk_factor_pro(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_factor = df_factor._append(df_factor_i)
    time.sleep(2)
df_factor.to_csv("上证股票技术因子数据200101-241217.csv")


#爬取股票行情数据
#日线
df_outputparams = pd.read_excel("Tushare-A股日线行情-输出参数list.xlsx")
df_daily = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s日线行情数据"%(ts_code_i))
    df_daily_i = pro.daily(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_daily = df_daily._append(df_daily_i)
    time.sleep(0.12)
df_daily.to_csv("上证股票日线行情数据200101-241217.csv")
#周线
df_outputparams = pd.read_excel("Tushare-A股周线行情-输出参数list.xlsx")
df_weekly = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s周线行情数据"%(ts_code_i))
    df_weekly_i = pro.weekly(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_weekly = df_weekly._append(df_weekly_i)
df_weekly.to_csv("上证股票周线行情数据200101-241217.csv")
#月线
df_outputparams = pd.read_excel("Tushare-A股月线行情-输出参数list.xlsx")
df_monthly = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s月线行情数据"%(ts_code_i))
    df_monthly_i = pro.monthly(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_monthly = df_monthly._append(df_monthly_i)
df_monthly.to_csv("上证股票月线行情数据200101-241217.csv")
#复权因子
df_outputparams = pd.read_excel("Tushare-A股复权因子-输出参数list.xlsx")
df_adj_factor = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s复权因子数据"%(ts_code_i))
    df_adj_factor_i = pro.adj_factor(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_adj_factor = df_adj_factor._append(df_adj_factor_i)
df_adj_factor.to_csv("上证股票复权因子数据200101-241217.csv")
#每日指标
df_outputparams = pd.read_excel("Tushare-A股每日指标-输出参数list.xlsx")
df_daily_basic = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s每日指标数据"%(ts_code_i))
    df_daily_basic_i = pro.daily_basic(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_daily_basic = df_daily_basic._append(df_daily_basic_i)
df_daily_basic.to_csv("上证股票每日指标数据200101-241217.csv")


#爬取资金流向数据
#行业资金流向
df_outputparams = pd.read_excel("Tushare-行业资金流向-输出参数list.xlsx")
df_moneyflow_ind_ths = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s行业资金流向数据"%(ts_code_i))
    df_moneyflow_ind_ths_i = pro.moneyflow_ind_ths(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_moneyflow_ind_ths = df_moneyflow_ind_ths._append(df_moneyflow_ind_ths_i)
df_moneyflow_ind_ths.to_csv("行业资金流向数据200101-241217.csv")
#板块资金流向
df_outputparams = pd.read_excel("Tushare-板块资金流向-输出参数list.xlsx")
df_moneyflow_ind_dc = pd.DataFrame(columns=list(df_outputparams["名称"]))
for ts_code_i in list(df_stkinfolis["ts_code"]):
    pro = ts.pro_api(token="e9e84ed87f29cf43fdac84cdbb14d306787")
    print("开始爬取%s板块资金流向数据"%(ts_code_i))
    df_moneyflow_ind_dc_i = pro.moneyflow_ind_dc(ts_code=ts_code_i,start_date="20200101",end_date="20241217")
    df_moneyflow_ind_dc = df_moneyflow_ind_dc._append(df_moneyflow_ind_dc_i)
df_moneyflow_ind_dc.to_csv("板块资金流向数据200101-241217.csv")






