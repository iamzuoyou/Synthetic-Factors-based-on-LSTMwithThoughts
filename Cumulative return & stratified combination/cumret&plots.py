import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

init_dir = os.getcwd()
#创建路径
predata_dir = init_dir+"/"+"10prediction"
weight_dir = init_dir+"/"+"10weights"
dailyret_dir = init_dir+"/"+"dailyret"
results_dir = init_dir+"/"+"results"
for idir in [results_dir]:
    if not os.path.exists(idir):
        os.makedirs(idir)
        
#股票换仓日预测值数据
os.chdir(predata_dir)
warnings.filterwarnings('ignore')
final_predata = pd.DataFrame(columns=["Date","Stock","Prediction"])
for predatafile in os.listdir(predata_dir):
    predata = pd.read_csv(predatafile)
    predata["Date"] = predatafile[:10]
    predata_sorted = predata.sort_values(by="Stock",ascending=1)
    predata_sorted["Rank"] = predata_sorted["Prediction"].rank(method="first",ascending=0)  #根据预测因子值降序排序
    final_predata = pd.concat([final_predata,predata_sorted],axis=0)
final_predata.columns = ["Date","Stkcd","Prediction","Rank"]

#权重数据
os.chdir(weight_dir)
warnings.filterwarnings('ignore')
final_weight = pd.DataFrame(columns=["Date","Stkcd","Min Volatility weight","Max Sharpe Ratio weight","Equal Weighted Weight"])
for weightfile in os.listdir(weight_dir):
    weight = pd.read_excel(weightfile)
    weight["Date"] = weightfile[:10]
    weight["Equal Weighted Weight"] = [0.1]*10
    final_weight = pd.concat([final_weight,weight],axis=0)

#日度复权收益率(不考虑现金股利再投资)
os.chdir(dailyret_dir)
warnings.filterwarnings('ignore')
dailyret = pd.DataFrame(columns=["Stkcd","Trddt","Dretnd"])
for drfile in os.listdir(dailyret_dir):
    dr = pd.read_excel(drfile)
    dailyret = pd.concat([dailyret,dr],axis=0)
dailyret.columns = ["Stkcd","Date","Dretnd"]
os.chdir(init_dir)
dailyret.to_csv("dailyret.csv")

#计算前十组合累计收益率
os.chdir(init_dir)
dailyret = pd.read_csv("dailyret.csv")
os.chdir(results_dir)
adj_date_lis = ["2023-01-03","2023-04-03","2023-07-03","2023-10-09","2024-01-02","2024-04-01","2024-07-01","2024-10-08","2025-01-01"]
dret10_lis = []
for i in range(8):
    adj_date = adj_date_lis[i]
    hold_portfolio_lis = final_weight[final_weight["Date"]==adj_date]
    hold_period_ret_lis = dailyret[(dailyret["Date"]<adj_date_lis[i+1]) & (dailyret["Date"]>=adj_date_lis[i])] #仅保留该持仓期间的收益率数据
    date_lis = list(set(hold_period_ret_lis["Date"]))
    date_lis.sort()
    for date in date_lis:
        minvol10_ret = 0
        maxsharpe10_ret = 0
        equalweight10_ret = 0
        for stkcd in list(hold_portfolio_lis["Stkcd"]):
            minvol10_ret += (list(hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Stkcd"]==stkcd)]["Dretnd"])[0]
                             *list(hold_portfolio_lis[hold_portfolio_lis["Stkcd"]==stkcd]["Min Volatility weight"])[0])
            maxsharpe10_ret += (list(hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Stkcd"]==stkcd)]["Dretnd"])[0]
                                *list(hold_portfolio_lis[hold_portfolio_lis["Stkcd"]==stkcd]["Max Sharpe Ratio weight"])[0])
            equalweight10_ret += (list(hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Stkcd"]==stkcd)]["Dretnd"])[0]
                                  *list(hold_portfolio_lis[hold_portfolio_lis["Stkcd"]==stkcd]["Equal Weighted Weight"])[0])
        dret10_lis.append([date,minvol10_ret,maxsharpe10_ret,equalweight10_ret])
dret10 = pd.DataFrame(dret10_lis,columns = ["Date","10 Min Volatility Ret","10 Max Sharpe Ratio Ret","10 Equal Weighted Ret"])
dret10.to_csv("10支股票组合日收益率.csv")
cumret10 = pd.read_csv("10支股票组合日收益率.csv",index_col=[1])
cumret10.drop(columns=["Unnamed: 0"],inplace=True)
print(cumret10)
for column in ["10 Min Volatility Ret","10 Max Sharpe Ratio Ret","10 Equal Weighted Ret"]:
    cumret10[column] = np.log(1+cumret10[column])
    cumret10[column] = cumret10[column].cumsum()
    cumret10[column] = (np.exp(cumret10[column])-1)*100
cumret10.columns = ["10_MinVolatility_CumRet(%)","10_MaxSharpeRatio_CumRet(%)","10_Equal_Weighted_CumRet(%)"]
cumret10.to_csv("10支股票组合日累计收益率.csv")
#画前十组合收益率分布图
plt.rcParams["font.sans-serif"]=["SimHei","Times New Roman"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(num=1,figsize=(50, 8))
fig,ax = plt.subplots()
ax.plot(cumret10.index,cumret10["10_Equal_Weighted_CumRet(%)"],label="10_Equal_Weighted_CumRet",color="slategrey")
ax.plot(cumret10.index,cumret10["10_MinVolatility_CumRet(%)"],label="10_MinVolatility_CumRet",color="cornflowerblue")
ax.plot(cumret10.index,cumret10["10_MaxSharpeRatio_CumRet(%)"],label="10_MaxSharpeRatio_CumRet",color="midnightblue")
ax.spines["top"].set_visible(False)  #设置坐标轴,下同
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data",0))
ax.legend(loc="upper left")
plt.xlabel("Date",labelpad=50)
plt.ylabel("(%)",labelpad=50)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
plt.savefig("10支股票组合日累计收益率.png")
plt.draw()
plt.close()

#绘制效果最好的组合与上证指数的对比图
os.chdir(init_dir)
sz = pd.read_excel("000001.SH.xlsx")
for i in range(len(sz)):
    y,m,d = [int(j) for j in sz["Date"][i].split("-")]
    sz["Date"][i] = "%d-%02d-%02d"%(y,m,d)
os.chdir(results_dir)
port10 = pd.read_csv("10支股票组合日累计收益率.csv")
data = pd.merge(port10,sz,on=["Date"],how="left")
data["000001SH_CumRet"] = np.log(1+data["000001SHRet"])
data["000001SH_CumRet"] = data["000001SH_CumRet"].cumsum()
data["000001SH_CumRet"] = (np.exp(data["000001SH_CumRet"])-1)*100
data.index = data["Date"]
#画图
plt.rcParams["font.sans-serif"]=["SimHei","Times New Roman"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(num=1,figsize=(50, 8))
fig,ax = plt.subplots()
ax.plot(data.index,data["000001SH_CumRet"],label="000001SH_CumRet",color="red",linestyle="--")
ax.plot(data.index,data["10_MaxSharpeRatio_CumRet(%)"],label="10_MaxSharpeRatio_CumRet",color="midnightblue")
ax.spines["top"].set_visible(False)  #设置坐标轴,下同
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data",0))
ax.legend(loc="upper left")
plt.xlabel("Date",labelpad=50)
plt.ylabel("(%)",labelpad=50)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
plt.savefig("最优策略vs上证指数日累计收益率.png")
plt.draw()
plt.close()

#计算分层累计收益率（使用等权组合）
os.chdir(init_dir)
dailyret = pd.read_csv("dailyret.csv")
os.chdir(results_dir)
adj_date_lis = ["2023-01-03","2023-04-03","2023-07-03","2023-10-09","2024-01-02","2024-04-01","2024-07-01","2024-10-08","2025-01-01"]
dret5layer_lis = []
for i in range(8):
    adj_date = adj_date_lis[i]
    hold_portfolio_lis = final_predata[final_predata["Date"]==adj_date]
    hold_portfolio_lis.drop(columns=["Date"],inplace=True)
    hold_period_ret_lis = dailyret[(dailyret["Date"]<adj_date_lis[i+1]) & (dailyret["Date"]>=adj_date_lis[i])] #仅保留该持仓期间的收益率数据
    hold_period_ret_lis = pd.merge(hold_period_ret_lis,hold_portfolio_lis,on=["Stkcd"],how="left")
    hold_period_ret_lis["Rank"] = hold_period_ret_lis.groupby("Stkcd")["Rank"].ffill()
    hold_period_ret_lis.dropna(inplace=True)
    date_lis = list(set(hold_period_ret_lis["Date"]))
    date_lis.sort()
    max_rank = max(list(hold_period_ret_lis["Rank"]))
    for date in date_lis:
        first_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=int(max_rank/5))]["Dretnd"]
        second_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=int(2*max_rank/5)) & (hold_period_ret_lis["Rank"]>int(max_rank/5))]["Dretnd"]
        third_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=int(3*max_rank/5)) & (hold_period_ret_lis["Rank"]>int(2*max_rank/5))]["Dretnd"]
        forth_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=int(4*max_rank/5)) & (hold_period_ret_lis["Rank"]>int(3*max_rank/5))]["Dretnd"]
        fifth_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=int(5*max_rank/5)) & (hold_period_ret_lis["Rank"]>int(4*max_rank/5))]["Dretnd"]
        first_ret = sum(list(first_ret_lis))/len(list(first_ret_lis))
        second_ret = sum(list(second_ret_lis))/len(list(second_ret_lis))
        third_ret = sum(list(third_ret_lis))/len(list(third_ret_lis))
        forth_ret = sum(list(forth_ret_lis))/len(list(forth_ret_lis))
        fifth_ret = sum(list(fifth_ret_lis))/len(list(fifth_ret_lis))
        dret5layer_lis.append([date,first_ret,second_ret,third_ret,forth_ret,fifth_ret])
dret5layer = pd.DataFrame(dret5layer_lis,columns = ["Date","1st Ret","2nd Ret","3rd Ret","4th Ret","5th Ret"])
dret5layer.to_csv("分5层日收益率.csv")
cumret5layer = pd.read_csv("分5层日收益率.csv",index_col=[1])
cumret5layer.drop(columns=["Unnamed: 0"],inplace=True)
print(cumret5layer)
for column in ["1st Ret","2nd Ret","3rd Ret","4th Ret","5th Ret"]:
    cumret5layer[column] = np.log(1+cumret5layer[column])
    cumret5layer[column] = cumret5layer[column].cumsum()
    cumret5layer[column] = (np.exp(cumret5layer[column])-1)*100
cumret5layer.columns = ["1st_CumRet(%)","2nd_CumRet(%)","3rd_CumRet(%)","4th_CumRet(%)","5th_CumRet(%)"]
cumret5layer.to_csv("分5层日累计收益率.csv")
#画分5层收益率分布图
plt.rcParams["font.sans-serif"]=["SimHei","Times New Roman"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(num=1,figsize=(50, 8))
fig,ax = plt.subplots()
ax.plot(cumret5layer.index,cumret5layer["1st_CumRet(%)"],label="1st_CumRet",color="midnightblue")
ax.plot(cumret5layer.index,cumret5layer["2nd_CumRet(%)"],label="2nd_CumRet",color="royalblue")
ax.plot(cumret5layer.index,cumret5layer["3rd_CumRet(%)"],label="3rd_CumRet",color="cornflowerblue")
ax.plot(cumret5layer.index,cumret5layer["4th_CumRet(%)"],label="4th_CumRet",color="lightsteelblue")
ax.plot(cumret5layer.index,cumret5layer["5th_CumRet(%)"],label="5th_CumRet",color="slategrey")
ax.spines["top"].set_visible(False)  #设置坐标轴,下同
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data",0))
ax.legend(loc="upper left")
plt.xlabel("Date",labelpad=50)
plt.ylabel("(%)",labelpad=50)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
plt.savefig("分5层组合日累计收益率.png")
plt.draw()
plt.close()

#计算多空累计收益率（使用等权组合）
os.chdir(init_dir)
dailyret = pd.read_csv("dailyret.csv")
os.chdir(results_dir)
adj_date_lis = ["2023-01-03","2023-04-03","2023-07-03","2023-10-09","2024-01-02","2024-04-01","2024-07-01","2024-10-08","2025-01-01"]
dretduokong_lis = []
for i in range(8):
    adj_date = adj_date_lis[i]
    hold_portfolio_lis = final_predata[final_predata["Date"]==adj_date]
    hold_portfolio_lis.drop(columns=["Date"],inplace=True)
    hold_period_ret_lis = dailyret[(dailyret["Date"]<adj_date_lis[i+1]) & (dailyret["Date"]>=adj_date_lis[i])] #仅保留该持仓期间的收益率数据
    hold_period_ret_lis = pd.merge(hold_period_ret_lis,hold_portfolio_lis,on=["Stkcd"],how="left")
    hold_period_ret_lis["Rank"] = hold_period_ret_lis.groupby("Stkcd")["Rank"].ffill()
    hold_period_ret_lis.dropna(inplace=True)
    date_lis = list(set(hold_period_ret_lis["Date"]))
    date_lis.sort()
    max_rank = max(list(hold_period_ret_lis["Rank"]))
    for date in date_lis:
        duo_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]<=10)]["Dretnd"]
        kong_ret_lis = hold_period_ret_lis[(hold_period_ret_lis["Date"]==date) & (hold_period_ret_lis["Rank"]>=max_rank-9)]["Dretnd"]*(-1)
        duokong_ret_lis = list(duo_ret_lis)+list(kong_ret_lis)
        duokong_ret = sum(duokong_ret_lis)/len(duokong_ret_lis)
        dretduokong_lis.append([date,duokong_ret])
dretduokong = pd.DataFrame(dretduokong_lis,columns = ["Date","Duokong Ret"])
dretduokong.to_csv("多空组合日收益率.csv")
cumretduokong = pd.read_csv("多空组合日收益率.csv")
cumretduokong.drop(columns=["Unnamed: 0"],inplace=True)
print(cumretduokong)
for column in ["Duokong Ret"]:
    cumretduokong[column] = np.log(1+cumretduokong[column])
    cumretduokong[column] = cumretduokong[column].cumsum()
    cumretduokong[column] = (np.exp(cumretduokong[column])-1)*100
cumretduokong.columns = ["Date","Duokong_CumRet(%)"]
cumretduokong.to_csv("多空组合日累计收益率.csv")
#合并其他数据
os.chdir(init_dir)
sz = pd.read_excel("000001.SH.xlsx")
for i in range(len(sz)):
    y,m,d = [int(j) for j in sz["Date"][i].split("-")]
    sz["Date"][i] = "%d-%02d-%02d"%(y,m,d)
os.chdir(results_dir)
port10 = pd.read_csv("10支股票组合日累计收益率.csv")
data = pd.merge(port10,sz,on=["Date"],how="left")
data["000001SH_CumRet"] = np.log(1+data["000001SHRet"])
data["000001SH_CumRet"] = data["000001SH_CumRet"].cumsum()
data["000001SH_CumRet"] = (np.exp(data["000001SH_CumRet"])-1)*100
data = pd.merge(data,cumretduokong,on=["Date"],how="left")
data.index = data["Date"]
#画多空组合收益率分布图
plt.rcParams["font.sans-serif"]=["SimHei","Times New Roman"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(num=1,figsize=(50, 8))
fig,ax = plt.subplots()
ax.plot(data.index,data["10_Equal_Weighted_CumRet(%)"],label="10_Equal_Weighted_CumRet",color="midnightblue")
ax.plot(data.index,data["Duokong_CumRet(%)"],label="Duokong_CumRet",color="slategrey")
ax.plot(data.index,data["000001SH_CumRet"],label="000001SH_CumRet",color="red",linestyle="--")
ax.spines["top"].set_visible(False)  #设置坐标轴,下同
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data",0))
ax.legend(loc="upper left")
plt.xlabel("Date",labelpad=50)
plt.ylabel("(%)",labelpad=50)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
plt.savefig("多空组合日累计收益率.png")
plt.draw()
plt.close()

