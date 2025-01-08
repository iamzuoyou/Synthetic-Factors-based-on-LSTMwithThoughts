import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from scipy.stats import pearsonr, spearmanr  # 用于计算 Pearson 和 Spearman 相关系数

class BackTest:
    def __init__(self, output_folder, label_file):
        """
        初始化 BackTest 类。

        参数:
        - output_folder: str, 包含 Predictions 文件夹的路径。
        - label_file: str, 包含实际收益率数据的文件路径。
        """
        self.output_folder = output_folder
        self.label_file = label_file

        # 检查 label_file 是否存在
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"文件 {self.label_file} 不存在，请检查路径")

        # 加载实际收益率数据
        self.label_data = pd.read_csv(self.label_file)

        # 统一股票代码格式
        self.label_data['Stkcd'] = self.label_data['Stkcd'].astype(str).str.zfill(6)
        self.label_data['Trdmnt'] = pd.to_datetime(self.label_data['Trdmnt'], format='%y-%b').dt.strftime('%Y-%m')


    def run(self):
        """
        运行回测并生成累计收益率、IC（Pearson）、Rank IC（Spearman）和 ICIR 的 CSV 文件和绘图结果。
        """
        # 获取 Predictions 文件夹中的所有训练轮文件夹（01 到 10）
        run_folders = sorted([
            d for d in os.listdir(os.path.join(self.output_folder, 'Predictions'))
            if os.path.isdir(os.path.join(self.output_folder, 'Predictions', d)) and re.match(r'^\d{2}$', d)
        ])

        # 如果没有找到训练轮文件夹，直接返回
        if not run_folders:
            print("未找到训练轮文件夹（01 到 10），请检查 Predictions 文件夹")
            return

        # 初始化累计收益率、IC（Pearson）、Rank IC（Spearman）和 ICIR
        cumulative_returns = {}  # 累计收益率
        ic_values = {}           # IC（Pearson）值
        rank_ic_values = {}      # Rank IC（Spearman）值
        icir_values = {}         # ICIR 值

        # 遍历每个训练轮文件夹
        for run_folder in run_folders:
            run = int(run_folder)  # 训练轮编号
            run_path = os.path.join(self.output_folder, 'Predictions', run_folder)

            # 获取该训练轮中的所有预测文件（格式为 YYYYMMDD.csv）
            prediction_files = sorted([
                f for f in os.listdir(run_path)
                if os.path.isfile(os.path.join(run_path, f)) and re.match(r'^\d{8}\.csv$', f)
            ])

            # 如果没有找到预测文件，跳过该训练轮
            if not prediction_files:
                print(f"训练轮 {run_folder} 中没有找到预测文件，跳过")
                continue

            # 初始化该训练轮的累计收益率、IC 和 Rank IC
            cumulative_returns[run] = []
            ic_values[run] = []
            rank_ic_values[run] = []

            # 遍历每个预测文件
            for prediction_file in prediction_files:
                # 提取调仓日（从文件名中提取，例如 20230103.csv -> 20230103）
                date = os.path.splitext(prediction_file)[0]

                # 将调仓日转换为 yyyy-mm 格式
                try:
                    rebalance_month = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m')
                except ValueError:
                    print(f"调仓日 {date} 格式错误，跳过")
                    continue

                # 检查月份是否在 label_monthly_return.csv 中
                if rebalance_month not in self.label_data['Trdmnt'].unique():
                    print(f"月份 {rebalance_month} 在 label_monthly_return.csv 中未找到，跳过")
                    continue

                # 读取预测数据
                predictions = pd.read_csv(os.path.join(run_path, prediction_file))

                # 统一股票代码格式
                predictions['Stock'] = predictions['Stock'].astype(str).str.zfill(6)

                # 确保列名正确
                if 'Stock' not in predictions.columns or 'Prediction' not in predictions.columns:
                    print(f"文件 {prediction_file} 列名不匹配，跳过")
                    continue

                # 获取预测值和实际值
                pred_returns = []
                actual_returns = []
                for stock in predictions['Stock']:
                    pred_return = predictions[predictions['Stock'] == stock]['Prediction'].values[0]
                    actual_return = self.label_data[
                        (self.label_data['Stkcd'] == stock) & (self.label_data['Trdmnt'] == rebalance_month)
                    ]['Mretwd'].values
                    if len(actual_return) > 0:
                        pred_returns.append(pred_return)
                        actual_returns.append(actual_return[0])

                # 计算 IC（Pearson 相关系数）
                if len(pred_returns) > 0 and len(actual_returns) > 0:
                    ic, _ = pearsonr(pred_returns, actual_returns)
                    ic_values[run].append(ic)
                else:
                    ic_values[run].append(0)  # 如果没有数据，IC 设为 0

                # 计算 Rank IC（Spearman 秩相关系数）
                if len(pred_returns) > 0 and len(actual_returns) > 0:
                    rank_ic, _ = spearmanr(pred_returns, actual_returns)
                    rank_ic_values[run].append(rank_ic)
                else:
                    rank_ic_values[run].append(0)  # 如果没有数据，Rank IC 设为 0

                # 选择收益率最高的十只股票
                top_10_stocks = predictions.nlargest(10, 'Prediction')

                # 查找这十只股票在调仓日之后三个月的实际收益率并累加
                actual_returns = []
                for stock in top_10_stocks['Stock']:
                    # 获取调仓日之后的三个月
                    rebalance_date = datetime.strptime(date, '%Y%m%d')
                    next_month_1 = (rebalance_date.replace(day=1)).strftime('%Y-%m')
                    next_month_2 = (rebalance_date.replace(day=1) + pd.DateOffset(months=1)).strftime('%Y-%m')
                    next_month_3 = (rebalance_date.replace(day=1) + pd.DateOffset(months=2)).strftime('%Y-%m')

                    # 获取这三个月的实际收益率并累加
                    stock_return_1 = self.label_data[
                        (self.label_data['Stkcd'] == stock) & (self.label_data['Trdmnt'] == next_month_1)
                    ]['Mretwd'].values
                    stock_return_2 = self.label_data[
                        (self.label_data['Stkcd'] == stock) & (self.label_data['Trdmnt'] == next_month_2)
                    ]['Mretwd'].values
                    stock_return_3 = self.label_data[
                        (self.label_data['Stkcd'] == stock) & (self.label_data['Trdmnt'] == next_month_3)
                    ]['Mretwd'].values

                    total_return = 0
                    if len(stock_return_1) > 0:
                        total_return += stock_return_1[0]
                    if len(stock_return_2) > 0:
                        total_return += stock_return_2[0]
                    if len(stock_return_3) > 0:
                        total_return += stock_return_3[0]

                    actual_returns.append(total_return)

                # 计算这十只股票的平均实际收益率
                if len(actual_returns) > 0:
                    avg_return = np.mean(actual_returns)
                else:
                    avg_return = 0  # 如果没有找到实际收益率，设为0

                # 更新累计收益率
                cumulative_returns[run].append(avg_return)

            cumulative_returns[run] = np.cumsum(np.array(cumulative_returns[run]))

            # 计算 ICIR
            if len(ic_values[run]) > 0:
                ic_mean = np.mean(ic_values[run])  # IC 均值
                ic_std = np.std(ic_values[run])    # IC 标准差
                icir_values[run] = ic_mean / ic_std if ic_std != 0 else 0  # ICIR
            else:
                icir_values[run] = 0  # 如果没有数据，ICIR 设为 0

        # 如果没有找到有效的累计收益率数据，直接返回
        if not cumulative_returns:
            print("没有找到有效的累计收益率数据，无法生成图表")
            return

        # 创建 Result 文件夹（如果不存在）
        result_folder = os.path.join(os.path.dirname(self.output_folder), 'Result')
        os.makedirs(result_folder, exist_ok=True)

        # 保存累计收益率数据到 CSV 文件
        cumulative_returns_df = pd.DataFrame(cumulative_returns)
        cumulative_returns_df.to_csv(os.path.join(result_folder, 'Cumulative_Returns.csv'), index=False)

        # 绘制累计收益率曲线
        plt.figure(figsize=(10, 6))
        for run in cumulative_returns:
            plt.plot(cumulative_returns[run], label=f'Run {run:02d}')

        plt.title('Cumulative Returns by Training Run (Based on Actual Returns)')
        plt.xlabel('Rebalance Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)

        # 保存累计收益率图表为 PNG 文件
        output_image_path = os.path.join(result_folder, 'Cumulative_Return.png')
        plt.savefig(output_image_path)
        plt.close()

        print(f"累计收益率图表已保存至: {output_image_path}")

        # 保存 IC（Pearson）数据到 CSV 文件
        ic_values_df = pd.DataFrame(ic_values)
        ic_values_df.to_csv(os.path.join(result_folder, 'IC_Values.csv'), index=False)

        # 绘制 IC（Pearson）曲线
        plt.figure(figsize=(10, 6))
        for run in ic_values:
            plt.plot(ic_values[run], label=f'Run {run:02d}')

        plt.title('IC (Pearson) by Training Run')
        plt.xlabel('Rebalance Date')
        plt.ylabel('IC (Pearson)')
        plt.legend()
        plt.grid(True)

        # 保存 IC（Pearson）图表为 PNG 文件
        ic_image_path = os.path.join(result_folder, 'IC_Pearson.png')
        plt.savefig(ic_image_path)
        plt.close()

        print(f"IC（Pearson）图表已保存至: {ic_image_path}")

        # 保存 Rank IC（Spearman）数据到 CSV 文件
        rank_ic_values_df = pd.DataFrame(rank_ic_values)
        rank_ic_values_df.to_csv(os.path.join(result_folder, 'Rank_IC_Values.csv'), index=False)

        # 绘制 Rank IC（Spearman）曲线
        plt.figure(figsize=(10, 6))
        for run in rank_ic_values:
            plt.plot(rank_ic_values[run], label=f'Run {run:02d}')

        plt.title('Rank IC (Spearman) by Training Run')
        plt.xlabel('Rebalance Date')
        plt.ylabel('Rank IC (Spearman)')
        plt.legend()
        plt.grid(True)

        # 保存 Rank IC（Spearman）图表为 PNG 文件
        rank_ic_image_path = os.path.join(result_folder, 'Rank_IC_Spearman.png')
        plt.savefig(rank_ic_image_path)
        plt.close()

        print(f"Rank IC（Spearman）图表已保存至: {rank_ic_image_path}")

        # 保存 ICIR 数据到 CSV 文件
        icir_df = pd.DataFrame(list(icir_values.items()), columns=['Run', 'ICIR'])
        icir_df.to_csv(os.path.join(result_folder, 'ICIR.csv'), index=False)

        print(f"ICIR 结果已保存至: {os.path.join(result_folder, 'ICIR.csv')}")

# 运行回测
if __name__ == "__main__":
    output_folder = 'Output_v2'
    label_file = 'label_monthly_return.csv'

    backtest = BackTest(output_folder, label_file)
    backtest.run()