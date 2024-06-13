from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import copy
import os
warnings.filterwarnings("ignore")
from sys import getsizeof

from factors import Factor

# get tradedates, zz500, target_stock, total_return, total_price

class XSMOM:
    def __init__(self, total_return, zz500, train_window=20,test_window=5):
        self.total_return = total_return
        self.zz500 = zz500['涨跌幅']
        self.train_window = train_window
        self.test_window = test_window
        self.trade_dates = zz500.index

    def get_window(
            self,
            data_dir: str,
            trade_dates: np.ndarray,
            time_window: int,
            current_index: int = 0,
            how:str = 'inner'
        ) -> pd.DataFrame:
        """
        DEPRECATED
        从硬盘中整合eps的数据，读入内存。
        输入参数为交易起始日，和时间窗口长度，
        输出参数为一个pandas的df
        
        这个函数只在第一次调用的时候用到，读出的表格已经写入新的CSV文件，
        之后的硬盘读取不再需要这个函数，而是直接使用read_csv()方法来完成。
        """
        # FIXME: change the way to read data
        file_name = data_dir + 'eps_{}.csv'.format(trade_dates[current_index])
        df = pd.read_csv(file_name, index_col=1).loc[:]['eps']

        for i in range(1, time_window):

            file_name = data_dir + 'eps_{}.csv'.format(trade_dates[current_index + i])
            new_df = pd.read_csv(file_name, index_col=1).iloc[:]['eps']
            df = pd.concat([df, new_df], join=how, axis=1)  # 取交集

        df.columns = trade_dates[current_index:current_index + time_window]
        
        return df
    
    def get_frame(
            self,
            data:pd.DataFrame,
            time_window:int,
            index_num:int,
            direction:str
        ) -> pd.DataFrame:
        """
        从内存中的数据框读取合适大小的数据用于计算
        """
        
        index_num += 1
        if direction == 'forward':
            df = data.iloc[:,index_num : index_num + time_window].dropna()
        elif direction == 'backward':
            df = data.iloc[:,index_num - time_window: index_num].dropna()
            
        return df
    
    def cal_strategy(
            self,
            period, 
            step, 
            start_day_index, 
            skip_first_day,
            get_Xt, #关键的获得信号的函数
            note,
        )-> tuple:
        """
        将权重函数（信号）和未来的五日收益率相乘的到策略的未来收益率
        ## 注意，这里的列名是指利用这一天及以前（含这一天）的数据对未来收益率做的预测。
        """
        

        daily_return = copy.deepcopy({'all':[],'top':[],'bottom':[]})
        daily_signal = copy.deepcopy({'all':[],'top':[],'bottom':[]})
        # 遍历我们要计算的每一天
        for i in range(0, period, step): 
            
            current_day_index =  start_day_index + i # 找到现在的交易日在交易日列表中的索引
            current_day = self.trade_dates[current_day_index]

            # 获取用于计算的数据框
            
    #         df_price = get_frame(data=total_price,
    #                              time_window=train_window,
    #                              index_num = current_day_index,
    #                              direction='backward') 
            
            df_return = self.get_frame(data=self.total_return,
                                time_window=self.train_window, 
                                index_num = current_day_index,
                                direction='backward') 
            
            #  将数据框传到计算信号的函数当中。注意生成交易信号的时候就要判断是否多空
            X = df_return.apply(get_Xt, axis=1).sort_values(ascending=False)
        
    #         取与股票池的交集
            # current_target_stocks = target_stock.loc[:][current_day].dropna()
            # current_target_stocks = current_target_stocks.astype(int)
            # X.index = X.index.astype(int)
            # intersec_stock = pd.Series(list(set(X.index).intersection(set(current_target_stocks.values))))
            # X = X.loc[intersec_stock][:]
            # df_return=df_return.loc[intersec_stock][:]
            
            X = X.astype(float)
            
            X = (X - np.mean(X)) / np.std(X)  # 做 z-score
            X[X > 3] = 3
            X[X < -3]  = -3

            top =  X[X > 0].index
            bottom = X[X < 0].index

    #         # 中位数分组
    #         median = X.median()
    #         top =  X[X > median].index
    #         bottom = X[X < median].index

            df_X = pd.DataFrame(
                np.zeros((len(df_return.index),3)), # 先生成全 0 矩阵，方便后面的加法
                columns=['all','top','bottom'], 
                index=df_return.index
            )

            df_X.loc[top, 'top'] = X[top] / X[top].sum()
            df_X.loc[bottom ,'bottom'] = - X[bottom] / X[bottom].sum()
            df_X['all'] = df_X['top'] + df_X['bottom']
            df_X.replace(0, np.nan, inplace=True)
            X = df_X

            if skip_first_day is True: # 如果要跳过第一天的话
                current_day_index += 1
            
            # 获取用于计算未来收益率的数据框
            test_return = self.get_frame(data=self.total_return,
                                time_window=self.test_window,
                                index_num=current_day_index,
                                direction='forward')
            
            # 计算今日的收益率
            for column in X.columns:
                daily_signal[column].append(X[column])
                daily_return[column].append((test_return.sum(axis=1) * X[column]).dropna())
        
        # 将列表中的Series统一拼接
        strategy_return = copy.deepcopy({})
        strategy_signal = copy.deepcopy({})
        for key in daily_return.keys():
            strategy_return[key] = pd.concat(daily_return[key], axis=1, join='outer')
            strategy_signal[key] = pd.concat(daily_signal[key], axis=1, join='outer')
            
            strategy_return[key].columns= self.trade_dates[range(start_day_index,start_day_index + period, step)]
            strategy_signal[key].columns= self.trade_dates[range(start_day_index,start_day_index + period, step)]
            
        strategy_return['all'] = (strategy_return['top'].replace(np.nan,0)-strategy_return['bottom'].replace(np.nan,0)).replace(0,np.nan)
        strategy_return['bottom'] = -strategy_return['bottom']
        if note:
            for key in daily_return.keys():
                strategy_return[key].to_csv(f'./data/strategy_return_{key}/' + note +'.csv',encoding='gbk',index=True)
                strategy_signal[key].to_csv(f'./data/strategy_signal_{key}/' + note +'.csv',encoding='gbk',index=True)
                header = not os.path.isfile('XSMOM_return.csv')
                daily_return = pd.DataFrame(strategy_return[key].sum(),columns=[note,]).T
                daily_return.to_csv('XSMOM_return.csv', header=header,index=True,encoding='gbk',mode='a')
        
        return strategy_return, strategy_signal

    def show_strategy(
            self,
            strategy_return:dict,
            strategy_signal:dict,
            bench_mark:pd.Series,
            long_short: str,
            step: int = 5, 
            note: str = '', 
        ):
        """
        展示策略收益的函数，具有普遍适用性。
        这个函数需要绘制一些图片、计算一些数据，并存储到CSV文件当中
        具体而言，需要计算的比例有：
        （全策略、纯空头、纯多头的）胜率、回测收益、
        """
        daily_return = copy.deepcopy({})
        backtest_params = copy.deepcopy({})
        cumulative_return = copy.deepcopy({})
        drawdowns = copy.deepcopy({})
        tov_seq = copy.deepcopy({})
        
        for key in strategy_return.keys(): # 遍历每种组合的情况
            
            # 计算组合的日收益率、累计收益率序列
            daily_return[key] = strategy_return[key].sum()

            
            cumulative_return[key] = daily_return[key].cumsum()


            # 计算日均收益率、日均波动率、IR、下行偏差DD
            backtest_params[key + '_' + '日均收益率R'] = daily_return[key].mean() / step
            backtest_params[key + '_' + '日均波动率Vol'] = daily_return[key].std() / np.sqrt(step)
            
            backtest_params[key + '_' + '年化收益率ER'] = backtest_params[key + '_' + '日均收益率R'] * 252
            backtest_params[key + '_' + '年化波动率VOL'] = backtest_params[key + '_' + '日均波动率Vol'] * np.sqrt(252)
            
            backtest_params[key + '_' + '信息比率IR'] = backtest_params[key + '_' + '日均收益率R'] / backtest_params[key + '_' + '日均波动率Vol']
            backtest_params[key + '_' + '夏普比率SR'] = backtest_params[key + '_' + '信息比率IR'] * np.sqrt(252)

            backtest_params[key + '_' + '下行偏差DD'] = daily_return[key][daily_return[key] < 0].std()
            
            # 计算胜率：取每日胜率的平均值
            ve_seq = np.count_nonzero(strategy_return[key] > 0, axis = 0) / strategy_return[key].notna().sum(axis=0)
            backtest_params[key + '_' + '胜率VE'] = ve_seq.mean()
        
            # 计算盈亏比：取每日盈亏比的平均值
            gain = strategy_return[key][strategy_return[key] > 0].mean(axis=0)
            loss = strategy_return[key][strategy_return[key] < 0].mean(axis=0)
            backtest_params[key + '_' + '盈亏比PnL'] = (gain / abs(loss)).mean()
            
            # 计算最大回撤
            max_so_far = cumulative_return[key].values[0]
            drawdowns[key] = []
            for trade_day in cumulative_return[key].index:
                if cumulative_return[key][trade_day] > max_so_far:
                    drawdown = 0
                    drawdowns[key].append(drawdown)
                    max_so_far = cumulative_return[key][trade_day]
                else:
                    drawdown =  max_so_far - cumulative_return[key][trade_day]
                    drawdowns[key].append(drawdown)
                
            
            backtest_params[key + '_' + '最大回撤MDD'] = max(drawdowns[key])
            
            # 计算Calmar比率、Sortino比率
            backtest_params[key + '_' + '卡玛比率Calmar'] = backtest_params[key + '_' + '年化收益率ER'] / backtest_params[key + '_' + '最大回撤MDD']
            backtest_params[key + '_' + '索提诺比率Sortino'] = backtest_params[key + '_' + '年化收益率ER'] / backtest_params[key + '_' + '下行偏差DD']
            
            # 计算换手率
    #         daily_holding = strategy_signal[key] / strategy_signal[key].notna().sum(axis=0)# 先将交易信号转化为持仓数
            daily_holding = strategy_signal[key]
            prior = daily_holding.iloc[:, :daily_holding.shape[1] - 1].fillna(0)
            rear = daily_holding.iloc[:,1:daily_holding.shape[1]].fillna(0)
            
            prior.columns, rear.columns = range(daily_holding.shape[1] - 1),range(daily_holding.shape[1] - 1) # 为了df可以准确做减法，需要修改对齐列名
            
            tov_seq[key] = (rear - prior).abs().sum(axis=0) / 2
            tov_seq[key] = pd.concat([pd.Series(.5), tov_seq[key]])
            backtest_params[key + '_' + '换手率Tov'] = tov_seq[key].mean()
            
        df = pd.DataFrame.from_dict(backtest_params,orient='index')

        
        # 修改显示方式
        names = np.array(df.index).reshape((-1,3),order='F').reshape((1,-1),order='C').tolist()[0]
        df = df.loc[names,:]

        self.plot_strategy_performance(cumulative_return, drawdowns, tov_seq, bench_mark, note)
        
        # 写入CSV文件。
        if note is None: 
            return
        df_log = df.T
        
        df_log.index = [note,]
        header = not os.path.isfile('XSMOM_report.csv') # 如果文件存在，则不要写入表头。如果文件不存在则写入表头
        df_log.to_csv('XSMOM_report.csv', mode='a', index=True, header=header, encoding='gbk')
        
        return 
    
    def plot_strategy_performance(self, cumulative_return, drawdowns, tov_seq, bench_mark, note):
        x = pd.to_datetime(cumulative_return['all'].index, format='%Y%m%d')
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        plt.figure(figsize=(16, 5))

        plt.plot(x, cumulative_return['all'] * 100, color='r')
        plt.plot(x, cumulative_return['top'] * 100, color='g')
        plt.plot(x, cumulative_return['bottom'] * 100, color='b')

        plt.plot(x, bench_mark * 100, color='k')

        plt.legend(['Strategy_all', 'Strategy_top', 'Strategy_bottom', 'Benchmark'])
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return %')
        plt.title('CUMULATIVE RETURN')
        if note is not None:
            note = time_str + '_' + note
            plt.savefig('./XSMOM_image/' + note + '.png')  # 保存图片

        plt.figure(figsize=(16, 6))
        plt.subplot(211)
        plt.bar(x, -np.array(drawdowns['all']) * 100,)
        plt.ylabel('Max Drawdown %')
        plt.title('DRAWDOWN')

        plt.subplot(212)
        tov_show = tov_seq['all']
        tov_show[0] = 0
        plt.bar(x, tov_show * 100)

        plt.ylabel('Turnover Rate %')
        plt.xlabel('Date')
        plt.title('TURNOVER RATE')

        plt.show()

    def run(
            self,
            get_Xt, # 这是需要的输出信号的名称
            start_day:int = 20200101,
            end_day:int = 20231231,
            skip_first_day:bool = True,
            step:int = 5,
            long_short: str = 'all', # 限制字段，可选'none','short','long','all',默认为'all'，这个字段仅用作画图。而所有的数据都会计算
            note: str = '', # 记录调参细节等内容的字段，将写入最后的日志文件
            read: bool = False,
        ):
        """
        回测的主函数。
        需要实现的功能有：
        1. 初始化
        2. 运行回测收益并存储数据
        3. 计算回测结果、相应收益指标等元素，并存储到csv文件中
        """

        # 初始化部分

        start_day = self.find_trade_day(start_day, 'forward')
        start_day_index = int(np.where(self.trade_dates == start_day)[0])
        end_day = self.find_trade_day(end_day, 'backward')
        end_day_index = int(np.where(self.trade_dates == end_day)[0])
        period = end_day_index - start_day_index

        bench_mark = self.zz500.iloc[start_day_index:end_day_index].cumsum().iloc[range(0, period, step)]
                
        strategy_return, strategy_signal = self.cal_strategy(period, step, start_day_index, skip_first_day, get_Xt, note) 

        self.show_strategy(strategy_return, strategy_signal, bench_mark, long_short, step, note)

        return 0

    @staticmethod
    def z_score(X: pd.DataFrame)->pd.DataFrame:
        return (X - np.mean(X)) / np.std(X)
    

    def find_trade_day(self, day:int, direction:str = 'backward'):
        """
        将输入的随便的数转化成列表中有的交易日
        """
        day = pd.to_datetime(str(day), format='%Y%m%d')

        if day in self.trade_dates: # 如果本来就在列表当中
            nearest_trade_day = day
        elif direction == 'backward':
            nearest_trade_day = self.trade_dates[
                max(
                    (0, np.argmax(self.trade_dates > day) - 1)
                )
            ]
        elif direction == 'forward':
            nearest_trade_day = self.trade_dates[
                np.argmax(self.trade_dates > day)
            ]
            
        return nearest_trade_day