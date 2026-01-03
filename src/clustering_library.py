import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataCleaner:
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=None
        
    def load_data(self):
        self.df=pd.read_csv(self.file_path,encoding='unicode_escape',parse_dates=['InvoiceDate'])
        return self.df
    
    def clean_data(self):
        self.df['TotalPrice']=self.df['Quantity']*self.df['UnitPrice']
        
        drop_cancelled=self.df['InvoiceNo'].astype(str).str.startswith('C')
        self.df=self.df[~drop_cancelled]
        
        self.df=self.df[self.df['Country']=='United Kingdom']
        
        self.df=self.df.dropna(subset=['CustomerID'])
        
        self.df=self.df[self.df['TotalPrice'] > 0.00]
        return self.df
    
    def create_time_features(self):
        self.df['Hour']=self.df['InvoiceDate'].dt.hour
        self.df['DayOfWeek']=self.df['InvoiceDate'].dt.dayofweek
        self.df['Date']=self.df['InvoiceDate'].dt.date
        self.df['Month']=self.df['InvoiceDate'].dt.to_period('M')
        return self.df
        
    class DataVisualize:
        def __init__(self,df):
            self.df=df
        
        def plot_over_date(self):
            dately=self.df.groupby('Date')['TotalPrice'].sum()
            plt.figure(figsize=(12, 6))
            plt.plot(dately.index,dately.values,color='darkblue')
            plt.xlabel('Ngay')
            plt.ylabel('Doanh thu (GBP)')
            plt.title('Doanh thu hang ngay')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            
        def plot_over_month(self):
            monthly=self.df.groupby('Month')['TotalPrice'].sum()
            plt.figure(figsize=(12, 6))
            plt.bar(monthly.index,monthly.values,color='darkblue')
            plt.xlabel('Thang')
            plt.ylabel('Doanh thu (GBP)')
            plt.title('Doanh thu hang thang')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.show()
            
        def plot_shopping_heatmap(self):
            pivot_table=self.df.pivot_table(index='DayOfWeek',columns='Hour',values='InvoiceNo',aggfunc='count')
            plt.figure(figsize=(12,6))
            sns.heatmap(pivot_table,cmap='viridis',fmt='g')
            plt.xlabel('Gio trong ngay')
            plt.ylabel('Ngay trong tuan (0=Thu 2, 6=Chu Nhat)')
            plt.title('Hoat dong mua hang theo ngay va gio')
            plt.tight_layout()
            plt.show()
            