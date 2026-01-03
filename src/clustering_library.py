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
    
    def calculate_rfm(self):
        snapshot_date=self.df['InvoiceDate'].max()+pd.Timedelta(days=1)
        recency= self.df.groupby('CustomerID').agg({'InvoiceDate': lambda x : (snapshot_date-x.max()).days}).rename(columns={'InvoiceDate':'Recency'})
        frequency=self.df.groupby('CustomerID').agg({'InvoiceNo': lambda x : x.nunique()}).rename(columns={'InvoiceNo':'Frequency'})
        monetary=self.df.groupby('CustomerID').agg({'TotalPrice': lambda x : x.sum()}).rename(columns={'TotalPrice':'Monetary'})
        rfm=pd.concat([recency,frequency,monetary],axis=1)
        return rfm
        
        
class DataVisualizer:
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
        plt.bar(monthly.index.astype(str),monthly.values,color='darkblue')
        plt.xlabel('Thang')
        plt.ylabel('Doanh thu (GBP)')
        plt.title('Doanh thu hang thang')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
            
    def plot_shopping_heatmap(self):
        pivot_table=self.df.pivot_table(index='DayOfWeek',columns='Hour',values='InvoiceNo',aggfunc='count')
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot_table,cmap='viridis',fmt='g')
        plt.xlabel('Gio trong ngay')
        plt.ylabel('Ngay trong tuan (0=Thu 2, 6=Chu Nhat)')
        plt.title('Hoat dong mua hang theo ngay va gio')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
            
    def plot_top_prodcucts(self):
        top_products=self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        top_products=top_products.sort_values(ascending=True)
        plt.figure(figsize=(12,6))
        plt.barh(top_products.index,top_products.values)
        plt.title('Top 10 san pham theo so luong ban')
        plt.xlabel('So luong ban')
        plt.ylabel('Description')
        plt.tight_layout()
        plt.show()
            
    def plot_top_price(self):
        top_price=self.df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        top_price=top_price.sort_values(ascending=True)
        plt.figure(figsize=(12,6))
        plt.barh(top_price.index,top_price.values)
        plt.title('Top 10 san pham theo doanh thu')
        plt.xlabel('Doanh thu (GBP)')
        plt.ylabel('Description')
        plt.tight_layout()
        plt.show()
        
    def plot_rfm_analysis(self,rfm):
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        
        sns.histplot(rfm['Recency'],kde=True,ax=ax[0],bins=50)
        ax[0].set_title('Phan pho Recency')
        ax[0].set_xlabel('Ngay')
        
        sns.histplot(rfm[rfm['Frequency'] < 1400]['Frequency'],kde=True,ax=ax[1],bins=50)
        ax[1].set_title('Phan pho Frequency')
        ax[1].set_xlabel('So giao dich')
        
        sns.histplot(rfm[rfm['Monetary'] < 17500]['Monetary'],kde=True,ax=ax[2],bins=50)
        ax[2].set_title('Phan pho Monetary')
        ax[2].set_xlabel('Tong chi tieu')
        
        plt.tight_layout()
        plt.show()
        