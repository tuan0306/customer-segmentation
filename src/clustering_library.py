import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
        

class FeatureEngineer:
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=None
        self.customer_features=None
        self.customer_features_transformed=None
        self.customer_features_scaled=None
        self.scaler = StandardScaler()
        
    def load_data(self):
        self.df=pd.read_csv(self.file_path,encoding='unicode_escape',parse_dates=['InvoiceDate','Date','Month'])
        return self.df
    
    def create_customer_features(self):
        volumn_value=self.df.groupby('CustomerID').agg({
            'Quantity':'sum',
            'UnitPrice':'mean',
            'TotalPrice':['mean','sum'],
            'InvoiceNo':'nunique',
            'StockCode':'nunique'
        })
    
        volumn_value.columns=[
            'Sum_Quantity',
            'Mean_UnitPrice',
            'Mean_TotalPrice',
            'Sum_TotalPrice',
            'Count_Invoice',
            'Count_Stock'
        ]
        
        transaction_behaviour_helper=self.df.groupby(['CustomerID','InvoiceNo']).agg({
            'StockCode':'nunique',
            'UnitPrice':'mean',
            'Quantity':'sum',
            'TotalPrice':['mean','sum']
        })
        
        transaction_behaviour_helper.columns=[
            'Mean_StockCountPerInvoice',
            'Mean_UnitPriceMeanPerInvoice',
            'Mean_QuantitySumPerInvoice',
            'Mean_TotalPriceMeanPerInvoice',
            'Mean_TotalPriceSumPerInvoice'
        ]
        
        transaction_behaviour=transaction_behaviour_helper.groupby('CustomerID').agg({
            'Mean_StockCountPerInvoice':'mean',
            'Mean_UnitPriceMeanPerInvoice':'mean',
            'Mean_QuantitySumPerInvoice':'mean',
            'Mean_TotalPriceMeanPerInvoice':'mean',
            'Mean_TotalPriceSumPerInvoice':'mean'
        })
        
        
        product_prefer_helper=self.df.groupby(['CustomerID','StockCode']).agg({
            'InvoiceNo':'nunique',
            'UnitPrice':'mean',
            'Quantity':'sum',
            'TotalPrice':['mean','sum']
        })
        
        product_prefer_helper.columns=[
            'Mean_InvoiceCountPerStock',
            'Mean_UnitPriceMeanPerStock',
            'Mean_QuantitySumPerStock',
            'Mean_TotalPriceMeanPerStock',
            'Mean_TotalPriceSumPerStack'
        ]
        
        product_prefer=product_prefer_helper.groupby('CustomerID').agg({
            'Mean_InvoiceCountPerStock':'mean',
            'Mean_UnitPriceMeanPerStock':'mean',
            'Mean_QuantitySumPerStock':'mean',
            'Mean_TotalPriceMeanPerStock':'mean',
            'Mean_TotalPriceSumPerStack':'mean'
        })
        
        self.customer_features=pd.concat([volumn_value,transaction_behaviour,product_prefer],axis=1)
        self.customer_features=self.customer_features.reset_index()
        return self.customer_features
    
    def transform_features(self):
        customer_features_indexed=self.customer_features.set_index('CustomerID')
        feature_values=customer_features_indexed.values+1
        self.customer_features_transformed=customer_features_indexed.copy()
        for i,feature in enumerate(customer_features_indexed.columns):
            transformed,lambda_param=stats.boxcox(feature_values[:,i])
            self.customer_features_transformed.iloc[:,i]=transformed
        return self.customer_features_transformed
    
    def plot_features_histograms(self,transform=False):
        if (transform):
            data_to_plot=self.customer_features_transformed
        else:
            data_to_plot=self.customer_features.set_index('CustomerID')

        color='darkred'
        n_rows=4
        n_cols=4
        fig,axes=plt.subplots(n_rows,n_cols,figsize=(20,16))
        axes=axes.flatten()
        
        for i,cols in enumerate(data_to_plot.columns):
            sns.histplot(data_to_plot[cols],kde=False,ax=axes[i],color=color,bins=30)
            axes[i].set_title(cols)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Tan suat')
        
        plt.tight_layout()
        plt.show()
        
    def scale_features(self):
        scaled_data=self.scaler.fit_transform(self.customer_features_transformed)
        self.customer_features_scaled=pd.DataFrame(
            scaled_data,
            index=self.customer_features_transformed.index,
            columns=self.customer_features_transformed.columns)
        
        return self.customer_features_scaled
    
    def save_featues(self,file_path="../data/processed"):
        os.makedirs(file_path,exist_ok=True)
        
        customer_features_indexed=self.customer_features.set_index('CustomerID')
        customer_features_indexed.to_csv(f"{file_path}/customer_features.csv")
        
        self.customer_features_transformed.to_csv(f"{file_path}/customer_features_transformed.csv")
        
        self.customer_features_scaled.to_csv(f"{file_path}/customer_features_scaled.csv")
        
        
class ClusterAnalyzer:
    def __init__(self,df_scaled_path,df_original_path):
        self.df_scaled_path=df_scaled_path
        self.df_original_path=df_original_path
        self.df_scaled=None
        self.df_original=None
        self.models={}
        self.labels={}
        self.pca_data=None
        self.pca=None
        self.cluster_results={}
        
    def load_data(self):
        self.df_scaled=pd.read_csv(self.df_scaled_path,index_col=0)
        self.df_original=pd.read_csv(self.df_original_path,index_col=0)
        
    def determine_optimal_k(self,k_range=range(2,11)):
        inertia=[]
        silhouettes=[]
        
        for k in k_range:
            model=KMeans(n_clusters=k,random_state=42,n_init=10)
            model.fit(self.df_scaled)
            inertia.append(model.inertia_)
            silhouettes.append(silhouette_score(self.df_scaled,model.labels_))
        
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        
        axes[0].plot(k_range,inertia,marker='o')
        axes[0].set_title('Phuong phap Elbow')
        axes[0].set_xlabel('So luong clusters(k)')
        axes[0].set_xticks(k_range)
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True)
        
        axes[1].plot(k_range,silhouettes,marker='o')
        axes[1].set_title('Phuong phap Silhouette score')
        axes[1].set_xlabel('So luong clusters(k)')
        axes[1].set_xticks(k_range)
        axes[1].set_ylabel('Silhouette score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def apply_pca(self,n_component=3):
        self.pca=PCA(n_components=n_component)
        pca_dat=self.pca.fit_transform(self.df_scaled)
        cols=[f"PC{i+1}" for i in range(pca_dat.shape[1])]
        self.pca_data=pd.DataFrame(pca_dat,index=self.df_scaled.index,columns=cols)
        return self.pca_data
    
    def plot_pca_variance(self):
        ratios=self.pca.explained_variance_ratio_
        indies=np.arange(len(ratios))+1
        plt.figure(figsize=(10,6))
        ratios_cumsum=np.cumsum(ratios)
        
        plt.bar(indies,ratios,color='g',label='Phuong sai tung thanh phan')
        
        plt.plot(indies,ratios_cumsum,marker='o',color='r',linestyle='-',linewidth=2,label='Tong hop phuon sai tich luy')
        
        plt.title('Phan tich phuong sai cua PCA', fontsize=15)
        plt.xlabel('Thanh phan chinh')
        plt.ylabel('Ty le phuong sai')
        plt.xticks(indies, [f"PC{i}" for i in indies])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(ratios):
            plt.text(i+1, v + 0.01, f'{v*100:.1f}%', ha='center', color='green', fontweight='bold')
        plt.show()
        
    def apply_kmeans(self,k_list=[3,4]):
        for k in k_list:
            model=KMeans(n_clusters=k,random_state=42,n_init=10)
            label=model.fit_predict(self.df_scaled)
            
            df_result=self.df_original.copy()
            
            df_result['Cluster']=label
            self.models[k]=model
            self.labels[k]=label
            
            self.cluster_results[k]=df_result
        return self.cluster_results
    
    def plot_clusters_pca(self,k_list=[3,4]):
        fig,axes=plt.subplots(len(k_list),1,figsize=(10,10))
        
        for i,k in enumerate(k_list):
            x=self.pca_data['PC1']
            y=self.pca_data['PC2']
            label=self.labels[k]
            scatter=axes[i].scatter(x,y,c=label,cmap='viridis',s=50,alpha=0.6)
            axes[i].set_title(f'Phan cum KMeans (k={k})')
            axes[i].set_xlabel('PC1')
            axes[i].set_ylabel('PC2')
            plt.colorbar(scatter,ax=axes[i],label='Cluster')
        
        plt.tight_layout()
        plt.show()
        
    def plot_clusters_pca_3d(self,k_list=[3,4]):
        fig=plt.figure(figsize=(10,10))
        
        for i,k in enumerate(k_list):
            x=self.pca_data['PC1']
            y=self.pca_data['PC2']
            z=self.pca_data['PC3']
            label=self.labels[k]
            ax=fig.add_subplot(len(k_list),1,i+1,projection='3d')
            scatter=ax.scatter(x,y,z,c=label,cmap='viridis',s=50,alpha=0.6)
            ax.set_title(f'Phan cum KMeans (k={k})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.colorbar(scatter,ax=ax,label='Cluster')
        
        plt.tight_layout()
        plt.show()
            
        
    def analyze_pca_meaning(self):
        loadings=pd.DataFrame(
            self.pca.components_.T,
            columns=['PC1','PC2','PC3'],
            index=self.df_scaled.columns
        )
        
        plt.figure(figsize=(10,10))
        sns.heatmap(loadings,annot=True,cmap='RdBu_r',center=0,fmt='.2f',linewidths=2)
        plt.title('Muc do dong gop cua cac feature goc vao PC (PCA Loadings)', fontsize=14)
        plt.ylabel('Cac dac trung goc (Original Features)')
        plt.xlabel('Cac thanh phan chinh (Principal Components)')
        plt.tight_layout()
        plt.show()
        
    def save_clusters(self,file_path='../data/processed'):
        os.makedirs(file_path,exist_ok=True)
        
        for k,df_result in self.cluster_results.items():
            output_cluster=df_result[['Cluster']].copy()
            output_cluster=output_cluster.reset_index()
            output_cluster=output_cluster.sort_values(by=['Cluster','CustomerID'])
            output_cluster.to_csv(f'{file_path}/customer_clusters_{k}.csv',index=False)
        