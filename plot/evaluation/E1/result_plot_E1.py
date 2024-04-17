# %%
import pandas as pd 
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np
# 设置字体大小、坐标轴标签大小、保存图像的分辨率、图像大小、线条宽度、图例字体大小和间距等参数
sns.set(rc={"font.size":14,"axes.labelsize":14,"xtick.labelsize":14,"ytick.labelsize":14,"savefig.dpi":200,"figure.figsize":(6, 4),"lines.linewidth": 2, "legend.fontsize": 13,"legend.borderpad":0.2, "legend.labelspacing":0.5, "legend.columnspacing":0.5})
# 设置绘图样式为白色背景
sns.set_style("white")

# 定义一组颜色列表
# colors = ["#426A5A","#97bb73","#CDE77F","#EDA758","#D58816"]
# colors = ["#426A5A","#97bb73","#86897F","#EDA758","#D58816"]
# colors = ["#D87532", "#CDE77F","#97bb73", "#426A5A","#D58816"]
# colors =  ["#E4693C","#C4C472","#1FBB97","#0C7A78"]
colors =  ["#3b6291","#bf7334","#0C7A78"]
# colors =  ["#8dbba5","#8dbba5","#6abb62","#0C7A78"]



# %%
#加载数据
# 从CSV文件加载数据
def caculate_single_avg_latency(data_path):

    data = pd.read_csv(data_path)
    # 创建数据副本
    data_copy = data.copy()
    data_copy['Latency'] = (data_copy['ResponseTime'] - data_copy['RequestTime'])/1e9
    avg_latency = np.mean(data_copy['Latency'])
    
    return avg_latency

# %%
base_path = '../../../evaluation/metrics/wula/E1_final/'

def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files
# 
all_data = pd.DataFrame()
for file in get_csv_files_in_bottom_directory(base_path):    
    df = pd.read_csv(file)    
    df['model_name'] = file.split("/")[-1].split(".")[0]  
    # df['system'] = file.split("/")[-1].split("\\")[0]
    df['system'] = file.split("/")[-4]
    
    all_data = pd.concat([all_data, df], ignore_index=True)

all_data

# %%
all_data['system'] = all_data['system'].replace('roger','Roger')
all_data['system'] = all_data['system'].replace('cocktail','Cocktail')
all_data['system'] = all_data['system'].replace('inferline','InferLine')
filtered_df = all_data.copy()
filtered_df['response_latency'] = filtered_df['ResponseTime'] - filtered_df['RequestTime']
latency_df = filtered_df.groupby(['system','model_name'])['response_latency'].mean().reset_index()
latency_df['response_latency'] = latency_df['response_latency']/1e9

latency_df

# %%
# latency_df_gp
# speed_up of each model_name, compare Roger to systems
speed_up_df = latency_df.copy()
speed_up_df['speed_up'] = speed_up_df.groupby('model_name')['response_latency'].transform(lambda x: (x-x.min())/x)

speed_up_df.to_csv('speed_up.csv',index=False)


# %%
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()
order = ['OPT-66B', 'BERT-21B', 'LLAMA-7B', 'BERT-QA', 'Resnet-152', 'Resnet-50']
systems = ['InferLine','Cocktail','Roger']
# 使用Seaborn绘制柱状图
sns.barplot(x="model_name", y="response_latency", hue="system", data=latency_df, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,hue_order=systems,order=order)

# 设置x轴和y轴的标签
ax.set_xlabel("Model")
ax.set_ylabel("Avg Latency (s)")

# set y ticks font size
ax.tick_params(axis='x', labelsize=11)
ax.set_ylim(0, 3.6)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=1, frameon=True)

# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
# ax.set_ylim(bottom=0.5)
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
ax.set_ylim(0, 3.3)
#add hatch
num_systems = len(systems)
for i in range(num_systems):
    for j in range(len(order)):
        if i==2:
            ax.patches[(i * len(order)) + j].set_hatch('***')

# 显示bar 数据
# 给图例添加hatch

for p in ax.patches:
    if p.get_height() > 0 :
       ax.annotate('%.2f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E1-avg_latency.pdf", bbox_inches='tight')
plt.show()

# %%
all_data

# %%
merged_data = all_data.copy()
merged_data['Latency'] = (merged_data['ResponseTime'] - merged_data['RequestTime'])/1e9
merged_data


# %%
import numpy as np
import pandas as pd

systems = ['InferLine', 'Cocktail', 'Roger']
percentiles = [55, 65, 75, 85, 95, 97, 98, 99]

latency_data = []

for system in systems:
    all_data = merged_data.loc[merged_data['system'] == system, 'Latency'].tolist()
    latency_percentiles = np.percentile(all_data, percentiles)
    latency_data.append({'system': system,
                         '55th': latency_percentiles[0],
                         '65th': latency_percentiles[1],
                         '75th': latency_percentiles[2],
                         '85th': latency_percentiles[3],
                         '95th': latency_percentiles[4],
                         '97th': latency_percentiles[5],
                         '98th': latency_percentiles[6],
                         '99th': latency_percentiles[7]})

latency_df = pd.DataFrame(latency_data)

latency_df

# %%
melted_df = pd.melt(latency_df, id_vars=['system'], value_vars=['55th', '65th', '75th', '85th', '95th', '97th', '98th', '99th'], var_name='metric', value_name='value')
melted_df

# %%
# from turtle import color


fig, ax = plt.subplots()
# systems = ['InferLine','Cocktail','Roger']

# colors_line =["#3b6291","#bf7334","#007f54"]
colors=['#4a4c7e','#33977c','#80c161']
for i, system in enumerate(systems):
    # linestyle = '-' if i != 2 else '--'
    linestyle = '--'
    system_data = melted_df[melted_df['system'] == system]
    sns.lineplot(x="metric", y="value", data=system_data, 
                 linewidth=2.5, ax=ax, color=colors[i] ,label=system, linestyle=linestyle, marker='h', markersize=10, markeredgewidth=1.5)


ax.set_xlabel('Quantile')
ax.set_ylabel("Latency (s)")

ax.grid(axis="y", linestyle='--', alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,  ncol=1, framealpha=0.6)
plt.tight_layout()
plt.savefig("E1-Quantile-Latency.pdf", bbox_inches='tight')
plt.show()

# %%
def merge_data(folder_path):
    if not os.path.isdir(folder_path):
        print("Folder path does not exist.")
        return None
    
    # Initialize a list to hold data from all CSV files
    all_data = []
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append DataFrame to the list
            all_data.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    merged_data = pd.concat(all_data, ignore_index=True)
    

    merged_data['latency'] = (merged_data['ResponseTime'] - merged_data['RequestTime'])/1e9
   
    
    return merged_data['latency']

# %%
#fault count
filtered_df

# %%
latency_df

# %%
filtered_df_500 = filtered_df[filtered_df['StatusCode'] == 500]
# 按照 system 和 model_name 分组，并计算每组的行数
fault_counts = filtered_df_500.groupby(['system', 'model_name'])['StatusCode'].count().reset_index(name='fault_count')

fault_counts


# %%
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()
# 使用Seaborn绘制柱状图

sns.barplot(x="model_name", y="fault_count", hue="system", data=fault_counts, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0)

# 设置x轴和y轴的标签
ax.set_xlabel("Model")
ax.set_ylabel("Fault Number")

# set y ticks font size
ax.tick_params(axis='x', labelsize=10)
ax.set_ylim(0,22000)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=1, frameon=True)
# plt.legend(title='Pipeline Length',loc='upper center', ncol=3)
# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
# ax.set_ylim(bottom=0.5)
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch

# 显示bar 数据 # if bar value == 0, the show "No Fault"
for p in ax.patches:
    if p.get_height() == 0 :
        ax.annotate('No Fault', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color='black', xytext=(0,15), textcoords='offset points', rotation=90)    
    ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color='black', xytext=(0,15), textcoords='offset points', rotation=90)



# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E1-fault-count.pdf", bbox_inches='tight')
plt.show()

# %%
#goodput rate
goodput_df = filtered_df[filtered_df['StatusCode'] == 200].copy()
goodput_df = goodput_df.groupby(['system','model_name'])['StatusCode'].count().reset_index().rename(columns={'StatusCode':'goodput'})
total_requests = filtered_df.groupby(['system', 'model_name'])['StatusCode'].count().reset_index().rename(columns={'StatusCode': 'total_num'})
# 合并两个 DataFrame，将总请求数添加到 goodput_df 中
goodput_df = pd.merge(goodput_df, total_requests, on=['system', 'model_name'], how='left')
goodput_df['goodput_rate'] = goodput_df['goodput']/goodput_df['total_num'] * 100
goodput_df

# %%
order = ['OPT-66B', 'BERT-21B', 'LLAMA-7B', 'BERT-QA', 'Resnet-152', 'Resnet-50']
systems = ['InferLine','Cocktail','Roger']
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()

# 使用Seaborn绘制柱状图
sns.barplot(x="model_name", y="goodput_rate", hue="system", data=goodput_df, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,hue_order=systems,order=order)

# 设置x轴和y轴的标签
ax.set_xlabel("Model")
ax.set_ylabel("Goodput (%)")

# set y ticks font size
ax.tick_params(axis='x', labelsize=10)
ax.set_ylim(0, 140)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, frameon=True, loc='upper left', fontsize=13, handletextpad=0.5, columnspacing=0.5, handlelength=1)

# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
ax.set_ylim(0,120)
ax.set_yticks(np.arange(0, 101, 20))
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch
num_systems = len(systems)
for i in range(num_systems):
    for j in range(len(order)):
        if i==2:
            ax.patches[(i * len(order)) + j].set_hatch('***')
# 显示bar 数据
for p in ax.patches:
    #if bar value == 0, the show "No Goodput" 
    if p.get_height() > 0 and p.get_height() < 99.9:
        ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=10, color='black', xytext=(0, 11), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E1-goodput-rate.pdf", bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Middleware Efficiency =wula_goodput_count / (prometheas所有阶段的总请求次数/阶段数）

# %%
goodput_df

# %%
goodput = goodput_df.groupby('system')['goodput_rate'].mean().reset_index()

# speedup
goodput['speedup'] = goodput['goodput_rate'].max() - goodput['goodput_rate']
goodput

# %%
#middleware efficiency
from pyexpat import model
from pyparsing import col


base_path = '../../../evaluation/metrics/system/E1_final/'

def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files
# 
openfaas_column_names = ['id','time','value','function_name','Task','status',"device_uuid"]
me_df = pd.DataFrame()
for file in get_csv_files_in_bottom_directory(base_path):
    if "invoke" in file:
        
        df = pd.read_csv(file,names=openfaas_column_names) 
        df['system'] = file.split("/")[-4]
        me_df = pd.concat([me_df, df], ignore_index=True)

#dropna on value column
me_df = me_df.dropna(subset=['value'])
me_df = me_df.drop(columns=['time','Task','device_uuid'])
me_df['model_name'] = me_df['function_name'].apply(lambda x: x.split("-")[:2])
me_df['model_name'] = me_df['model_name'].apply(lambda x: '-'.join(x))
me_df['model_name'] = me_df['model_name'].apply(lambda x: x.split(".")[0])
me_df['model_name'] = me_df['model_name'].replace(['bert-21b','llama-7b','bert-qa','resnet-152','resnet-50','opt-66b'],['BERT-21B','LLAMA-7B','BERT-QA','Resnet-152','Resnet-50','OPT-66B'])


#rename system 
me_df['system'] = me_df['system'].replace(["inferline","cocktail","roger"],['InferLine','Cocktail','Roger'])
me_df

# %%
# 根据 'function_name'、'system' 和 'cv' 分组，计算状态为 'end' 的 'value' 列与状态为 'start' 的 'value' 列之间的差值
end_values = me_df[me_df['status'] == 'end'].groupby(['model_name', 'system'])['value'].mean()
start_values = me_df[me_df['status'] == 'start'].groupby(['model_name', 'system'])['value'].mean()
difference = end_values - start_values
#change difference to dataframe
difference = difference.reset_index()
#rename new columns "invoker_num"
difference = difference.rename(columns={'value':'invoker_num'})
difference

# %% [markdown]
# 

# %%
#merge me_df and difference
merged_df = pd.merge(goodput_df, difference, on=['model_name', 'system'], how='left')
merged_df
#caculate me using goodput/invoker_num

# %%
merged_df['me'] = merged_df['goodput']/merged_df['invoker_num'] * 100
merged_df

# %%
order = ['InferLine','Cocktail','Roger']
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots(figsize=(7,3.5))
# 使用Seaborn绘制柱状图
sns.barplot(x='model_name', y='me', hue='system', data=merged_df, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,errorbar=None,hue_order=order)

# 设置x轴和y轴的标签
# plt.title('Percentage of StatusCode 500 by System and CV')
plt.xlabel('Model')
plt.ylabel('Middleware Efficiency (%)')


ax.set_ylim(0, 115)
ax.set_yticks(range(0, 101, 20))
# Rename legend name ["Cocktail",'InferLine','Roger']
ax.legend(['InferLine','Cocktail','Roger'], title='',fontsize=10, title_fontsize=10, frameon=True, ncol=3,loc='upper left')
# make x bar ticks font size smaller
ax.tick_params(axis='x', labelsize=11)

# ax.legend(fontsize='large', handleheight=2, handlelength=2
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch for Roger
for i in range(3):
    for j in range(6):
        if i==2:
            ax.patches[(i *6) + j].set_hatch('***')
# 显示bar 数据, if bar value == 0, the show "All Goodput", if bar value > 83, make it bold and red.
for p in ax.patches:
    if p.get_height() > 0 and p.get_height() < 99.8:
        ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=10, color='black', xytext=(0, 15), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()
# 将图形保存为PDF文件，并显示图形
plt.savefig("E1_middleware-efficiency.pdf",bbox_inches='tight')
plt.show()

# %% [markdown]
# ## avg queue length

# %%
base_path = '../../../evaluation/metrics/system/E1_final'

def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files

q_df = pd.DataFrame()
queque_column_names = ['id','time','value','function_name','task',"uuid"]
for file in get_csv_files_in_bottom_directory(base_path):
    
    if "openfaas" in file:
        system = file.split("/")[-4]
    
        df = pd.read_csv(file, names=queque_column_names)
        df['system'] = system
        # df['cv'] = file.split("/")[-2].split("_")[0]
        q_df = pd.concat([q_df, df], ignore_index=True)
# all_data = all_data_a[all_data_a['system'].isin(["ME","PARAM"])]
            # print(all_data)
    
q_df = q_df[(q_df['task'] == 'queue') & q_df['value'] != 0].dropna()
q_df['model_name'] = q_df['function_name'].apply(lambda x: x.split("-")[:2])
q_df['model_name'] = q_df['model_name'].apply(lambda x: '-'.join(x))
q_df['model_name'] = q_df['model_name'].apply(lambda x: x.split(".")[0])
q_df['model_name'] = q_df['model_name'].replace(['bert-21b','llama-7b','bert-qa','resnet-152','resnet-50','opt-66b'],['BERT-21B','LLAMA-7B','BERT-QA','Resnet-152','Resnet-50','OPT-66B'])


q_df = q_df.drop(columns=['time','task','uuid'])
q_df.reset_index(drop=True)
q_df
# # queue_length_df = queue_length_df.groupby(['function_name','cv_mu_duration'])["value"].mean().reset_index()
q_df = q_df.groupby(['system','model_name'])["value"].mean().reset_index()
#rename system name to ['Cocktail','InferLine','Roger']
q_df['system'] = q_df['system'].replace(['cocktail','inferline','roger'],['Cocktail','InferLine','Roger'])

q_df

# %%
queue = q_df.groupby('system')['value'].mean().reset_index()
queue['speedup'] = (queue['value'] - queue['value'].min())/ queue['value'] *100
queue

# %%
hue_order = ['InferLine','Cocktail','Roger']
order = ['OPT-66B','BERT-21B','LLAMA-7B','BERT-QA','Resnet-152','Resnet-50']
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots(figsize=(6,4))
# 使用Seaborn绘制柱状图
sns.barplot(x='model_name', y='value', hue='system', data=q_df , palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,errorbar=None,hue_order=hue_order,order=order)

# 设置x轴和y轴的标签
# plt.title('Percentage of StatusCode 500 by System and CV')
plt.xlabel('Model')
plt.ylabel('Avg Queue Length')


ax.set_ylim(0, 24)
# ax.set_yticks(range(0, 101, 20))
# Rename legend name ["Cocktail",'InferLine','Roger']
ax.legend(['InferLine','Cocktail','Roger'], title='System',fontsize=12, title_fontsize=14, frameon=True)
# make x bar ticks font size smaller
ax.tick_params(axis='x', labelsize=11)

# ax.legend(fontsize='large', handleheight=2, handlelength=2
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch for Roger
for i in range(3):
    for j in range(6):
        if i==2:
            ax.patches[(i *6) + j].set_hatch('***')
# 显示bar 数据, if bar value == 0, the show "All Goodput", if bar value > 83, make it bold and red.
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=10, color='black', xytext=(0, 15), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()
# 将图形保存为PDF文件，并显示图形
plt.savefig("E1_avg_queue_length_bar.pdf",bbox_inches='tight')
plt.show()


