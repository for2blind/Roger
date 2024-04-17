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
order = ["InferLine","Cocktail","Roger"]


# %%
base_path = '../../../evaluation/metrics/wula/E2_final/'

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
    df['system'] = file.split("/")[-5]
    df['cv'] = file.split("/")[-2].split("_")[0]
    df['mu'] = file.split("/")[-2].split("_")[1]
    df['duration'] = file.split("/")[-2].split("_")[2]
    all_data = pd.concat([all_data, df], ignore_index=True)

all_data


# %%
all_data['system'] = all_data['system'].replace('roger','Roger')
all_data['system'] = all_data['system'].replace('cocktail','Cocktail')
all_data['system'] = all_data['system'].replace('inferline','InferLine')

# %%
merged_data = all_data.copy()
all_data_renamed = all_data.copy()
merged_data['Latency'] = (merged_data['ResponseTime'] - merged_data['RequestTime'])/1e9
merged_data

# %%
#filter data from merged_data where model_name ='BERT-21B' and 'OPT-66B' and 'cv'!=8
merged_df = merged_data[merged_data['model_name'].isin(['BERT-21B','OPT-66B'])]
merged_df = merged_df[merged_df['cv']!='8']
merged_df 

# %%
#caculate percentile of latency groupby model_name and cv where percentile is [55,65,75,85,95,97,98,99]
percentile = [0.55,0.65,0.75,0.85,0.95,0.97,0.98,0.99]
percentile_latency = merged_df.groupby(['model_name','cv','system'])['Latency'].quantile(percentile).unstack()
percentile_latency = percentile_latency.reset_index()
percentile_latency


# %%
melted_df = pd.melt(percentile_latency, id_vars=['model_name','cv','system'], value_vars=percentile)
melted_df = melted_df.rename(columns={'variable':'percentile','value':'latency'})
melted_df

# %%
#rename of percentile to ['55th', '65th', '75th', '85th', '95th', '97th', '98th', '99th']
melted_df['percentile'] = melted_df['percentile'].replace({0.55:'55th',0.65:'65th',0.75:'75th',0.85:'85th',0.95:'95th',0.97:'97th',0.98:'98th',0.99:'99th'})
melted_df

# %%
melted_df.to_csv('latency_percentile.csv',index=False)

# %%
#draw eight subplots 2 row 4 column, each subplot is a lineplot of latency groupby cv and model_name, hue is system,x-axis is percentile, y-axis is latency
colors =['#4a4c7e','#367182','#80c161']
fig, axes = plt.subplots(2, 4, figsize=(20, 6))
# each subplot is a lineplot of latency groupby 'cv' and 'model_name', hue is 'system', x-axis is 'percentile', y-axis is 'latency'
#every subplot has a title of model_name and cv
for i, (model_name, df) in enumerate(melted_df.groupby('model_name')):
    for j, (cv, df_cv) in enumerate(df.groupby('cv')):
        ax = axes[i, j]
        #add markers to lineplot
        sns.lineplot(data=df_cv, x='percentile', y='latency', hue='system', palette=colors, ax=ax,marker='h', markersize=8,linewidth=2.5,hue_order=order)

        ax.set_title(f'{model_name} cv={cv}')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Latency (s)')
        ax.set_xticks(range(8))
        ax.set_xticklabels(['55th', '65th', '75th', '85th', '95th', '97th', '98th', '99th'])
        #For ax.lenged if cv =='4'  loc = 'bottom right' else loc = 'upper left'
        if cv == '4':
            ax.legend(title='', loc='lower right')
        else:
            ax.legend(title='', loc='upper left')
         
        #add gird only y-axis and aplha=0.7
        ax.grid(linestyle='--', alpha=0.7)
        
plt.tight_layout()
plt.savefig("E2-Quantile-Latency-cv.pdf", bbox_inches='tight')
plt.show()



# %% [markdown]
# ## Goodput rate

# %%
goodput_cv = merged_data[merged_data['StatusCode'] == 200].copy()
goodput_cv = goodput_cv.groupby(['system','model_name','cv'])['StatusCode'].count().reset_index().rename(columns={'StatusCode':'goodput'})
# 计算每个系统和模型名称组合的总请求数
goodput_cv
total_requests = all_data.groupby(['system', 'model_name','cv'])['StatusCode'].count().reset_index().rename(columns={'StatusCode': 'total_num'})
# 合并两个 DataFrame，将总请求数添加到 goodput_df 中
goodput_cv = pd.merge(goodput_cv, total_requests, on=['system', 'model_name','cv'], how='left')

goodput_cv['goodput_rate'] = goodput_cv['goodput']/goodput_cv['total_num'] * 100

 
goodput_cv = goodput_cv[goodput_cv['cv']!='8']
# caculate avg goodput_rate x-axis is cv, y-axis is goodput_rate,hue is system
goodput_cv_avg = goodput_cv.groupby(['cv','system'])['goodput_rate'].mean().unstack().reset_index()

# goodput_cv_avg = goodput_cv.groupby(['cv'])['goodput_rate'].mean().unstack().reset_index()
goodput_cv_avg 

# %%
# melted goodput_cv_avg
melted_goodput_cv_avg = pd.melt(goodput_cv_avg, id_vars=['cv'], value_vars=['Cocktail','InferLine','Roger'])
melted_goodput_cv_avg

# %%
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots(figsize=(6, 4))
# colors=["#94c67a","#3d65b2","#6f31a1","#ff9538"]
# 使用Seaborn绘制柱状图
sns.barplot(x="cv", y="value", hue="system", data=melted_goodput_cv_avg, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,hue_order=order)

# 设置x轴和y轴的标签
ax.set_xlabel("Coefficient of Variation (CV)")
ax.set_ylabel("Goodput (%)")

# set y ticks font size
ax.tick_params(axis='x', labelsize=10)
ax.set_ylim(60, 115)
#set y_ticks
ax.set_yticks(range(60, 101, 20))
# handles, labels = ax.get_legend_handles_labels()
# ax.legend( handles, labels,title = 'System', ncol=3, frameon=True ,loc='upper center') 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, frameon=True, loc='upper left', fontsize=13, handletextpad=0.5, columnspacing=0.5, handlelength=1)
# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
# ax.set_ylim(bottom=0.5)
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch
# num_systems = len(systems)
for i in range(3):
    for j in range(4):
        # if i==0:
        #     ax.patches[(i *4) + j].set_hatch('**')
        # if i==1:
        #     ax.patches[(i *4) + j].set_hatch('///')
        if i==2:
            ax.patches[(i *4) + j].set_hatch('***')
# 显示bar 数据
for p in ax.patches:
    if p.get_height() > 0 and p.get_height() < 99.8:
        ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=11, color='black', xytext=(0, 14), textcoords='offset points', rotation=90)
    # ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=8, color='black', xytext=(0, 11), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E2-goodput-final.pdf", bbox_inches='tight')
plt.show() 

# %% [markdown]
# ### Middleware Efficiency =wula_goodput_count / (prometheas所有阶段的总请求次数/阶段数）

# %%
#
me_df = all_data_renamed.copy()
#filter goodput from me_df where StatusCode = 200
me_df = me_df[me_df['StatusCode'] == 200]
#count goodput groupby system,model_name,cv
me_df = me_df.groupby(['system','model_name','cv'])['StatusCode'].count().reset_index().rename(columns={'StatusCode':'goodput'})
#change system name to lower case
me_df['system'] = me_df['system'].str.lower()
me_df


# %%
base_path = '../../../evaluation/metrics/system/E2_final/'

def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files
# 
prome_column_names = ['id','time','value','function_name','Task','status',"device_uuid","cv_mu_duration"]
prome_df = pd.DataFrame()
for file in get_csv_files_in_bottom_directory(base_path):
    
    if 'invoke' in file:    
        df = pd.read_csv(file,names=prome_column_names)    
        df['system'] = file.split("/")[-4]
        prome_df = pd.concat([prome_df, df], ignore_index=True)


prome_df['cv']=prome_df['cv_mu_duration'].apply(lambda x: x.split("_")[0])
prome_df = prome_df.drop(columns=['time','Task','device_uuid','cv_mu_duration'])
prome_df['model_name'] = prome_df['function_name'].apply(lambda x: x.split("-")[:2])

prome_df['model_name'] = prome_df['model_name'].apply(lambda x: '-'.join(x))
prome_df['model_name'] = prome_df['model_name'].apply(lambda x: x.split(".")[0])
#replace model_name into upper case
prome_df['model_name'] = prome_df['model_name'].apply(lambda x: x.upper())
#replace model_name from ['BERT-21B', 'LLAMA-7B', 'BERT-QA', 'RESNET-152', 'RESNET-50','OPT-66B'] to ['BERT-21B', 'BERT-QA', 'LLAMA-7B', 'OPT-66B', 'Resnet-152','Resnet-50'] respectively
prome_df['model_name'] = prome_df['model_name'].replace(['BERT-21B', 'LLAMA-7B', 'BERT-QA', 'RESNET-152', 'RESNET-50','OPT-66B'],['BERT-21B', 'LLAMA-7B', 'BERT-QA', 'Resnet-152', 'Resnet-50','OPT-66B'])
#replace system's first letter into upper case
# prome_df['system'] = prome_df['system'].apply(lambda x: x.capitalize())
# prome_df['model_name'] = prome_df['model_name'].replace('BERT-21B','BERT-21')
prome_df

# %%
# 根据 'function_name'、'system' 和 'cv' 分组，计算状态为 'end' 的 'value' 列与状态为 'start' 的 'value' 列之间的差值
end_values = prome_df[prome_df['status'] == 'end'].groupby(['model_name', 'system', 'cv'])['value'].mean()
start_values = prome_df[prome_df['status'] == 'start'].groupby(['model_name', 'system', 'cv'])['value'].mean()
difference = end_values - start_values
#change difference to dataframe
difference = difference.reset_index()
#rename new columns "invoker_num"
difference = difference.rename(columns={'value':'invoker_num'})
difference



# %%
#merge me_df and difference
me_df = pd.merge(me_df, difference, on=['model_name', 'system', 'cv'], how='left')
me_df
#caculate me using goodput/invoker_num


# %%
me_df['me'] = me_df['goodput']/me_df['invoker_num'] * 100


# %%
#reset index and filter cv != 8 and cover the original index
# me_df = me_df[me_df['cv']!='8']
me_df = me_df.reset_index(drop=True)
me_df
me_df_no8 = me_df.copy()
me_df_no8 = me_df_no8[me_df_no8['cv']!='8']



# %%
order = ["inferline","cocktail","roger"]
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots(figsize=(6,4))
# 使用Seaborn绘制柱状图
sns.barplot(x='cv', y='me', hue='system', data=me_df, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,errorbar=None,hue_order=order)

# 设置x轴和y轴的标签
# plt.title('Percentage of StatusCode 500 by System and CV')
plt.xlabel('Coefficient of Variation (CV)')
plt.ylabel('Middleware Efficiency (%)')


ax.set_ylim(0, 140)
ax.set_yticks(range(0, 101, 20))
# Rename legend name ["Cocktail",'InferLine','Roger']
# ax.legend(['InferLine','Cocktail','Roger'], title='System',ncol=3, frameon=True, loc='upper right') 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, frameon=True, loc='upper left', fontsize=13, handletextpad=0.5, columnspacing=0.5, handlelength=1)
# ax.legend(fontsize='large', handleheight=2, handlelength=2
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch for Roger
for i in range(3):
    for j in range(5):
        if i==2:
            ax.patches[(i *5) + j].set_hatch('***')
# 显示bar 数据, if bar value == 0, the show "All Goodput", if bar value > 83, make it bold and red.
for p in ax.patches:
    if p.get_height() > 0 and p.get_height() < 99.9:
        ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=11, color='black', xytext=(0, 17), textcoords='offset points', rotation=90)
    # ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=12, color='black', xytext=(0, 17), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()
# 将图形保存为PDF文件，并显示图形
plt.savefig("E2_middleware-efficiency_cv8.pdf",bbox_inches='tight')
plt.show()

# %%
order = ["inferline","cocktail","roger"]
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots(figsize=(6,4))
# 使用Seaborn绘制柱状图
sns.barplot(x='cv', y='me', hue='system', data=me_df_no8, palette='viridis', ax=ax, width=0.75, edgecolor="white", linewidth=0,errorbar=None,hue_order=order)

# 设置x轴和y轴的标签
# plt.title('Percentage of StatusCode 500 by System and CV')
plt.xlabel('Coefficient of Variation (CV)')
plt.ylabel('Middleware Efficiency (%)')


ax.set_ylim(60, 115)
ax.set_yticks(range(60, 101, 20))
# Rename legend name ["Cocktail",'InferLine','Roger']
# ax.legend(['InferLine','Cocktail','Roger'], title='System',ncol=3, frameon=True, loc='upper right') 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, frameon=True, loc='upper left', fontsize=13, handletextpad=0.5, columnspacing=0.5, handlelength=1)
# ax.legend(fontsize='large', handleheight=2, handlelength=2
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch for Roger
for i in range(3):
    for j in range(4):
        if i==2:
            ax.patches[(i *4) + j].set_hatch('***')
# 显示bar 数据, if bar value == 0, the show "All Goodput", if bar value > 83, make it bold and red.
for p in ax.patches:
    if p.get_height() > 0 and p.get_height() < 99.9:
        ax.annotate('%.1f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=11, color='black', xytext=(0,15), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()
# 将图形保存为PDF文件，并显示图形
plt.savefig("E2_middleware-efficiency_nocv8.pdf",bbox_inches='tight')
plt.show()

# %%
me_df_no8

# %%
me_df_no8_group = me_df_no8.groupby(['system','cv'])['me'].mean().reset_index()
me_df_no8_group.to_csv('me_no8.csv',index=False)

# %% [markdown]
# ## Avg Queue Length

# %%

base_path = '../../../evaluation/metrics/system/E2_final'

def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files

q_df = pd.DataFrame()
queque_column_names = ['id','time','value','function_name','metric',"uuid","cv_mu_duration"]
for file in get_csv_files_in_bottom_directory(base_path):
    
    if "openfaas" in file:
        system = file.split("/")[-4]
    
        df = pd.read_csv(file, names=queque_column_names)
        df['system'] = system
        # df['cv'] = file.split("/")[-2].split("_")[0]
        q_df = pd.concat([q_df, df], ignore_index=True)
# all_data = all_data_a[all_data_a['system'].isin(["ME","PARAM"])]
            # print(all_data)
    
q_df = q_df[(q_df['metric'] == 'queue') & q_df['value'] != 0].dropna()

q_df['cv'] = q_df['cv_mu_duration'].apply(lambda x: x.split("_")[0])
q_df = q_df.drop(columns=['time','metric','uuid','cv_mu_duration'])
q_df.reset_index(drop=True)
q_df
# # queue_length_df = queue_length_df.groupby(['function_name','cv_mu_duration'])["value"].mean().reset_index()
q_df = q_df.groupby(['system','cv'])["value"].mean().reset_index()
#rename system name to ['Cocktail','InferLine','Roger']
q_df['system'] = q_df['system'].replace(['cocktail','inferline','roger'],['Cocktail','InferLine','Roger'])
q_df = q_df[q_df['cv']!='8']
q_df

# %%
fig, ax = plt.subplots()
sns.lineplot(x="cv", y="value",  hue='system', marker='h',data=q_df,linewidth=3,palette='viridis', ax=ax, markersize=11,hue_order=["InferLine","Cocktail","Roger"])
ax.set_xlabel('Coefficient of Variation (CV)')
ax.set_ylabel('Avg Queue Length')
ax.grid(axis="y", linestyle='--', alpha=0.7)

legend = ax.legend( fontsize=15, title_fontsize=15)

plt.tight_layout()
plt.savefig("E2_avg_queue_length.pdf", bbox_inches='tight')

# Show plot
plt.show()

# %% [markdown]
# ## Fault count

# %%
filtered_df_500 = all_data[all_data['StatusCode'] == 500]
# 按照 system 和 model_name 分组，并计算每组的行数
fault_counts = filtered_df_500.groupby(['system', 'model_name'])['StatusCode'].count().reset_index(name='fault_count')

fault_counts

# %%
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()
# 使用Seaborn绘制柱状图
sns.barplot(x="model_name", y="fault_count", hue="system", data=fault_counts, palette=colors, ax=ax, width=0.75, edgecolor="white", linewidth=0)

# 设置x轴和y轴的标签
ax.set_xlabel("Model")
ax.set_ylabel("Fault Number")

# set y ticks font size
ax.tick_params(axis='x', labelsize=10)
ax.set_ylim(0, 23000)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=1, frameon=True)
# plt.legend(title='Pipeline Length',loc='upper center', ncol=3)
# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
# ax.set_ylim(bottom=0.5)
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch

# 显示bar 数据
for p in ax.patches:
    ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color='black', xytext=(0,20), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E2-fault-count.pdf", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Goodput rate

# %%
goodput_df = all_data[all_data['StatusCode'] == 200].copy()

goodput_df = goodput_df.groupby(['system','model_name'])['StatusCode'].count().reset_index().rename(columns={'StatusCode':'goodput'})
# 计算每个系统和模型名称组合的总请求数
total_requests = all_data.groupby(['system', 'model_name'])['StatusCode'].count().reset_index().rename(columns={'StatusCode': 'total_num'})
# 合并两个 DataFrame，将总请求数添加到 goodput_df 中
goodput_df = pd.merge(goodput_df, total_requests, on=['system', 'model_name'], how='left')

goodput_df['goodput_rate'] = goodput_df['goodput']/goodput_df['total_num'] * 100

# goodput_df['goodput_rate'] = goodput_df['goodput']/
goodput_df

# %%
# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()
colors=["#94c67a","#3d65b2","#6f31a1","#ff9538"]
# 使用Seaborn绘制柱状图
sns.barplot(x="model_name", y="goodput_rate", hue="system", data=goodput_df, palette=colors, ax=ax, width=0.75, edgecolor="white", linewidth=0)

# 设置x轴和y轴的标签
ax.set_xlabel("Model")
ax.set_ylabel("Goodput (%)")

# set y ticks font size
ax.tick_params(axis='x', labelsize=10)
ax.set_ylim(0, 130)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, frameon=True ,loc='upper center') 

# 获取图例的句柄和标签，并修改标签名称
# 设置x轴刻度的标签
# ax.set_ylim(bottom=0.5)
# 添加网格线
ax.grid(axis="y", linestyle='--', alpha=0.7)
#add hatch
# num_systems = len(systems)
for i in range(3):
    for j in range(6):
        if i==0:
            ax.patches[(i * 6) + j].set_hatch('**')
        if i==1:
            ax.patches[(i * 6) + j].set_hatch('///')
        if i==2:
            ax.patches[(i * 6) + j].set_hatch('+++')
# 显示bar 数据
for p in ax.patches:
    ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=8, color='black', xytext=(0, 11), textcoords='offset points', rotation=90)

# 紧凑布局
plt.tight_layout()

# 将图形保存为PDF文件，并显示图形
plt.savefig("E2-goodput-rate-2.pdf", bbox_inches='tight')
plt.show() 

# %%
merged_data

# %%
goodput_cv = merged_data[merged_data['StatusCode'] == 200].copy()
goodput_cv = goodput_cv.groupby(['system','model_name','cv'])['StatusCode'].count().reset_index().rename(columns={'StatusCode':'goodput'})
# 计算每个系统和模型名称组合的总请求数
goodput_cv
total_requests = all_data.groupby(['system', 'model_name','cv'])['StatusCode'].count().reset_index().rename(columns={'StatusCode': 'total_num'})
# 合并两个 DataFrame，将总请求数添加到 goodput_df 中
goodput_cv = pd.merge(goodput_cv, total_requests, on=['system', 'model_name','cv'], how='left')

goodput_cv['goodput_rate'] = goodput_cv['goodput']/goodput_cv['total_num'] * 100


goodput_cv


# %% [markdown]
# 

# %%
# draw six sub-plots
fig, ax = plt.subplots(2, 3, figsize=(18, 8))
plt.rcParams.update({'font.size': 10})
# 增大子图之间的间距
fig.subplots_adjust(wspace=0.5)
# models = ['OPT-66B', 'BERT-21B', 'LLAMA-7B', 'BERT-QA', 'Resnet-152', 'Resnet-50']
systems = ['InferLine','Cocktail','Roger']
# # 加虚线
# ensure grid lines are drawn below all plot elements
for i in range(2):
    for j in range(3):
        ax[i, j].grid(ls='--', axis='y')  

# 将子图按照model_name分组
colors=["#94c67a","#3d65b2","#6f31a1","#ff9538"]

# Loop through each model_name
for model_name, ax in zip(goodput_cv['model_name'].unique(), ax.flatten()):
    # Filter data for current model_name
    model_data = goodput_cv[goodput_cv['model_name'] == model_name]
    
    # Plot each system's data on the same subplot
    sns.barplot(data=model_data, x='cv', y='goodput_rate', hue='system', ax=ax,palette=colors,width=0.75, edgecolor="white", linewidth=0)
    # 设置标题和标签
    ax.set_title(model_name)
    ax.set_xlabel('cv')
    ax.set_ylabel('Goodput Rate(%)')
    ax.set_ylim(0,150)
    # 添加图例
    ax.legend(title='System',loc='upper right', ncol=3)
    for p in ax.patches:
        ax.annotate('%.0f' % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()+0.1), ha='center', va='center', fontsize=10, color='black', xytext=(0, 11), textcoords='offset points', rotation=90)
# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig("E2-Goodput-cv.pdf", bbox_inches='tight')
plt.show()


