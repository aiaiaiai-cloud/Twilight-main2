---
title: 数据可视化
published: 2021-12-02
tags:
  - python
  - 学习笔记
category: Examples
draft: false
---
# 数据可视化考试知识点笔记

## 第 1 章 数据可视化基础

### 核心知识点

1. **数据可视化的本质**
    
    利用图形化手段，将一组数据以图形形式呈现，借助开发工具和数据分析，挖掘其中未知信息的处理过程，本质是从数据空间到图形空间的映射。
2. **数据可视化的核心目标**
    
    准确、高效且全面地传递数据信息，建立数据间关联，发现数据规律与特征，挖掘有价值信息，提升数据沟通效率。
3. **数据可视化的分类**
    
    分为层次数据可视化、多维数据可视化、时序数据可视化、地理数据可视化四类。
4. **各类数据的核心特征**
    - 层次数据：存在父子 / 包含关系
    - 多维数据：由多属性进行描述
    - 时序数据：包含时间戳，关注数据变化
    - 地理数据：绑定地理位置信息
5. **数据可视化流程三大核心要素**
    
    数据表示与变换、可视化呈现、用户交互（流程环节依次为原始数据、数据预处理、数据过滤、数据映射、几何图形绘制、图像数据分析）
6. **数据可视化设计 “四级级联” 层次**
    - 问题刻画层：明确要解决的具体问题
    - 抽象层：将业务需求转化为对应的数据
    - 编码层：选择合适的图表类型
    - 算法实现层：使用 Python 工具完成图表绘制

## 第二章 Matplotlib 绘图基础

### 核心知识点

1. **Matplotlib 导入与基础配置**

python

运行

```python
import numpy as np  # 负责生成数据
import matplotlib.pyplot as plt  # 核心绘图库，简写为plt
# 解决中文显示乱码（小方框）和负号显示异常问题
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# Mac系统中文设置：plt.rcParams['font.family'] = 'Arial Unicode MS'
```

2. **绘图核心对象**
    
    - **Figure 画布**：是所有图表的顶层容器，相当于一张白纸，可通过`fig.patch`设置背景矩形，`fig.axes`获取子图列表，`fig.texts`设置图表文字。
    - **Axes 子图**：拥有诸多属性与方法，具体如下：
        
        |属性名|描述|示例操作|
        |---|---|---|
        |ax.patch|Axes 的背景矩形|ax.patch.set_facecolor('yellow')|
        |ax.lines|Axes 上所有折线 (Line2D) 的列表|ax.lines[0].set_color('red')|
        |ax.xaxis|x 轴对象 (控制刻度、标签等)|ax.xaxis.set_label_text('X')|
        |ax.yaxis|y 轴对象 (控制刻度、标签等)|ax.yaxis.set_ticks([0, 2, 4])|
        |ax.legend|图例对象|ax.legend(loc='upper right')|
        
    
    同时 Axes 支持多种绘图方法，对应不同图表类型：
    
    |Axes 方法|图表类型|存储列表|示例代码|
    |---|---|---|---|
    |ax.plot()|折线图|ax.lines|ax.plot(x, y, 'r-')|
    |ax.scatter()|散点图|ax.collections|ax.scatter(x, y, c='b')|
    |ax.bar()|柱状图|ax.patches|ax.bar(x, y, width=0.8)|
    |ax.hist()|直方图|ax.patches|ax.hist(data, bins=10)|
    |ax.text()|添加文字|ax.texts|ax.text (1, 2,' 文字 ')|
    
3. **子图划分**
    
    - **plt.subplot()**

python

运行

```python
x = np.arange(1, 11)
plt.subplot(1, 2, 1)  # 第1行第1列子图（可简写为plt.subplot(121)）
plt.plot(x, x, color='red')
plt.title('y=x')
plt.subplot(122)  # 第1行第2列子图
plt.plot(x, -x, color='blue')
plt.title('y=-x')
# 关键：调整子图间距，避免标题/标签重叠
plt.tight_layout()  # 自动调整间距
plt.show()
```

- **plt.subplots()**（批量创建子图）

python

运行

```python
import matplotlib.pyplot as plt
import numpy as np

# 返回一个画布对象fig, ax是二维数组，2行2列共4个子图
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
fig.patch.set_color('b')  # 画布背景蓝色
fig.patch.set_alpha(0.1)  # 画布背景透明度0.1

x = np.arange(1, 11)  # 生成1到10的整数数组（共10个元素）
ax[0, 0].plot(x, x)        # 子图(0,0)：绘制y = x的直线
ax[0, 1].plot(x, -x)       # 子图(0,1)：绘制y = -x的直线
ax[1, 0].plot(x, x**2)     # 子图(1,0)：绘制y = x²的抛物线
ax[1, 1].plot(x, np.log(x))# 子图(1,1)：绘制y = ln(x)的对数曲线

# ravel()将二维数组ax转为一维数组，遍历每个子图设置标题
for i, axes in enumerate(ax.ravel()):
    axes.set_title('ax' + str(i + 1), fontsize='20')  # 子图标题：ax1、ax2、ax3、ax4

fig.tight_layout()  # 自动调整子图间距，避免标题/坐标轴标签重叠
plt.show()
```

- **fig.add_subplot()**（灵活添加子图）

python

运行

```python
fig = plt.figure() 
# 创建新画布 
ax1 = fig.add_subplot(221) # 2行2列的第1个子图 
ax1.text(0.4, 0.5, 'ax1' , c='r' , fontsize=18) 
ax3 = fig.add_subplot(223) # 2行2列的第3个子图 
ax3.text(0.4, 0.5, 'ax3' , c='r' , fontsize=18) 
ax2 = fig.add_subplot(122) # 1行2列的第2个子图 
ax2.text(0.4, 0.5, 'ax2' , c='r' , fontsize=18)
```

注：`ax.text(0.4, 0.5, ...)` 中的 x=0.4 和 y=0.5 是子图内归一化坐标，范围 x∈[0,1]、y∈[0,1]；`add_subplot(num)`中 num 为 3 位数字，代表行数 - 列数 - 子图编号，编号从左上到右下递增。4. **基础绘图函数与图形类型**折线图（`plt.plot()`）、柱状图（`plt.bar()`）、水平条形图（`plt.barh()`）、散点图（`plt.scatter()`）、饼图（`plt.pie()`）、箱线图（`plt.boxplot()`）、雷达图（`plt.polar()`）5. **图形基础设置**标题（`plt.title()`）、轴标签（`plt.xlabel()`/`plt.ylabel()`）、图例（`plt.legend()`）、网格线（`plt.grid()`）、图形显示（`plt.show()`）、图形保存（`plt.savefig()`）6. **Axes 对象相关操作**可通过`plt.gca()`获取当前子图

## 第三章 Matplotlib 图形类别

### 核心知识点

1. **连续型数据可视化**
    - **折线图**：将数据点用直线连接，用于展示连续型数据变量间的关系
    - **阶梯图**：折线图的 y 值保持恒定直至发生变化再跳跃至下一个值，形状类似阶梯
    - **面积图**：又称堆积折线图，将折线与 x 轴间区域填充色块或纹理，多由多组时序数据构成，强调数据随时间的变化及总值变化趋势，总体面积代表总量，各层面积代表分量总和
2. **离散型数据可视化**
    - **散点图**：由散乱的点构成，在回归分析和预测中使用频繁，函数格式为`plt.scatter(x,y,s=None,c=None,marker=None,cmap=None,edgecolors=None, **kwargs)`
        - s：点的大小；c：颜色；marker：点的标记样式
        - cmap：字符串 / Colormap，c 为数值数组时映射为渐变色
        - edgecolors：散点边缘色，None 时同 c 色
        - alpha：0-1 浮点数，控制透明度以解决重叠遮挡
        - label：图例标签
    - **柱状图**：以柱体高度或长度表示数据变化，函数`plt.bar(x, height, width=0.8, bottom=None,align='center' , *kwargs)`
        - x：柱位置；height：柱高度
        - width：默认 0.8，控制柱子宽度
        - bottom：默认 None，设置柱子底部起始位置
        - align：默认 'center'，指定 x 与柱子的对齐方式
    - **堆积柱状图**：将同一类别的若干柱体上下堆积，可同时对比单个数据和类别总和数据
3. **比例数据可视化**
    - **饼图**：适合表现比例、份额类数据，能清晰呈现各部分占比，但不适合精确对比，且子类最好控制在 10 个左右，使用`plt.pie()`绘制
    - **分裂式饼图**：通过设置`explode`参数突出特定扇形，如`explode=(0, 0.2, 0, 0, 0, 0)`可将第 2 个扇形突出 0.2 的位置

python

运行

```python
x=[10, 5, 20, 10, 25, 15]
labels=('波斯猫','加菲猫', '短毛猫', '狸花猫','孟加拉猫', '其它')
explode =(0, 0.2, 0, 0, 0, 0) #第2个扇形突出0.2
patch, txt, pct = plt.pie(x, labels=labels, autopct='%.1f%%',explode=explode, textprops={'fontsize' : 12})
#patch:扇形列表, txt标签文字列表,pct: 百分比文字列表
for p in patch:
    p.set_alpha(0.6) #设置每个扇形的透明度
plt.legend(bbox_to_anchor=(1, 1))
```

- **环形图**：以外围不同颜色弧形长度表示数值占比，与饼图不同，可表示多个指标分类的占比情况

4. **关系数据可视化**
    - **气泡图**：类似散点图，通过气泡大小反映第三个变量变化，若辅以不同颜色可反映四维变量
    - **直方图**：统计图表，以高度不等的柱体表示数据分布，横轴为数据区间段，纵轴为对应区间数据频次
    - **堆积直方图**：可在同一画布展示多组数据分布，需提供两组数据并指定`stacked=True`，同时`bins`参数可设为数值列表自定义分段区域（数据边界左闭右开）

## 第四章 Matplotlib 统计图形绘制

### 核心知识点

1. **箱线图绘制及参数设置**
    
    箱线图用于展示数据分布，由箱体和一对箱须构成，箱体下边沿为下四分位数 Q1、上边沿为上四分位数 Q3，箱内横线为中位数 Q2，箱须外数值为离群值，用小圆圈标注。
    
    绘制函数`plt.boxplot(x,sym=,whis=,widths=,patch_artist=)`，参数说明如下：
    
    |参数名|说明|参数名|说明|
    |---|---|---|---|
    |x|数据序列|widths|设置箱体宽度|
    |patch_artist|是否给箱体设置颜色|vert|True 纵向，False 横向|
    |showmeans|默认 False 不显示，True 则显示均值线|meanline|默认 False 均值用点表示，True 均值用线表示|
    |labels|刻度标签|sym|离群点的标记样式|
    |whis|四分位间距的倍数，确定箱须包含数据的范围，默认值 1.5|-|-|
    
2. **极线图（雷达图）绘制**
    
    使用`plt.polar()`绘制，绘制在极坐标系上，通过极角和极径对比数据差异，可同时绘制多个数据序列，各点代表对应数据指标
3. **误差棒图、等高线图、3D 绘图的基础概念**
    - **误差棒图**：可视化实验数据时用于表示测量偏差，以测量值算术平均值为中点，上下两端线段代表数据可能的置信区间，计算方法有单一数值、置信区间、标准差、标准误等，样式分水平、垂直、对称、非对称误差棒，绘制函数`plt.errorbar()`，参数包括 x/y（数据坐标）、yerr/xerr（对应轴方向误差）、ecolor（误差棒颜色）、elinewidth（误差棒线宽）、capsize（误差棒头部小横线宽度）、errorevery（绘制间隔）
    - **等高线图**：数据分析中用于反映数值变化，本质是绘制函数 z=f (x,y) 的变化，将 z 值相等的点连成平滑曲线投影到平面，需先用`np.meshgrid`生成平面网格坐标，可用于机器学习决策区域绘制，`contour()`绘制决策边界线，`contourf()`给决策区域着色
    - **3D 绘图**：需将 Axes 子图设为三维坐标系，有两种方法，一是创建子图时指定`projection='3d'`，二是导入`mpl_toolkits.mplot3d`中的`Axes3D`对象转换画布；可绘制 3D 柱状图（多组数据可在三维空间分层对比，zdir 仅改变观察角度）和 3D 曲面图（绘制 z=f (x,y)，用`plot_surface(x, y, z, rstride=1, cstride=1)`，rstride 和 cstride 控制曲面行列跨度，最小值为 1 时光滑度最高）

## 第五章 Matplotlib 绘图高阶设置

### 核心知识点

1. **坐标轴与刻度设置**
    - 轴标签：`plt.xlabel()`/`plt.ylabel()`或`ax.set_xlabel()`/`ax.set_ylabel()`
    - 轴范围：`plt.xlim()`/`plt.ylim()`/`plt.axis()`或`ax.set_xlim()`/`ax.set_ylim()`
    - 轴刻度：`plt.xticks()`/`plt.yticks()`或`ax.set_xticks()`/`ax.set_yticks()`，还可设置主次刻度、绘制双轴图、实现多子图共享坐标轴
2. **图表配色**
    - **颜色表示方式**：英文单词（如 white、black、red 等）、字母缩写（如 w、k、r 等）、十六进制、RGB 元组，常见颜色对应如下：
        
        |英文单词|字母缩写|英文单词|字母缩写|
        |---|---|---|---|
        |white|w|blue|b|
        |black|k|maroon|m|
        |yellow|y|lightgreen|-|
        |green|g|skyblue|-|
        |cyan|c|pink|-|
        |purple|-|red|r|
        
    - **颜色映射**
        - 可使用默认或指定映射表（如 spring、winter、Greys、Greys_r），示例如下：

python

运行

```python
np.random.seed(7)
p = np.random.rand(20, 2)
dottype = np.random.randint(0, 3, 20)
plt.scatter(p[:, 0], p[:, 1], s=60, c=dottype) #默认映射表
plt.scatter(p[:, 0]+0.1, p[:, 1]+0.1, s=60, c=dottype, cmap='spring') #指定spring映射表
```

- 也可自定义颜色映射表，示例如下：

python

运行

```python
import matplotlib as mpl
np.random.seed(7)
p=np.random.rand(20, 2)
dottype =np.random.randint(0, 3, 20) #类型值0、1、2
mycolormap = mpl.colors.ListedColormap(['r', 'g', 'b']) #自定义颜色映射表
plt.scatter(p[:, 0], p[:, 1], s=60, c=dottype, cmap=mycolormap)
```

- **颜色标尺**：用`plt.colorbar()`添加，可通过`shrink`设定标尺高度（如`shrink=0.8`为图形高度的 80%）、`aspect`规定宽度（值越大越窄）

3. **文本属性设置**

python

运行

```python
plt.rcParams['font.family'] = 'SimHei' # 设置使用黑体字体以正常显示中文
plt.rcParams['font.size'] = 18 # 全局文本默认大小18
plt.rcParams['text.color'] = 'blue' # 全局文本蓝色
```

文本属性参数如下：

|属性参数|取值|含义|
|---|---|---|
|family|Arial/sans-serif/simhei/simsun|字体名称或类型，按列表顺序匹配|
|size/fontsize|9/10/12…/30xx、small/large 等|字号大小|
|style/fontstyle|normal/italic/oblique|字体风格|
|weight/fontweight|0~1000、light/normal/bold 等|字体粗细|

## 第六章 Matplotlib 库其他绘图函数

### 核心知识点

1. **文本数据可视化**
    - **jieba 分词**：使用`jieba.lcut()`进行分词，`jieba.add_word()`添加新词，示例如下：

python

运行

```python
import jieba
s='欢迎报考广东金融学院的相关专业!'
print('原句子:', s)
lst =jieba.lcut(s)
print(lst)
#更新词库之前的分词结果
print('**************更新词库后**************')
jieba.add_word('广东金融学院')
#临时添加新词"广东金融学院"
#更新词库之后的分词结果
print(jieba.lcut(s))
```

- **词云图**：需使用 wordcloud 库，安装命令`pip install wordcloud`，示例如下：

python

运行

```python
from wordcloud import WordCloud
s= "With the improving of semiconductor technology, a single chip integrates more and more processing cores. Highly parallel applications are distributed to tens of processing units. The inter-processor communication delay becomes more and more important for parallel applications."
#1.构造一个WordCould对象(背景颜色,宽,高)
wc= WordCloud(background_color='white', width=1000, height=800)
# 2. 传入字符串s, s应为英文句子格式(词汇间空格分隔)
#生成词云
wc.generate(s)
# 3.调用to_file()方法将词云保存为图片文件(png, jpg等)
wc.to_file('img/wc.png')
```

2. **动态图、世界地图绘制的基础概念**
    - **动态图**：Matplotlib 可绘制动态图形并保存为.gif 文件，核心函数为`FuncAnimation`，参数包括 fig（承载动画的画布）、func（动画更新函数）、frames（一轮动画数据）、init_func（初始化函数，非必须）、repeat（是否循环，默认 True）、interval（帧更新间隔，默认 200 毫秒），示例如下：

python

运行

```python
%matplotlib notebook
from matplotlib.animation import FuncAnimation #引入动图函数, 绘制一条会"动" 的余弦曲线
x =np.arange(0, 2*np.pi, 0.01)
#执行后返回Line2D列表
lines_lst = plt.plot(x, np.cos(x))
#取出第0根线,这是update函数需要改变的全局对象
line =lines_lst[0]
def init():
#初始化函数,每次新一轮动画开始时执行一次,不是必须的
    line.set_ydata(np.cos(x))
#最重要的更新动画函数,每帧自动重复调用
def update(n):
#print('帧号:', n)
    line.set_ydata(np.cos(x +n/10))
#每次传入的n值不同,更新Y轴数据,图像自动更新
ani = FuncAnimation(fig=plt.gcf(),
#创建动画对象,gcf()获取当前画布,作为动画的载体
func=update,
#设置更新函数
frames=100,
#如为单个整数,则数据序列为range(0,100)
init_func=init,
#设置初始化函数
interval=20, repeat=True)
#每帧间隔20毫秒, 默认值True, 动画一直循环
ani.save('img/cos_ani.gif', fps=10)
#保存为gif动画文件,每秒10帧
```

- **世界地图绘制**：使用 Cartopy 包，可与 Matplotlib 配合创建出版级地图，安装命令`pip install cartopy==0.23`（Cartopy0.23 适配 Matplotlib3.7，Cartopy0.22 适配 Matplotlib3.5），该包提供多种地图投影和地理信息处理方法。

## 第七章 Seaborn 统计数据可视化

### 核心知识点

1. **Seaborn 简介与绘图特点**
    - **与 Matplotlib 的关系**：基于 Matplotlib 的统计可视化库，专为统计数据展示设计，相比 Matplotlib，代码更简洁、可自动美化、支持语义映射、深度集成 Pandas
    - **主题样式**：有 5 种预设主题，默认`darkgrid`（灰色背景 + 网格线，适合数据对比），还包括`whitegrid`、`dark`、`white`、`ticks`（白色背景 + 无网格 + 刻度线，适合论文 / 报告），可设置中文字体：`sns.set_style('ticks', {'font.sans-serif' : 'SimHei' })`
    - **数据集加载**
        - 在线加载：`tips = sns.load_dataset('tips')`
        - 本地加载：`tips = sns.load_dataset('tips', data_home='/seaborndata')`
        - Pandas 读取 CSV：`tips = pd.read_csv('seaborndata/tips.csv')`
2. **核心绘图函数**
    
    Seaborn 绘图函数分 figure-level 图级和 axes-level 轴级，图级函数（如`catplot`）支持 kind/col/row/height/aspect，返回 FacetGrid 对象；轴级函数（如 barplot/boxplot）不支持多子图，返回 Matplotlib 的 Axes 对象。
    - **sns.catplot()**：默认图形类型为 strip，x 为分类变量（一对多关系）
    - **sns.relplot()**：支持 hue 参数分类着色，用于展示变量间相关关系，x/y 为连续变量（一对一关系），kind 参数可选'scatter'（默认，散点图）或 'line'（折线图）
3. **Seaborn 数据集加载**
    
    可通过`sns.load_dataset()`加载内置数据集（如 titanic/tips/iris）

## 数据处理相关知识点（Pandas、NumPy）

### 核心知识点

1. **Pandas 数据读写**
    - CSV 文件：`pd.read_csv()`/`pd.to_csv()`
    - Excel 文件：`pd.read_excel()`/`pd.to_excel()`
    - JSON 文件：`pd.read_json()`
2. **NumPy 随机数据生成**
    - 标准正态分布：`np.random.randn()`
    - 整数随机数：`np.random.randint()`
    - 网格坐标：`np.meshgrid()`
    - 序列生成：`np.linspace()`
3. **NumPy 随机种子**
    
    通过`np.random.seed()`可实现随机数结果重现