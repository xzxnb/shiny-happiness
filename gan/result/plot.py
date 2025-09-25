import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件中的两个表格
# sheet_name=0 表示第一个表格，sheet_name=1 表示第二个表格
# 如果表格有表头，可添加 header=0 参数（默认）；如果没有表头，用 header=None
df1 = pd.read_excel('color1.xlsx', sheet_name=0, header=None)  # 第一个表格
df2 = pd.read_excel('color1.xlsx', sheet_name=1, header=None)  # 第二个表格
df3 = pd.read_excel('color1.xlsx', sheet_name=2, header=None)
# 提取数据
x1 = df1.iloc[1:, 0]  # 第一个表格的第一列作为x轴
y1 = pd.to_numeric(df1.iloc[1:, 1], errors='coerce')  # 第一个表格的第二列作为第一条线
x2 = df2.iloc[1:, 0]
y2 = pd.to_numeric(df2.iloc[1:, 1], errors='coerce')   # 第二个表格的第二列作为第二条线
x3 = df3.iloc[1:, 0]
y3 = pd.to_numeric(df3.iloc[1:, 1], errors='coerce')
# 创建画布并绘图
plt.figure(figsize=(10, 6))  # 设置图表大小

# 绘制两条折线
plt.plot(x1, y1, label='bayes', marker='o', linestyle='-', color='b')
plt.plot(x2, y2, label='domain10', marker='s', linestyle='--', color='r')
plt.plot(x3, y3, label='domain15', marker='*', linestyle='-', color='g')
# 添加图表元素
plt.xlabel('TV', fontsize=12)
plt.ylabel('val_acc', fontsize=12)
plt.title('color1', fontsize=14)
plt.legend(fontsize=10)  # 显示图例
plt.grid(alpha=0.3)  # 添加网格线
plt.tight_layout()  # 自动调整布局

# 显示图表
plt.show()