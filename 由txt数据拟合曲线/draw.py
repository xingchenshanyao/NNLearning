from matplotlib import pyplot as plt
 
filename = 'epoch_map.txt'
step, y = [], []
# 相比open(),with open()不用手动调用close()方法
with open(filename, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。然后将每个元素中的不同信息提取出来
    lines = f.readlines()
    # j用于判断读了多少条，step为画图的X轴
    j = 0
    for line in lines:
        t = line
        step.append(j)
        j = j + 1
        y.append(float(t))
 
fig = plt.figure(figsize=(8, 5))  # 创建绘图窗口，并设置窗口大小
ax1 = fig.add_subplot(111)  # 将画面分割为1行1列选第一个
ax1.plot(step, y, 'blue', label='train mAP')  # 画dis-loss的值，颜色红
ax1.legend(loc='lower right')  # 绘制图例，plot()中的label值
ax1.set_xlabel('epoch')  # 设置X轴名称
ax1.set_ylabel('mAP0.5')  # 设置Y轴名称
plt.title('a mAP curve') #设置标题
plt.grid(linestyle=":", color="r")  # 绘制网格线
plt.savefig('logs/log_2023_05_04_bdd100k/epoch_mAP.png', dpi=100)

plt.show()  # 显示绘制的图
