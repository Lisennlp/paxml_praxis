import matplotlib.pyplot as plt


xlabels = ['64', '128', '256', '512', '1024', '2048', '4096']
x = range(len(xlabels))
y = [0.0679, 0.0766, 0.0786, 0.0793, 0.0788, 0.076, 0.0688]
y1 = [0.0220, 0.02654, 0.02707, 0.02747, 0.02729, 0.02709, 0.0229]

coefficients = np.polyfit(x, y, 2) # 2表示2次项拟合
p = np.poly1d(coefficients)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = p(x_fit)

coefficients = np.polyfit(x, y1, 2) # 2表示2次项拟合
p = np.poly1d(coefficients)
x_fit1 = np.linspace(min(x), max(x), 100)
y_fit1 = p(x_fit1)

fig, ax = plt.subplots()

ax.scatter(xlabels, y)
ax.plot(x_fit, y_fit)

ax.scatter(xlabels, y1)
ax.plot(x_fit1, y_fit1)

# 在该点上标记其值
for index_to_label in range(len(x)):
    plt.annotate(f'{y[index_to_label]}',
                 (x[index_to_label], y[index_to_label]),
                 textcoords="offset points", xytext=(0,10), ha='center')

    plt.annotate(f'{y1[index_to_label]}',
                 (x[index_to_label], y1[index_to_label]),
                 textcoords="offset points", xytext=(0,10), ha='center')

ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels)
# ax.set_xlim(-1, 7)
ax.set_ylim(0, 0.088)

ax.set_title('Speed curve')
ax.set_xlabel('query block size')
ax.set_ylabel('speed(step/s)')
ax.legend(['v3-32', 'v3-32-fit', 'v3-128', 'v3-128-fit'], loc='right')

plt.show()
