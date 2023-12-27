import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.patches as patches
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import os
fig, ax = plt.subplots(figsize=(6, 6))
def fig_template(dataPath,title=None,):
    # plt.subplot(subplot)
    try:
        data = pd.read_csv(dataPath)
        x = data.detectLabel
        y = data.realLabel
    except:
        data = pd.read_csv(dataPath,sep=' ')
        x = data.detectLabel
        y = data.realLabel
    x,y=y,x

    # plt.xlim([0,120])
    # plt.ylim([0,120])
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.xticks(range(0,25,5))
    plt.yticks(range(0, 25, 5))


    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    dists = np.abs(m*x - y + c) / np.sqrt(m**2 + 1)

    max_size = 100
    min_size = 10
    size = (max_size - min_size) * (1 - dists / dists.max()) + min_size


    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    y_pred = model.predict(x.values.reshape(-1, 1))
    r2 = r2_score(y, y_pred).round(3)

    rmse = np.sqrt(mean_squared_error(x, y)).round(3)
    mae = mean_absolute_error(x, y).round(3)
    box = {
        'facecolor': 'white',
        'edgecolor': 'black',
        'boxstyle': 'round',
    }
    font_size_ann = 15
    r2_text = plt.annotate("R2 = {:.3f}".format(r2), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=font_size_ann, color='#449945')
    rmse_text = plt.annotate("RMSE = {:.3f}".format(rmse), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=font_size_ann, color='#83639f') # 添加RMSE的注释
    mae_text = plt.annotate("MAE = {:.3f}".format(mae), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=font_size_ann, color='#1f70a9')

    if c<0:
        plt.annotate(f"y = {m:.2f}x{c:.2f}", xy=(0.05, 0.78), xycoords='axes fraction', fontsize=font_size_ann, color='#c22f2f')
    else:
        plt.annotate(f"y = {m:.2f}x+{c:.2f}", xy=(0.05, 0.78), xycoords='axes fraction', fontsize=font_size_ann, color='#c22f2f')

    # colors = np.where(y > m*x+c, '#04ae54', '#Fc8005')
    a_c = '#Fc8005'
    b_c = '#04ae54'
    a_c,b_c=b_c,a_c
    colors = np.where(y > x, a_c,b_c)
    plt.plot(range(0,120), range(0,120), linestyle='--', color='black')
    plt.scatter(x, y, alpha=0.5, c=colors, s=size,)


    plt.xlabel('Ground Truth',fontsize=16)
    plt.ylabel('Prediction',fontsize=16)


    # plt.title(title,fontsize=24,loc="center")


    scatter1 = plt.scatter([], [], c=b_c, alpha=0.5, s=100, label='Prediction$\leq    $Truth')
    scatter2 = plt.scatter([], [], c=a_c, alpha=0.5, s=100, label='Prediction$\greater$Truth')
    plt.scatter([], [], alpha=0, s=0, label="R2 = {:.3f}, RMSE = {:.3f}".format(r2, rmse))


    legend = plt.legend(handles=[scatter2, scatter1], loc='lower right',)
    legend.prop.set_size(12)  # 设置图例字体大小为14
    for text in legend.get_texts():
        text.set_fontsize(15)  # 设置字体大小为12

    # colors = np.where(y > x, '#04ae54', '#fcce05')
    # plt.scatter(x, y, alpha=0.5, c=colors, s=size)


    regression_line_x = np.array([min(x), max(x)])
    regression_line_y = m * regression_line_x + c
    plt.plot(regression_line_x, regression_line_y, color='#1f70a9', linestyle='-', linewidth=3)

    ax.plot([0,120], [0,120], color='BLACK', linestyle='--', dashes=(5, 5))

    plt.xticks(fontsize=15, fontweight='medium', )
    plt.yticks(fontsize=15, fontweight='medium', )
    # ax.tick_params(axis='x', width=2, length=6)
    # ax.tick_params(axis='y', width=2, length=6)



    # 增粗X轴和Y轴相对着的边界线
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
if __name__ == '__main__':
    data_paths=[
        "#"
    ]
    datapath =r"#"
    title = os.path.split(datapath)[1].split('.')[0]
    fig_template(
        dataPath=datapath
    )
    plt.savefig(fr'#/{title}.svg', dpi=500)
    plt.show()
