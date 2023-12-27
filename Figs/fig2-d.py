import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.patches as patches
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import os
# 创建一个图形对象和一个子图对象
fig, ax = plt.subplots(figsize=(5, 5))

# 构造数据
# np.random.seed(1000)
#输入文件路径
path=r"##"
#图片标题
title="9models"
def fig_template(subplot,dataPath,title,):
    plt.subplot(subplot)
    try:
        data = pd.read_csv(dataPath)
        x = data.detectLabel
        y = data.realLabel
    except:
        data = pd.read_csv(dataPath,header=2,sep=' ')
        x = data.detectLabel
        y = data.realLabel
    x,y=y,x

    plt.xlim([0,120])
    plt.ylim([0,120])


    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]


    dists = np.abs(m*x - y + c) / np.sqrt(m**2 + 1)

    max_size = 100
    min_size = 10
    size = (max_size - min_size) * (1 - dists / dists.max()) + min_size

    # 计算R2和RMSE和MAE
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    y_pred = model.predict(x.values.reshape(-1, 1))
    r2 = r2_score(y, y_pred).round(3)

    rmse = np.sqrt(mean_squared_error(x, y)).round(3) # 计算RMSE
    mae = mean_absolute_error(x, y).round(3)
    box = {
        'facecolor': 'white',
        'edgecolor': 'black',
        'boxstyle': 'round',
    }
    font_size_ann = 18
    r2_text = plt.annotate("R2 = {:.3f}".format(r2), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=font_size_ann, color='#449945')
    rmse_text = plt.annotate("RMSE = {:.3f}".format(rmse), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=font_size_ann, color='#83639f') # 添加RMSE的注释
    mae_text = plt.annotate("MAE = {:.3f}".format(mae), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=font_size_ann, color='#1f70a9')

    if c<0:
        plt.annotate(f"y = {m:.2f}x{c:.2f}", xy=(0.05, 0.78), xycoords='axes fraction', fontsize=font_size_ann, color='#c22f2f')
    else:
        plt.annotate(f"y = {m:.2f}x+{c:.2f}", xy=(0.05, 0.78), xycoords='axes fraction', fontsize=font_size_ann, color='#c22f2f')
    # plt.text(0.05, 0.78, f"y = {m:.2f}x{c:.2f}", transform=ax.transAxes, fontsize=14, verticalalignment='bottom', color='#c22f2f')


    colors = np.where(y > m*x+c, '#04ae54', '#Fc8005')
    a_c='#Fc8005'
    b_c='#04ae54'
    c_c='#0000CD'
    a_c,b_c=b_c,a_c
    nx=x[x!=y]
    ny=y[x!=y]
    ex=x[x==y]
    ey=y[y==x]
    print(ex.shape,ey.shape)
    colors = np.where(ny > nx , a_c,b_c)
    colors_2=np.where(ex==ey,b_c,b_c)
    ndists = np.abs(m * nx - ny + c) / np.sqrt(m ** 2 + 1)
    nsize = (max_size - min_size) * (1 - ndists / ndists.max()) + min_size
    ndists_2 = np.abs(m * ex - ey + c) / np.sqrt(m ** 2 + 1)
    nsize_2 = (max_size - min_size) * (1 - ndists_2 / ndists_2.max()) + min_size
    plt.plot(range(1,120), range(1,120), linestyle='--', color='black')
    plt.scatter(nx, ny, alpha=0.5, c=colors, s=nsize,)
    plt.scatter(ex, ey, alpha=0.5, c=colors_2, s=nsize_2, )


    # plt.xlabel('Truth of Wheat Ears',fontsize=15)
    # plt.ylabel('Predict value of Wheat Ears',fontsize=15)

    plt.title(title,fontsize=24,loc="center")


    scatter1 = plt.scatter([], [], c=b_c, alpha=0.5, s=100, label='Prediction$\leq    $Truth')
    scatter2 = plt.scatter([], [], c=a_c, alpha=0.5, s=100, label='Prediction$\greater$Truth')
    plt.scatter([], [], alpha=0, s=0, label="R2 = {:.3f}, RMSE = {:.3f}".format(r2, rmse))


    legend = plt.legend(handles=[scatter2, scatter1], loc='lower left', bbox_to_anchor=(0.55, 0.0))
    legend.prop.set_size(18)
    for text in legend.get_texts():
        text.set_fontsize(15)

    # colors = np.where(y > x, '#04ae54', '#fcce05')
    # plt.scatter(x, y, alpha=0.5, c=colors, s=size)

    # 绘制回归线
    regression_line_x = np.array([min(x), max(x)])
    regression_line_y = m * regression_line_x + c
    plt.plot(regression_line_x, regression_line_y, color='#1f70a9', linestyle='-', linewidth=3)

    ax.plot([0,120], [0,120], color='BLACK', linestyle='--', dashes=(5, 5))

    # ax.tick_params(axis='x', width=2, length=6)
    # ax.tick_params(axis='y', width=2, length=6)

    plt.xticks(fontsize=15,fontweight='medium',)
    plt.yticks(fontsize=15,fontweight='medium',)

    # ax.axhline(y=1, color='BLACK', linewidth=3)
    # ax.axvline(x=1, color='BLACK', linewidth=3)


    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(1)
if __name__ == '__main__':
    fig, axs = plt.subplots(3, 3, figsize=(20,20))
    dir=r'##'
    path=[os.path.join(dir,x) for x in sorted(os.listdir(dir))]
    titles= [
    "Faster-RCNN",
    "YOLOv7",
    "YOLOv8",
    "CenterNet",
    "SSD",
    "RetinaNet",
    "EfficientDet",
    "Deformable-DETR",
    "DINO"
]
    for i,p in enumerate(path):
        title_=titles[i]
        fig_template(
            subplot=330+i+1,
            title=title_,
            dataPath=p
        )
    plt.subplots_adjust(wspace=0.18, hspace=0.18)
    width, height = fig.get_size_inches()
    v_line_pos = width / 3
    h_line_pos = height / 3

    for i in range(1, 3):

        line = Line2D([i * v_line_pos, i * v_line_pos], [0, height], transform=fig.dpi_scale_trans, color='#00008B',
                      linestyle='--',linewidth=0.6)
        fig.lines.append(line)


        line = Line2D([0, width], [i * h_line_pos, i * h_line_pos], transform=fig.dpi_scale_trans, color='#00008B',
                      linestyle='--',linewidth=0.6)
        fig.lines.append(line)
    fig.patch.set_linewidth(6)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    plt.savefig(fr'D:\Desktop\wheat_moni\yolov7-main_1\my_util\17 -不同数据量不同指标\data\result/{title}.svg', dpi=600)
    plt.show()

