
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

path='classify.json'
trans={
        "ori":"Original",
        "blur":"Gaussian Blur",
        "down":"Downsampling",
        "light-up":"Light Enhancement",
        "light-down":"Light Reduction"
    }

names=list(trans.keys())
trans = {
        "gwhd": "GWHD",
        "21": "2021",
        "22": "2022",
        "23": "2023"
    }
names = ["GWHD", '2021', '2022', '2023']
names=list(trans.keys())
def getData(path):

    with open(path) as f:
        data=json.load(f)
    return data
# print(data)
data=getData(path)
def getTarget_by_range(target,ran=list(range(100,900,100)),data=data,names=names):
    if isinstance(data,str):
        data=json.load(open(data))
        data=data[target]
        res={

            # ,names[4]:[]
        }
        for name in names:
            res[name]=[]
        # names=['gwhd','21','22','23']
        for r in ran:
            r=str(r)
            for i,name in enumerate(names):
                res[name].append(data[r][i])
        return res
    res={}
    ans={}
    for k,v in data.items():
        dataset_data=v
        # print(k)
        temp={}
        for k2,v2 in v.items():
            try:
                # print(k2)
                temp[k2]=v2[target]


            except:

                pass


    for idx in ran:
        idx=str(idx)
        v_=res[idx]
        for k3,v3 in v_.items():
            if ans.get(k3, None) is None:
                ans[k3] = []
            ans[k3].append(v3)

    return ans
def getTarget(target,data=data):
    if isinstance(data,str):
        data=json.load(open(data))
        return data[target]
    res={}
    for k,v in data.items():
        dataset_data=v
        # print(k)
        temp={}
        for k2,v2 in v.items():
            try:
                temp[k2]=v2[target]
            except:
                pass

        # res[k]=temp
        res[k] = [temp['gwhd'], temp['21'], temp['22'], temp['23']]
    return res
def set_matplotlib_style():
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14




plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
# plt.gca().set_facecolor('lightgray')
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.tight_layout()




def get_plot_parm_real_noise():
    # set_matplotlib_style()

    return dict(
        marker='^', linewidth=5.5,
        markersize=14,
        linestyle='--',

    )
def get_plot_parm_real():
    # set_matplotlib_style()

    return dict(
        marker='o',
        linewidth=3,
        markersize=11,
        linestyle='-',
        # linewidth=1.0
        # color='b'
    )

def getTickParams():
    return dict(
        axis='both', which='major', pad=5
    )


def fig_template(subplot, title, target, xlabel=None, ylabel=None, ylim=0.5, loc='lower right', data=r'#'
                 , trans=None, ytick=None, leg=None,colors=None,my_range=None,xtick=None,names=names):

    if leg is None:
        leg = dict(loc=loc, fontsize='x-large', )
    if trans is None:
        trans = {
            "gwhd": "GWHD",
            "21": "SDAU2021",
            "22": "SDAU2022",
            "23": "SDAU2023"
        }
    plt.subplot(subplot)

    plt.ylim(ylim)

    if colors is None:

        if len(trans)==4:

            colors=['red','blue','green','orange',]
        else:
            colors = [ 'blue', '#43CD80', "red" ,'orange',"#00BFFF"]
        print(colors)
    plt.title(title, fontsize=22,fontweight='bold',)
    if my_range is None:
        my_range = list(range(100, 900, 100))
    res = getTarget_by_range(target,
                             data=data,
                             ran=my_range,names=names
                             )

    param = get_plot_parm_real()
    idx=0
    for _ in range(len(trans)):
        if list(trans.keys())[_].lower().find('ori')!=-1:
            plt.plot(my_range, res[list(trans.keys())[_]], label=f'{list(trans.values())[_]}', color="blue",
                     **get_plot_parm_real_noise())
            idx+=1
            continue

        try:
            if True:
                print(res,list(trans.keys())[idx])
                plt.plot(my_range, res[list(trans.keys())[idx]],color=colors[idx], label=f'{list(trans.values())[idx]}',
                         **get_plot_parm_real())
        except:
            pass
            # ori
        idx += 1


    plt.tick_params(**getTickParams())
    if xtick is None:
        plt.xticks(fontsize=22,fontweight='medium',)
    else:
        plt.xticks(xtick,fontsize=22, fontweight='medium', )
    if ytick is None:
        plt.yticks( fontsize=22, fontweight='bold', )
    else:
        plt.yticks(ytick,fontsize=22, fontweight='bold', )

    plt.legend(**leg )
    ymax=ylim
    ymin=ylim
    xmax=max(my_range)
    xmin=min(my_range)

    if isinstance(ylim,list):
        ymax=ylim[-1]
        ymin=ylim[0]
    else:
        ymax=1
    ymax=ymax-ymin
    xmax=xmax-xmin
    print(ymax)
    def hspan(start,sep=0.3):
        plt.axvspan(start * xmax + xmin, (start+sep) * xmax + xmin, facecolor='#5F9EA0', alpha=0.08, hatch='', label='Striped Area')


    hspan(0.0)
    hspan(0.4)
    hspan(0.8)



class Noise:
    trans={
        "ori":"Original",
        "blur":"Gaussian Blur",
        "down":"Downsampling",
        "light-up":"Light Enhancement",
        "light-down":"Light Reduction"
    }
    data=r'#'
    names=list(trans.keys())
    def r2(self,sub=231):
        fig_template(
            subplot=sub,
            title="$R^{2}$",
            target='r2',
            xlabel="Data Size",
            ylim=0.25,
            data=self.data,
            ytick=np.arange(0.25,1.05,0.05),
            trans=self.trans
            ,names=self.names

        )
    def rmse(self,sub=232):
        fig_template(
            subplot=sub,
            title="$RMSE$",
            target='rmse',
            xlabel="Data Size",
            ylim=[0,30],
            data=self.data,
            # ytick=np.arange(0.25,1.05,0.05)
            leg=dict(loc="upper right", fontsize='x-large'),
            trans=self.trans,names=self.names

        )
    def mae(self,sub=233):
        fig_template(
            subplot=sub,
            title="$MAE$",
            target='mae',
            xlabel="Data Size",
            ylim=[0, 30],
            data=self.data,
            trans=self.trans,
            # ytick=np.arange(0.25,1.05,0.05)
            leg=dict(loc="upper right", fontsize='x-large'),names=self.names
        )
    def pre(self,sub=234):
        fig_template(
            subplot=sub,
            title="$Precision$",
            target='p',
            xlabel="Data Size",
            ylim=0.5,
            data=self.data,
            trans=self.trans,
            ytick=np.arange(0.5, 1.05, 0.05),names=self.names

        )
    def recall(self,sub=235):
        fig_template(
            subplot=sub,
            title="$Recall$",
            target='rec',
            xlabel="Data Size",
            ylim=0.5,
            data=self.data,
            trans=self.trans,
            ytick=np.arange(0.5, 1.05, 0.05),names=self.names

        )
        pass
    def f1(self,sub=236):
        fig_template(
            subplot=sub,
            title="$F1-score$",
            target='f1',
            xlabel="Data Size",
            ylim=0.5,
            data=self.data,
            trans=self.trans,
            ytick=np.arange(0.5, 1.05, 0.05),names=self.names

        )
        pass
    def main(self):
        self.r2()
        self.rmse()
        self.mae()
        self.pre()
        self.recall()
        self.f1()
class Datasize:
    def r2(self,sub=231):
        fig_template(
            subplot=sub,
            title="$R^{2}$",
            target='r2',
            xlabel="Data Size",
            ylim=0.25,
            data=r'#',
            ytick=np.arange(0.25,1.05,0.05)

        )
    def rmse(self,sub=232):
        fig_template(
            subplot=sub,
            title="$RMSE$",
            target='rmse',
            xlabel="Data Size",
            ylim=[0,30],
            data=r'#',
            # ytick=np.arange(0.25,1.05,0.05)
            leg=dict(loc="upper right", fontsize='x-large')

        )
    def mae(self,sub=233):
        fig_template(
            subplot=sub,
            title="$MAE$",
            target='mae',
            xlabel="Data Size",
            ylim=[0, 30],
            data=r'#',
            # ytick=np.arange(0.25,1.05,0.05)
            leg=dict(loc="upper right", fontsize='x-large')
        )
    def pre(self,sub=234):
        fig_template(
            subplot=sub,
            title="$Precision$",
            target='p',
            xlabel="Data Size",
            ylim=0.5,
            data=r'#',
            ytick=np.arange(0.5, 1.05, 0.05)

        )
    def recall(self,sub=235):
        fig_template(
            subplot=sub,
            title="$Recall$",
            target='rec',
            xlabel="Data Size",
            ylim=0.5,
            data=r'#',
            ytick=np.arange(0.5, 1.05, 0.05)

        )
        pass
    def f1(self,sub=236):
        fig_template(
            subplot=sub,
            title="$F1-score$",
            target='f1',
            xlabel="Data Size",
            ylim=0.5,
            data=r'#',
            ytick=np.arange(0.5, 1.05, 0.05)

        )
        pass
    def main(self):
        self.r2()
        self.rmse()
        self.mae()
        self.pre()
        self.recall()
        self.f1()
class RealAndSimulated:
    # colors=['r', 'orange', 'green', '#1E90FF' ]
    # colors=None
    colors=None
    def s_r2(self,sub=235):
        fig_template(
            subplot=sub,
            title="Simulated - $R^{2}$",
            target='r2',
            xlabel="Data Size",
            ylim=0.25,
            data=r'#',
            ytick=np.arange(0.25, 1.05, 0.1),
            colors=self.colors

        )
    def s_rmse(self,sub=236):
        fig_template(
            subplot=sub,
            title="Simulated - $RMSE$",
            target='rmse',
            xlabel="Data Size",
            ylim=[0,30],
            data=r'#',
            # ytick=np.arange(0.25, 1.05, 0.1)
            colors=self.colors,
            loc='upper right'
        )
    def s_mae(self,sub=235):
        fig_template(
            subplot=sub,
            title="Simulated - $MAE$",
            target='mae',
            xlabel="Data Size",
            ylim=[0,30],
            data=r'#',
            # ytick=np.arange(0.25, 1.05, 0.1)
            colors=self.colors,
            loc='upper right',
        )
    def t_r2(self,sub=231):
        fig_template(
            subplot=sub,
            title="Real - $R^{2}$",
            target='r2',
            xlabel="Data Size",
            ylim=0.25,
            data=r'#',
            ytick=np.arange(0.25, 1.1, 0.1),
            colors=self.colors,
            my_range=range(2500,0,-500),
            # loc='upper right'
            xtick=range(500,3000,500)
        )
    def t_rmse(self,sub=233):
        fig_template(
            subplot=sub,
            title="Real - $RMSE$",
            target='rmse',
            xlabel="Data Size",
            ylim=[0,30],
            data=r'#',
            # ytick=np.arange(0.25, 1.1, 0.1),
            colors=self.colors,
            my_range=range(2500, 0, -500),
            loc='upper right',
            xtick=range(500, 3000, 500)

        )
    def t_mae(self,sub=232):
        fig_template(
            subplot=sub,
            title="Real - $MAE$",
            target='mae',
            xlabel="Data Size",
            ylim=[0, 30],
            data=r'#',
            # ytick=np.arange(0.25, 1.1, 0.1),
            colors=self.colors,
            my_range=range(2500, 0, -500),
            loc='upper right',
            xtick=range(500, 3000, 500)


        )
    def main(self):
        self.s_r2()
        self.s_rmse()
        self.s_mae()
        self.t_r2()
        self.t_rmse()
        self.t_mae()
    pass

if __name__ == '__main__':
    #
    plt.figure(figsize=(28, 16))
    dirName = 'real-sim'
    # if os.path.exists(dirName) == False:
    #     os.makedirs(dirName)
    # main_noise()
    # RealAndSimulated().main()
    # Datasize().main()
    # Datasize().main()
    Noise().main()

    kind = ''

    fig = plt.gcf()
    # x_line = fig.get_size_inches()[0] / 2



    fig.patch.set_linewidth(6)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor("white")



    plt.tight_layout()
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    print(f"{kind} finished")
    plt.savefig(f'result/{dirName}.svg', dpi=600)

    plt.show()
