import math

import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

def get_r_s(X,Y):
    y_bar=np.mean(X)
    y_m=Y
    y=X
    up=np.sum((y-y_m)**2)
    down=np.sum((y-y_bar)**2)
    return 1-(up/down)


def computeCorrelation(X, Y):

    x=Y
    y=X
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    y_pred = model.predict(x.values.reshape(-1, 1))

    r2 = r2_score(y, y_pred).round(5)
    print("r2:", r2)


def main(txtPath):
    owPath = txtPath
    ow = pd.read_csv(owPath, sep=' ', header=2)
    computeCorrelation(ow['realLabel'], ow['detectLabel'])
