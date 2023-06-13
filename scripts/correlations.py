#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv("../data/data.csv")

data = data.select_dtypes(include=[np.number])


plt.figure(figsize=(30, 16))
sns.heatmap(data.corr(), annot=True)
plt.savefig("../figures/correlations.pdf")
