import pandas as pd
from math import sqrt, pow
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# establish null and alternate hypothesis
print(
    """
Let µL = true mean final grade on a scale from 0-20 (with 0 being the lowest and 20 is the highest) of college students similar to those in the study who are NOT in a romantic relationship, in Portuguese schools.

Let µR = true mean final grade on a scale from 0-20 (with 0 being the lowest and 20 is the highest) of college students similar to those in the study who are in a romantic relationship, in Portuguese schools.

H0: µL - µR = 0
HA: µL - µR ≠ 0
"""
)

# read online data
data = pd.read_csv("dataset.csv")


# calculate sample statistics from dataframe
def sample_statistics(frame):
    g3_mean = frame["G3"].mean()
    g3_std = frame["G3"].std()
    return g3_mean, g3_std


ssize = 60
print(f"sample size: both {ssize}")

# random number generator with fixed seeds

# students in a relationship
r = data[data["romantic"] == "yes"].sample(n=ssize, random_state=13)
# students not in a relationship
nr = data[data["romantic"] == "no"].sample(n=ssize, random_state=17)

# mean and standard deviation for students in a relationship
r_mean, r_sd = sample_statistics(r)
print(f"students in relationship: x_bar={round(r_mean, 2)}, sigma_x={round(r_sd, 2)}")

# mean and standard deviation for students not in a relationship
nr_mean, nr_sd = sample_statistics(nr)
print(
    f"students not in relationship: x_bar={round(nr_mean, 2)}, sigma_x={round(nr_sd,2 )}"
)

# null hypothesis: assumed difference
mu_null = 0
# combined two-sample standard deviation
combined_s = sqrt(pow(r_sd, 2) / ssize + pow(nr_sd, 2) / ssize)
# calculated test-statistic
t_statistic = ((r_mean - nr_mean) - mu_null) / combined_s
# conservative degrees of freedom (min. sample size - 1)
df = ssize - 1
# probability of test statistic assuming null is true (2-tailed)
p_val = stats.t.sf(abs(t_statistic), df) * 2
# significance level
alpha = 0.05

# plot t-statistic on graph
x = np.linspace(-4, 4, 200)
y = stats.t.pdf(x, df=df)
plt.plot(x, y)
plt.fill_betweenx(
    y, x, abs(t_statistic), where=(x >= abs(t_statistic)), color="blue", alpha=0.3
)
plt.fill_betweenx(
    y, x, -abs(t_statistic), where=(x <= -abs(t_statistic)), color="blue", alpha=0.3
)
plt.text(-4, 0.3, f"t(df={df})", fontsize=16)

print("T=" + str(round(t_statistic, 2)))
print("P=" + str(round(p_val, 2)))

if p_val < alpha:
    print("Reject H0")
elif p_val >= alpha:
    print("Fail to reject H0")


plt.show()
