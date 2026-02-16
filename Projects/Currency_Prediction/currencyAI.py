import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred)
    margin = 0.05 * (max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min()))
    lims = [min(y_true.min(), y_pred.min()) - margin,
            max(y_true.max(), y_pred.max()) + margin]
    ax.plot(lims, lims, '--', label='Perfect prediction')
    ax.set_xlabel('Actual Target')
    ax.set_ylabel('Predicted Target')
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def f(x, theta):
    return x @ theta.reshape(-1,1)


def mse(x, y, theta):
    diff = f(x, theta) - y
    return np.mean((f(x, theta) - y) ** 2) # this will just return 1 number


def grad(x, y, theta):
    m = x.shape[0]
    diff = f(x, theta) - y
    return (x.T @ diff).ravel() / m
    
url = 'https://storage.data.gov.my/finsector/exr/monthly.parquet'
df = pd.read_parquet(url)

#df.plot(x="date", y="USD", figsize=(10,5))
#plt.show()

df_monthly = df[df["indicator"] == "avg"].copy()

df_monthly["date"] = pd.to_datetime(df_monthly["date"], dayfirst=True)

df_monthly = df_monthly.sort_values("date").reset_index(drop=True)

df_monthly["lag1"] = df_monthly["USD"].shift(1)
df_monthly["lag2"] = df_monthly["USD"].shift(2)
df_monthly["lag3"] = df_monthly["USD"].shift(3)
df_monthly["rolling_mean"] = df_monthly["USD"].rolling(3).mean().shift(1) # doesn't seem to do much to do the model

df_monthly = df_monthly.dropna()

x = df_monthly[["lag1", "lag2", "lag3", "rolling_mean"]]
y = df_monthly["USD"]

# split the data to train and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=False)
n_train = x_train.shape[0]
n_test = x_test.shape[0]

# convert to numpy, I prob dont need it since I have multiple features now
#x_train = x_train.values.reshape(-1, 1)
#x_test = x_test.values.reshape(-1, 1)

# convert to numpy
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

bias_train = np.ones([n_train, 1])
bias_test = np.ones((n_test, 1))
x_train_b = np.concatenate([x_train_s, bias_train], axis=1)
x_test_b = np.concatenate([x_test_s, bias_test], axis=1)

lr = 0.1
n_iters  = 50
theta = np.zeros(x_train_b.shape[1])

# training the model with gradient descent
for iter in range(n_iters):
    theta -= lr * grad(x_train_b, y_train, theta)
    if iter % 10 == 0 or iter == n_iters - 1:
        train_mse = mse(x_train_b, y_train, theta)
        print(f"Iter {iter:4d} | Train MSE: {train_mse:.2f}")

# evaluate the model
print(f"  Train MSE: {mse(x_train_b, y_train, theta):.2f}")
print(f"  Test  MSE: {mse(x_test_b, y_test, theta):.2f}")

y_train_pred = f(x_train_b, theta)
y_test_pred = f(x_test_b, theta)
actual_vs_predicted(y_test_pred, y_test)

# the following is for prediction. To keep it simple, 
# I'll just use the sklearn linear regression

print(df_monthly[["date", "USD"]].tail(10))

last_row = df_monthly.iloc[-3:]
lag1 = last_row["USD"].iloc[-1]
lag2 = last_row["USD"].iloc[-2]
lag3 = last_row["USD"].iloc[-3]
rolling_mean = df_monthly["USD"].iloc[-3:].mean()

print(lag1, " ", lag2, " ", lag3)
model = LinearRegression()
model.fit(x_train_b, y_train)
x_next_raw = np.array([[lag1, lag2, lag3, rolling_mean]])

# normalize the features
scaler = StandardScaler()
x_next = scaler.fit_transform(x_next_raw)

# add bias
bias = np.ones((1,1))
x_next_b = np.concatenate([x_next, bias], axis=1)

y_next = model.predict(x_next_b)

print("Predicted MYR to USD for next month:", y_next[0])

