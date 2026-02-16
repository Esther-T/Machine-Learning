Currently in progress.

I'm building a simple linear regression model that predicts the currency exchange from MYR to USD. I will be documenting my success and failures while evaluating the model

1. Understanding the Data
- The original dataset contains multiple indicators per month: avg, start, end, high, low
- For modeling, we only need one value per month. Common choices: avg (monthly average) and end (end-of-month rate)
- The dataset was also sorted by date before creating the lag features


2. Feature Engineering
- For features, I used lags which are previous month values used to predict the next month (apparently a strategy used in finance)
- I also used rolling mean which is the avergae of previous 3 months. In this case, it didn't seem to change the model's performance.

3. The first half of the code, I implemented linear regression from scratch using gradient descent to minimize the cost function and improving the model. Then the second half, I used sklearn.LinearRegression to predict the next month value.




