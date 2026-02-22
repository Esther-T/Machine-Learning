This is a simple linear regression model that predicts the currency exchange from MYR to USD. The following are main concepts/strategies that I used while building this model.

1. Understanding the Data
- The original dataset contains multiple indicators per month: avg, start, end, high, low
- For modeling, I only need one value per month. Common choices: avg (monthly average) and end (end-of-month rate)
- The dataset was also sorted by date before creating the lag features


2. Feature Engineering
- For features, I used lags which are previous month values used to predict the next month (apparently a strategy used in finance)
- I also used rolling mean which is the avergae of previous 3 months. In this case, it didn't seem to change the model's performance.

3. The first half of the code, I implemented linear regression from scratch using gradient descent to minimize the cost function and improving the model. Then the second half, I used sklearn.LinearRegression to predict the next month value. Using both methods allow me to learn to implement from scratch as well as to gain familiarity of the existing Python libraries.


I've integrated this model to an existing chatbot application that I built: https://esther-t.github.io/ShiroBot/

