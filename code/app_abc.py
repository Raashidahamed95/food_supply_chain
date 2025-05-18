from flask import Flask, render_template, request
import pickle
import pandas as pd


import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import base64
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np



app = Flask(__name__)

data = pd.read_csv('crop_sales_data.csv')


os.makedirs("static/plots", exist_ok=True)

# Load pre-trained model
MODEL_PATH = 'models/sales_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    year = int(request.form['year'])
    month = int(request.form['month'])
    crop = request.form['crop']
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'Year': [year], 'Month': [month], 'Crop': [crop]})
    
    # Preprocessing (e.g., encoding)
    input_data['Crop'] = input_data['Crop'].map({'Rice': 0, 'Maize': 1, 'Wheat': 2})
    
    # Predict sales
    prediction = model.predict(input_data)[0]
    
    return render_template('results.html', prediction=prediction)


@app.route('/plot_ds')
def plot_ds():
    # Load the dataset
    data = pd.read_csv("crop_sales_data.csv")

    # Pie Chart: Proportion of Sales by Crop
    pie_path = "static/plots/pie_chart.png"
    sales_by_crop = data.groupby("Crop")["Sales (kg)"].sum()
    sales_by_crop.plot.pie(autopct='%1.1f%%', figsize=(6, 6), legend=True)
    plt.title("Sales Distribution by Crop")
    plt.savefig(pie_path)
    plt.close()

    # Bar Chart: Sales by Year
    bar_path = "static/plots/bar_chart.png"
    sales_by_year = data.groupby("Year")["Sales (kg)"].sum()
    sales_by_year.plot.bar(color='skyblue', figsize=(8, 5))
    plt.title("Total Sales by Year")
    plt.xlabel("Year")
    plt.ylabel("Sales (kg)")
    plt.savefig(bar_path)
    plt.close()

    # Line Plot: Year-wise Crop Sales
    line_path = "static/plots/line_chart.png"
    data.groupby("Year").sum().plot(y="Sales (kg)", figsize=(8, 5), marker="o")
    plt.title("Sales Trend Over Years")
    plt.xlabel("Year")
    plt.ylabel("Sales (kg)")
    plt.grid(True)
    plt.savefig(line_path)
    plt.close()

    # Pass plot paths to the HTML template
    return render_template(
        "index1.html",
        pie_chart=pie_path,
        bar_chart=bar_path,
        line_chart=line_path
    )


@app.route('/graph')
def graph():
    return render_template('graph.html')


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = pd.read_csv("dataset1.csv")

target_column = "Number of products sold"
features = ['Price', 'Availability', 'Stock levels', 'Lead times', 'Order quantities']

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target_column], test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



    
@app.route('/fsm')
def index5():
  
    return render_template('index5.html')

@app.route('/predicts', methods=['POST'])
def predicts():
    if request.method == 'POST':


        price = float(request.form['price'])
        availability = float(request.form['availability'])
        stock_levels = float(request.form['stock_levels'])
        lead_times = float(request.form['lead_times'])
        order_quantities = float(request.form['order_quantities'])

        user_input = pd.DataFrame([[price, availability, stock_levels, lead_times, order_quantities]],
                                    columns=features)

        try:
            prediction = rf_model.predict(user_input)[0]
            return render_template('index5.html', prediction=prediction)
        except Exception as e:
            return render_template('index5.html', error=str(e))
        




import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


@app.route('/lightgbm')
def testx():
        
        import pandas as pd
        import matplotlib.pyplot as plt

        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        
        data = pd.read_csv("dataset1.csv")
        print(data.head())

        print(data.info())


        #data.describe()

        #data.columns

        #data.dtypes

        missing_values = data.isnull().any(axis=1)
        print("Rows with missing values:")
        print(missing_values)
        duplicate_values = data[data.duplicated()]
        print("Duplicate Rows:")
        print(duplicate_values)
        data.dropna(axis=0,inplace=True)
        data.drop_duplicates(inplace=True)

        defect_rate_by_product = data.groupby("Product type")["Defect rates"].mean()
        defect_rate_by_product = data.groupby("Product type")["Defect rates"].mean()

        plt.figure(figsize=(10,6))
        defect_rate_by_product.plot(kind="bar",color='green')
        plt.title("Defect Rates by Product Type")
        plt.xlabel("Product Type")
        plt.ylabel("Mean Defect Rate")
        plt.xticks(rotation=45)
        plt.show()

        selected_columns = ["SKU","Lead times","Stock levels"]
        risk_data = data[selected_columns]
        risk_data["Risk Score"] = risk_data["Lead times"] * (1-risk_data["Stock levels"])
        risk_data = risk_data.sort_values(by="Risk Score", ascending=False)
        print("Top 15 high risk SKUs")
        print(risk_data.head(15))
        import numpy as np
        HoldingCost = 0.2
        D = data["Number of products sold"]
        S = data["Costs"]
        H = data["Number of products sold"] * HoldingCost
        EOQ = np.sqrt((2*S*D)/H)

        data["Current Order Quantity"] = data["Order quantities"]
        data["EOQ"] = EOQ.astype(int)
        comparison_columns = ["SKU","EOQ","Current Order Quantity"]
        print(data[comparison_columns])

        mean_revenue = data.groupby(['Customer demographics','Product type'])['Revenue generated'].mean().reset_index()
        sum_revenue = data.groupby(['Customer demographics','Product type'])['Revenue generated'].sum().reset_index()
        print("Mean Revenue for Each Customer Demographics")
        print(mean_revenue)
        print("Sum Revenue for Each Customer Demographics")
        print(sum_revenue)

        plt.figure(figsize=(10,6))
        plt.bar(mean_revenue["Customer demographics"] + '-' + mean_revenue["Product type"], mean_revenue["Revenue generated"])
        plt.xlabel("Customer Demographics - Product Type")
        plt.ylabel("Mean Revenue")
        plt.title("Mean Revenue by Customer Demographics & Product Type")
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout
        plt.show()

        plt.figure(figsize=(10,6))
        plt.bar(sum_revenue["Customer demographics"] + '-' + sum_revenue["Product type"], sum_revenue["Revenue generated"])
        plt.xlabel("Customer Demographics - Product Type")
        plt.ylabel("Sum Revenue")
        plt.title("Sum Revenue by Customer Demographics & Product Type")
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout
        plt.show()

        lead_times_column = "Lead times"
        transportation_modes_column = "Transportation modes"
        routes_column = "Routes"

        average_lead_time_by_mode = data.groupby(transportation_modes_column)[lead_times_column].mean().reset_index()

        best_transportation_mode = average_lead_time_by_mode.loc[average_lead_time_by_mode[lead_times_column].idxmin()]

        best_mode = data[data[transportation_modes_column]==best_transportation_mode[transportation_modes_column]]

        average_lead_time_by_route = data.groupby(routes_column)[lead_times_column].mean().reset_index()

        best_route = average_lead_time_by_route.loc[average_lead_time_by_route[lead_times_column].idxmin()]

        print("Average Lead Times by Transportation Mode:")
        print(average_lead_time_by_mode)
        print("The Best Transportation Mode (Shortest Average Lead Time):")
        print(best_transportation_mode)
        print("The Average Lead Times by Route within the Best Transportation Mode:")
        print(average_lead_time_by_route)
        print("The Best Routes (Shortest Average Lead Times) within the Best Transportation Mode:")
        print(best_route)

        target_column = "Number of products sold"
        features = ['Price','Availability','Stock levels','Lead times','Order quantities']

        x_train, x_test, y_train, y_test = train_test_split(data[features], data[target_column],
                                                            test_size=0.2, random_state=42)

        train_data = lgb.Dataset(x_train, label=y_train)

        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate':0.05,
            'feature_fraction': 0.9
        }

        num_round = 100
        bst = lgb.train(params, train_data, num_round)

        y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
        #x_test

        print("Forecasted Customer Demand:", y_pred)

        target_column = "Manufacturing costs"
        feature_column = "Production volumes"

        x = data[feature_column].values.reshape(-1,1)
        y = data[target_column].values

        scaler = MinMaxScaler()
        x_scaled = scaler.fit(x)

        x_train, x_test, y_train, y_test = train_test_split(data[feature_column], data[target_column],
                                                            test_size=0.2, random_state=42)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64,activation='relu',input_dim=1),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',loss='mean_square_error')

        print("x shape:", x.shape)
        print("y shape:", y.shape)

        min_production_volume = data["Order quantities"].min()
        max_production_volume = 1000
        step_size = 10

        cheapest_cost = float("inf")
        best_production_volume = None

        for production_volume in range(min_production_volume,max_production_volume + 1, step_size):
            normalized_production_volume = scaler.transform(np.array([[production_volume]]))
            predicted_cost = model.predict(normalized_production_volume)
            if production_volume == best_production_volume:
                best_cost = predicted_cost[0][0]
            if predicted_cost[0][0] >= 0:
                cheapest_cost = predicted_cost[0][0]
                best_production_volume = production_volume
        print("The Most Optimal Production Volume to Minimize Manufacturing Cost:", best_production_volume)
        print("The Cheapest Manufacturing Cost", cheapest_cost)

        target_column = "Number of products sold"
        features = ['Price','Availability','Stock levels','Lead times','Order quantities']

        num_folds = 5

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        mse_scores = []

        x_train, x_test, y_train, y_test = train_test_split(data[features], data[target_column],
                                                            test_size=0.2, random_state=42)

        for train_index, test_index in kf.split(data):
            train_data = data.loc[train_index, features]
            train_target = data.loc[train_index, target_column]
            test_data = data.loc[test_index, features]
            test_target = data.loc[test_index, target_column]

        train_data = lgb.Dataset(x_train, label=y_train)
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate':0.05,
            'feature_fraction': 0.9
        }

        num_round = 100
        bst = lgb.train(params, train_data, num_round)

        y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
        print("Forecasted Customer Demand:", y_pred)


        df = pd.read_csv('crop_sales_data.csv')

        # Drop the target column 'Revenue'
        X = df.drop(columns=['Revenue'])  # Features
        y = df['Revenue']  # Target

        # One-hot encode the categorical 'Crop' column
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        crop_encoded = encoder.fit_transform(X[['Crop']])

        # Convert to DataFrame and merge with numerical features
        crop_encoded_df = pd.DataFrame(crop_encoded, columns=encoder.get_feature_names_out(['Crop']))
        X = pd.concat([X.drop(columns=['Crop']), crop_encoded_df], axis=1)

        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train LightGBM model
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Calculate Accuracy as (1 - MAPE)
        accuracy = 100 - mape  # Convert to percentage format
    
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f'R² Score: {r2}')
        print(f'Accuracy: {accuracy:.2f}%')  # Display accuracy as a percentage

       



        # Example:
        import numpy as np
        from sklearn.metrics import confusion_matrix

        # Define bins (adjust as needed)
        bins = np.linspace(data[target_column].min(), data[target_column].max(), 5)

        # Bin the predictions and true values
        y_test_binned = np.digitize(y_test, bins)
        y_pred_binned = np.digitize(y_pred, bins)



        # Create the confusion matrix
        cm = confusion_matrix(y_test_binned, y_pred_binned)
        print(f"Confusion Matrix:\n{cm}")

        return render_template('abc.html' ,x=accuracy)



data = pd.read_csv('crop_sales_data.csv')

@app.route('/plot', methods=['POST'])
def plot():
    # Get input values
    start_year = int(request.form['start_year'])
    start_month = int(request.form['start_month'])
    end_year = int(request.form['end_year'])
    end_month = int(request.form['end_month'])
    crop = request.form['crop']

    # Filter data based on input
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))
    filtered_data = data[(data['Date'] >= f'{start_year}-{start_month}-01') & 
                         (data['Date'] <= f'{end_year}-{end_month}-01') & 
                         (data['Crop'] == crop)]
    
    # Aggregating sales and revenue
    agg_data = filtered_data.groupby('Date').agg({'Sales (kg)': 'sum', 'Revenue': 'sum'}).reset_index()

    # Generate plots
    img_pie = create_pie_chart(agg_data)
    img_bar = create_bar_chart(agg_data)
    img_line = create_line_chart(agg_data)

    # Future prediction
    future_sales, future_revenue = predict_future(agg_data)

    return render_template('plot.html', 
                           img_pie=img_pie, img_bar=img_bar, img_line=img_line, 
                           future_sales=future_sales, future_revenue=future_revenue)

def create_pie_chart(agg_data):
    # Pie chart
    fig, ax = plt.subplots()
    agg_data.groupby('Date')['Sales (kg)'].sum().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    return save_plot_to_base64(fig)

def create_bar_chart(agg_data):
    # Bar chart
    fig, ax = plt.subplots()
    agg_data.plot.bar(x='Date', y='Sales (kg)', ax=ax)
    return save_plot_to_base64(fig)

def create_line_chart(agg_data):
    # Line chart
    fig, ax = plt.subplots()
    agg_data.plot.line(x='Date', y='Revenue', ax=ax)
    return save_plot_to_base64(fig)

def predict_future(agg_data):
    # Time-series forecasting
    model_sales = ExponentialSmoothing(agg_data['Sales (kg)'], trend="add", seasonal=None).fit()
    model_revenue = ExponentialSmoothing(agg_data['Revenue'], trend="add", seasonal=None).fit()

    future_sales = model_sales.forecast(steps=3).tolist()
    future_revenue = model_revenue.forecast(steps=3).tolist()

    return future_sales, future_revenue

def save_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

if __name__ == '__main__':
    app.run(debug=True)
