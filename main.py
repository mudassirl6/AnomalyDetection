import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




# Load your data and model implementation
def main():
    # Streamlit Interface
    st.title("Customer Purchase Behavior Anomaly Detection")
    st.write("This app uses Isolation Forest to detect anomalies in customer purchase behavior.")
    data = pd.read_csv('customerPurchase.csv')
    st.write("Data Preview:")
    st.write(data.head())
    data = cleandata(data)
    data['Outlier'] = 0
        # Train Model
    st.subheader("Please select the model parameters as required")
    contamination = st.slider("Contamination (Anomaly Proportion)", 0.01, 0.5, 0.1)
    n_estimators = st.slider("Number of Trees", 50, 500, 100)
    if st.button("Train Model"):
        model,data_pred,outliers = train_isolation_forest(data, n_estimators=n_estimators, contamination=contamination)
        st.write("Anomaly Detection Complete")
        st.write(data.head())

        # Plotting
        st.subheader("Anomaly Detection Plot")
        st.write("Pair Plots of some important features which contribute to more in detecting anomalies")
        plot_anomalies(data)
        st.pyplot(pca(data))
        st.write("Anomalies are detected in the data. Green points are normal data points, while red points are anomalies.")
        st.write("The insights from the model is provided in the above chart...")
        st.write(data.describe(include='all'))
        st.write("Please tweak")
        st.write(f"The size of outliers are {len(outliers)}")
    

# Function to train Isolation Forest
def train_isolation_forest(data, n_estimators=100, contamination=0.1):    
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    data_pred = model.fit_predict(data)
    data['Outlier'] = data_pred
    outliers = data[data['Outlier'] == -1]
    normal_points = data[data['Outlier'] == 1]
    return model, data_pred,outliers
    
# Plotting function
def plot_anomalies(data):
    fig, ax = plt.subplots()
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='Outlier', data=data, ax=ax)
    st.pyplot(fig)
    fig1 = sns.pairplot(data,hue='Outlier',vars=['Age','AnnualIncome','PurchaseAmount'],palette={1:'green',-1:'red'})
    st.pyplot(fig1)


def cleandata(df):
    categorical_columns = ['Gender','ProductCategory']
    numerical_columns = ['Age','Gender','AnnualIncome','PurchaseAmount','PurchaseFrequency',]
    # calculate the percentage of missing values..
    proportion = df.isnull().mean()
    for col in df.columns:
        if df[col].isnull().mean() < 0.05:
            df.dropna(subset=[col],inplace=True)
        
    #since the customerid is not neccessary for outlier detection we can drop it..
    df.drop(columns=['CustomerID'],axis=1,inplace=True)
    #Performing Categorical Encoding..
    df = pd.get_dummies(df,columns=categorical_columns,drop_first=True,dtype=int)

    # Convert 'LastPurchaseDate' to a numerical feature representing days since the last purchase
    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], format="%d-%m-%Y")
    latest_date = df['LastPurchaseDate'].max()
    df['DaysSinceLastPurchase'] = (latest_date - df['LastPurchaseDate']).dt.days
    df.drop(columns=['LastPurchaseDate'], inplace=True)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def pca(df_scaled):
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=pca_transformed,columns = ['pca1','pca2'])
    plt.figure(figsize=(10, 6))
    pca_df['outlier'] = df_scaled['Outlier']
    pca_outlier = pca_df[pca_df['outlier'] == -1]
    pca_inlier = pca_df[pca_df['outlier'] == 1]
    plt.scatter(x = pca_inlier['pca1'], y = pca_inlier['pca2'], c='green', label='Normal Data')
    plt.scatter(x = pca_outlier['pca1'], y = pca_outlier['pca2'], c='red', label='Outlier Data')


    plt.title('PCA Visualization of Isolation Forest Anomalies')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    


        
if __name__ == "__main__":
    main()


