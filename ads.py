import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from scipy.optimize import curve_fit


def merchandise_data():
    """
    Preprocesses population data for clustering and forecasting.
    Reads imports and exports of merchandise data from CSV files, drops rows \
        with missing data for 2021, 
    selects relevant columns, merges the imports  and exports data on \
        country name, renames columns, 
    normalizes data using StandardScaler, performs k-means clustering,\
        and adds cluster labels to the DataFrame.

    Returns
    -------
    df_2021 : pandas DataFrame
        A merged DataFrame of imports and exports population data for 2021,\
            along with a column of cluster labels.
    """
    imports = pd.read_csv("Merchandise imports (current US$).csv", skiprows=4)
    exports = pd.read_csv("Merchandise exports (current US$).csv", skiprows=4)

    # drop rows with nan's in 2021
    imports = imports[imports["2021"].notna()]
    exports = exports.dropna(subset=["2021"])

    # select relevant columns
    imports2021 = imports[["Country Name", "Country Code", "2021"]].copy()
    exports2021 = exports[["Country Name", "Country Code", "2021"]].copy()

    # merge male and female data on country name
    df_2021 = pd.merge(imports2021, exports2021, on="Country Name", \
                       how="outer")

    # drop rows with missing data
    df_2021 = df_2021.dropna()

    # rename columns
    df_2021 = df_2021.rename(columns={"2021_x":"Merchandise imports", \
                                      "2021_y":"Merchandise exports"})

    # normalize data using StandardScaler
    scaler = StandardScaler()
    df_cluster = scaler.fit_transform(df_2021[["Merchandise imports", \
                                               "Merchandise exports"]])
    # Determine the optimal number of clusters using the elbow method
    distortions = []
    max_clusters = 10  # Maximum number of clusters to consider
    for num_clusters in range(2, max_clusters + 1):
        kmeans = cluster.KMeans(n_clusters=num_clusters)
        kmeans.fit(df_cluster)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure()
    plt.plot(range(2, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Curve')
    plt.show()
    
    # Find the optimal number of clusters
    optimal_clusters = distortions.index(min(distortions)) + 2
    print("Optimal Number of Clusters:", optimal_clusters)
    
    # perform k-means clustering and add cluster labels to DataFrame
    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(df_cluster)
    df_2021["labels"] = kmeans.labels_

    return df_2021
def forecast_trade():
    """
    Reads merchandise data, fits an exponential function to the data,
    and plots the forecast with confidence range.
    """

    # Read data from CSV file
    trade = pd.read_csv("Merchandise trade (% of GDP).csv", skiprows=4)
    trade = trade.set_index('Country Name', drop=True)
    trade = trade.loc[:, '1960':'2021']
    trade = trade.transpose()
    trade = trade.loc[:, 'United States']
    df = trade.dropna(axis=0)

    # Create DataFrame with year and population columns
    df_trade = pd.DataFrame()
    df_trade['Year'] = pd.DataFrame(df.index)
    df_trade['Merchandise trade (% of GDP)'] = pd.DataFrame(df.values)

    # Fit exponential function to data
    def exponential(t, n0, r):
        """Calculates the exponential function with initial value n0 and
        growth rate r"""
        f = n0 * np.exp(r * (t - df_trade["Year"].min()))
        return f

    df_trade["Year"] = pd.to_numeric(df_trade["Year"])
    param, covar = curve_fit(exponential, df_trade["Year"],
                             df_trade["Merchandise trade (% of GDP)"],
                             p0=(1.2e12, 0.03), maxfev=10000)

    # Generate forecast using exponential function
    year = np.arange(1960, 2031)
    forecast = exponential(year, *param)

    # Calculate confidence interval
    stderr = np.sqrt(np.diag(covar))
    conf_interval = 1.96 * stderr
    upper = exponential(year, *(param + conf_interval))
    lower = exponential(year, *(param - conf_interval))
    
    # Generate forecast for the next 5 years
    future_years = np.arange(2022, 2027)
    forecast_future = exponential(future_years, *param)
# Plot the data, forecast, and confidence interval
    plt.figure()
    plt.plot(df_trade["Year"], df_trade["Merchandise trade (% of GDP)"],
             label="Historical Data")
    plt.plot(year, forecast, label="Forecast")
    plt.plot(future_years, forecast_future, label="Next 5 Years Forecast")
    plt.fill_between(year, upper, lower, color='purple', alpha=0.2,
                     label="95% Confidence Interval")
    plt.xlabel("Year")
    plt.ylabel("Merchandise trade (% of GDP)")
    plt.title("Exponential forecast for United States")
    plt.legend()
    plt.show()


df_2021 = merchandise_data()

# perform k-means clustering to obtain cluster centers
kmeans = cluster.KMeans(n_clusters=6)
kmeans.fit(df_2021[["Merchandise imports", "Merchandise exports"]])
centers = kmeans.cluster_centers_

# plot all clusters and centers in one plot
plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('tab10')
for i, label in enumerate(np.unique(df_2021["labels"])):
    plt.scatter(df_2021[df_2021["labels"] == label]["Merchandise imports"], \
                df_2021[df_2021["labels"] == label]["Merchandise exports"], \
                    10, label="Cluster {}".format(label), cmap=cm, alpha=0.7)
plt.scatter(centers[:,0], centers[:,1], 50, "k", marker="D", \
            label="Cluster centers")
plt.xlabel("Merchandise imports")
plt.ylabel("Merchandise exports")
plt.title("Kmeans Clustering")
plt.legend()
plt.show()

forecast_trade()
