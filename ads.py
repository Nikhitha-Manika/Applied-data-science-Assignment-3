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
