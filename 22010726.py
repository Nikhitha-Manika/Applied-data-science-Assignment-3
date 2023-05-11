#Import the required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec



#computing the death rate crude summary statistics and then ploting a histogram for the data in the 2019
# Read in the dataset
df = pd.read_csv('API_SP.DYN.CDRT.IN_DS2_en_csv_v2_5359308.csv', skiprows=4)

# Generate summary statistics
summary_stats = df.describe()

# Print summary statistics
print("Summary statistics:")
print(summary_stats)

# Create a histogram
plt.hist(df['2019'], bins=20)
plt.title('Death Rate in 2019')
plt.xlabel('Death Rate')
plt.ylabel('Frequency')

# Save the histogram in png format
plt.savefig('death_rate_histogram.png', dpi=300)

# Show the histogram
plt.show()


#computing the summary statistics for the pandemic details of the world and then plotting a histogram for the top5 epidemics with the highest death toll rate
# Read in the dataset
df_pandemic = pd.read_csv('pandemic_details_of_world.csv')

# Convert 'Death toll' column to numeric dtype
df_pandemic['Death toll'] = pd.to_numeric(df_pandemic['Death toll'], errors='coerce')

# Generate summary statistics for the dataset
summary_stats_pandemic = df_pandemic['Death toll'].describe()

# Print summary statistics for the dataset
print("Summary statistics for pandemic dataset:")
print(summary_stats_pandemic)

# Choose data only for five epidemics with highest death tolls
df_top_five = df_pandemic.nlargest(5, 'Death toll')

# Generate summary statistics for the top five epidemics
summary_stats_top_five = df_top_five['Death toll'].describe()

# Print summary statistics for the top five epidemics
print("\nSummary statistics for top five epidemics:")
print(summary_stats_top_five)

# Create a histogram for the Death Toll column of the top five epidemics
plt.hist(df_top_five['Death toll'], bins=20)
plt.title('Histogram of Death Toll for Top Five Pandemics')
plt.xlabel('Death Toll')
plt.ylabel('Frequency')
# Save the histogram in png format
plt.savefig('death_toll_top_five_histogram.png', dpi=300)
plt.show()


#computing the summary statistics for life expectancy and then creating a histogram for the year 2019
# Read in the dataset
df = pd.read_csv('API_SP.DYN.LE00.IN_DS2_en_csv_v2_5358385.csv', skiprows=4)

# Generate summary statistics
summary_stats = df.describe()

# Print summary statistics
print("Summary statistics:")
print(summary_stats)

# Create a histogram
plt.hist(df['2019'], bins=20)
plt.title('Life Expectancy in 2019')
plt.xlabel('Life Expectancy (years)')
plt.ylabel('Frequency')

# Save the histogram in png format
plt.savefig('life_expectancy_histogram.png', dpi=300)

# Show the histogram
plt.show()



#pie chart showing top five epidemic death toll recorded
# Read in the dataset
df = pd.read_csv('pandemic_details_of_world.csv')

# Convert 'Death toll' column to numeric dtype
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')

# Get the top 5 epidemics by death toll
top_epidemics = df.nlargest(5, 'Death toll')

# Create a pie chart of the death tolls for the top 5 epidemics
plt.pie(top_epidemics['Death toll'], labels=top_epidemics['Epidemics/pandemics'])

# Add a title to the pie chart
plt.title('Top 5 Epidemics by Death Toll')

# Save the pie chart as a .png file
plt.savefig('pie_chart.png')

# Show the pie chart
plt.show()



#Grouped bar graph showing life expectancy for 8 countries of choice for the years 2012-2020
# Load the data from the CSV file
df = pd.read_csv('API_SP.DYN.LE00.IN_DS2_en_csv_v2_5358385.csv', skiprows=4)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(['United States', 'China', 'Indonesia', 'India', 'Russian Federation', 'Brazil', 'Japan', 'Germany']), ['Country Name', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Set the figure size and create a new subplot
plt.figure(figsize=(10, 6))
ax = plt.subplot()

# Set the years and the number of bars per group
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
n_bars = len(years)

# Set the bar width and the offset between the groups
bar_width = 0.8 / n_bars
offset = bar_width / 2

# Set the colors for each year
colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#2b83ba', '#abdda4', '#ffffbf', '#fdae61']

# Set the x ticks to be the country names
x_ticks = df_countries.index

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar([j + offset + bar_width*i for j in range(len(x_ticks))], df_countries[year], width=bar_width, label=year, color=colors[i])

# Set the axis labels and title
ax.set_xlabel('Country')
ax.set_ylabel('Life expectancy (years)')
ax.set_title('Life Expectancy by Country and Year')

# Set the x ticks and labels
ax.set_xticks([j + 0.4 for j in range(len(x_ticks))])
ax.set_xticklabels(x_ticks, rotation=60)

# Add a legend
ax.legend()

# Save the bar chart in png format
plt.savefig('grouped_bar.png', dpi=300)

# Show the plot
plt.show()


#scatter plot showing death rate crude for 8 countries of choice for the years 2012-2020
# Load the data from the CSV file
df = pd.read_csv('API_SP.DYN.CDRT.IN_DS2_en_csv_v2_5359308.csv', header=2)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(['United States', 'China','Indonesia', 'India', 'Russian Federation', 'Brazil', 'Japan', 'Germany']), ['Country Name', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Create a scatter plot for each country
for country in df_countries.index:
    plt.scatter(df_countries.columns, df_countries.loc[country])

# Set the axis labels and title
plt.xlabel('Year')
plt.ylabel('Death Rate Crude')
plt.title('Death Rate Crude by Year for 8 Countries')

# Add a legend
plt.legend(df_countries.index)

# Save the scatter plot as a PNG image
plt.savefig('scatter_plot.png', dpi=300)

# Show the plot
plt.show()


# creating a dash
# Define the file paths for the saved images
death_rate_hist_file = 'death_rate_histogram.png'
death_toll_top_five_hist_file = 'death_toll_top_five_histogram.png'
grouped_bar_file = 'grouped_bar.png'
life_expectancy_hist_file = 'life_expectancy_histogram.png'
pie_chart_file = 'pie_chart.png'
scatter_plot_file = 'scatter_plot.png'

# Load the saved images as numpy arrays
death_rate_hist = imread(death_rate_hist_file)
death_toll_top_five_hist = imread(death_toll_top_five_hist_file)
grouped_bar = imread(grouped_bar_file)
life_expectancy_hist = imread(life_expectancy_hist_file)
pie_chart = imread(pie_chart_file)
scatter_plot = imread(scatter_plot_file)

# Set the grid using GridSpec()
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 2, figure=fig)

# Add subplots for your visualizations
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Remove the tick marks and labels from the subplots
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Add your saved visualizations to the subplots
ax1.imshow(death_rate_hist)
ax1.set_title('Death Rate Histogram')

ax2.imshow(death_toll_top_five_hist)
ax2.set_title('Death Toll Top Five Histogram')

ax3.imshow(grouped_bar)
ax3.set_title('Grouped Bar Chart')

ax4.imshow(life_expectancy_hist)
ax4.set_title('Life Expectancy Histogram')

ax5.imshow(pie_chart)
ax5.set_title('Pie Chart')

ax6.imshow(scatter_plot)
ax6.set_title('Scatter Plot')

# Add textboxes to explain each visualization
ax1.text(0.5, -0.1, "1. This is a histogram of death rates", transform=ax1.transAxes, ha='center', va='center')
ax2.text(0.5, -0.1, "2. This is a histogram of death toll for top five countries", transform=ax2.transAxes, ha='center', va='center')
ax3.text(0.5, -0.1, "3. This is a grouped bar chart comparing death rates and population\n by country for 8 chosen countries of choice for thr years 2012-2020", transform=ax3.transAxes, ha='center', va='center')
ax4.text(0.5, -0.1, "4. This is a histogram of life expectancy", transform=ax4.transAxes, ha='center', va='center')
ax5.text(0.5, -0.1, "5. This is a pie chart of Top 5 epidemics by death toll recorded", transform=ax5.transAxes, ha='center', va='center')
ax6.text(0.5, -0.1, "6. This is a scatter plot of 8 chosen countries \nfor the years 2012-2020", transform=ax6.transAxes, ha='center', va='center')

# Set overall title and adjust the layout
fig.suptitle('DASHBOARD OF GROUPED BAR CHART INDICATING LIFE EXPECTANCY, PIE CHART REPRESENTING TOP FIVE EPIDEMICS AND SCATTER PLOT REPRESENTING DEATH RATE AND THEIR HISTOGRAMS\n NAME: Merline Pricilla Peter \n STUDENT ID: 22010726\n', fontweight='bold')
fig.tight_layout()
plt.savefig('22010726.png', dpi=300)
plt.show()

