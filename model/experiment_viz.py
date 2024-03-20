import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

nr_scenarios=5
nr_runs=10

#open the csv's and save as dataframes
dataframes = {}

#collect average driving time per scenario
avg_driving_times = {}

#collect all trucks per scenario
all_trucks = pd.DataFrame()

for i in range(nr_scenarios): #save as dataframes and collect average driving times.
    dataframes[i] = pd.read_csv('../model/experiment/Scenario{}.csv'.format(i)) #save the csv
    avg_driving_times['scenario_{}'.format(i)] = dataframes[i].mean(axis=0).mean() #save the average driving times
    all_trucks = pd.concat([all_trucks,pd.melt(dataframes[i], value_vars=dataframes[i].columns.tolist(),
                             var_name='seed_name', value_name='scenario_{}'.format(i))['scenario_{}'.format(i)]], axis=1) #collect all trucks per scenario in a different column

print(avg_driving_times)

#boxplot per scenario, aggregated over all runs
ax=sns.boxplot(all_trucks)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
#plt.savefig('boxplot_per_scenario.png')
plt.show()
