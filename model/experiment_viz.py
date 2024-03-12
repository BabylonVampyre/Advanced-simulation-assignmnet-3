import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

nr_scenarios=5
nr_runs=10
#open the csv's and save as dataframes
df={} #collect the trucks and driving time per tick
for i in range(nr_scenarios): #scenario
    df[i] = pd.DataFrame()
    for j in range(nr_runs): #run
        df[i]=pd.concat([df[i],pd.read_csv('../model/experiment/Scenario{}_sim{}.csv'.format(i,j))['Driving time of cars leaving']])
    df[i].columns=['Driving time of cars leaving']

#collect per scenario all the driving times of all the runs
df2={}
for framenr in range(nr_scenarios): #iterate the dataframes
    df2[framenr]=np.array([]) #collect all the truck times per scenario
    for tick in df[framenr]['Driving time of cars leaving']: #iterate the entries, where each entry is a list of truck that leave at that tick
        extracted_list = ast.literal_eval(tick) #thx chatgpt
        for truck_time in extracted_list: #iterate over the trucks
            df2[framenr]=np.append(df2[framenr],truck_time[1]) #add the truck's time to the dataframe
    print(df2[framenr][0])

#plot for every scenario a density plot
plt.figure(figsize=(30, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("KDE Driving time of cars leaving per scenario", fontsize=18, y=0.95)


for scenario in range(1,nr_scenarios):
    ax = plt.subplot(2, 4, scenario)
    #ax.set_ylim(0,0.15) #if you want the same yaxis
    ax.set_xlim(0,2500) #if you want the same xaxis
    sns.kdeplot(df2[scenario], ax=ax)
    ax.set_title('Scenario {}'.format(str(scenario)))
    ax.set_xlabel("Driving time of cars leaving in minutes")
    plt.yticks(fontsize=7, rotation=45)

plt.show()