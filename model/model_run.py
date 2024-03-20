from model import BangladeshModel
import pandas as pd
"""
    Run simulation
    Print output at terminal
"""
# ---------------------------------------------------------------
#iniciate scenario's
S0 = [0, 0, 0, 0]
S1 = [0, 0, 0, 5]
S2 = [0, 0, 5, 10]
S3 = [0, 5, 10, 20]
S4 = [5, 10, 20, 40]


# run time 5 x 24 hours; 1 tick 1 minute
#run_length = 7200

# run time 1000 ticks
run_length = 7200

seed_list = [1234567, 1234568, 1234569, 1234560, 1234561, 1234562, 1234563, 1234564, 1234565, 1234566]
scenario_list = [S0, S1, S2, S3, S4]
#seed_list = [1234567]
#scenario_list = [S0]

#create variables to keep track of the total runs and how much is done
number_of_runs = len(scenario_list)*len(seed_list)
run_counter = 0
scenario_counter = 0

#itterate through the seed and scenario lists and run the model for the run_length
for scenario_index, i in enumerate(scenario_list):
    #make an empty dataframe to save all runs of a single scenario
    df_scenario = pd.DataFrame()
    for seed_index, j in enumerate(seed_list):
        sim_model = BangladeshModel(scenario=i, seed=j)

        for k in range(run_length):
            sim_model.step()

        #call the dataframe of the single run and change the column names to reflect what seed it used
        df_scenario_singlerun = sim_model.df_driving_time
        df_scenario_singlerun.columns = ['Driving_Time' + str(j)]
        #merge the single run data frame with the base dataframe to save all runs of a single scenario in one DF
        df_scenario = pd.concat([df_scenario, df_scenario_singlerun], axis=1)

        # print how far the full run is
        run_counter += 1
        print(run_counter, '/', number_of_runs, 'Done. Next up: scenario:', scenario_counter, 'seed: ', j)
    scenario_counter += 1
    df_scenario.to_csv(f"../model/experiment/Scenario{scenario_index}.csv", index=False)
