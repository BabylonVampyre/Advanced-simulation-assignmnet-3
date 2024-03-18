from model import BangladeshModel
import pandas as pd
"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------
#Function to format possible scenario's
#It takes a name, a list of strings of the possible cataggories of bridges and a list of the chances of breakdown
#The chances need to be in percentage
def make_scenario(name,chance_breakdown):
    if not isinstance(chance_breakdown, list):
        raise TypeError("Please provide the chance of breakdown as a list")
    return [name,chance_breakdown]

#iniciate scenario's
S0 = make_scenario(0,[0,0,0,0])
S1 = make_scenario(1,[0,0,0,5])
S2 = make_scenario(2,[0,0,5,10])
S3 = make_scenario(3,[0,5,10,20])
S4 = make_scenario(4,[5,10,20,40])


# run time 5 x 24 hours; 1 tick 1 minute
#run_length = 7200

# run time 1000 ticks
run_length = 7200

seed_list = [1234567]
scenario_list = [S0]

number_of_runs = len(scenario_list)*len(seed_list)
counter = 0

#itterate through the seed and scenario lists and run the model for the run_length
for i in scenario_list:
    for seed_index, j in enumerate(seed_list):
        sim_model = BangladeshModel(scenario=i, seed=j)

        for k in range(run_length):
            sim_model.step()
        df = sim_model.datacollector.get_agent_vars_dataframe()
        # subset dataframe to only show the sinks and sourcesinks
        # because we want to collect what vehicles they removed and their driving time
        df = df[df['Driving time of cars leaving'].notnull()]
        df = df[df['Driving time of cars leaving'].str.len() != 0]
        #df.to_csv("../model/OutputModel.csv")
        df.to_csv(f"../model/experiment/Scenario{i[0]}_sim{seed_index}.csv")

        #print how far the full run is
        counter += 1
        print(counter, '/', number_of_runs, 'Done. Next up: scenario:', i[0], 'seed: ', j)




# Check if the seed is set
print("SEED " + str(sim_model._seed))

