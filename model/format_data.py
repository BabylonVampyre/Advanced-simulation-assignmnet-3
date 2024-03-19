import numpy as np
import pandas as pd

# To help with printing/debugging
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 10)


def extract_data():
    """
    Extracts the data from the Excel file and removes all unnecessary information, and returns a dataframe.
    """
    # Assuming _roads3.csv is in the current directory
    file_path = "../data/BMMS_overview.xlsx"
    # Read the CSV file into a DataFrame
    df_read = pd.read_excel(file_path, header=0)  # header=0 means use the first row as column names
    # Select all rows where the column "road" is equal to "N1"
    # bridge_df = df_read[df_read['road'] == 'N107']
    bridge_df = df_read[df_read['road'].str.startswith(('N1', 'N2'))]
    bridge_df = bridge_df[bridge_df['road'] != 'N106']
    # bridge_df = df_read[df_read['road'].isin(['N1', 'N2'])]

    # List of column names to remove
    columns_to_remove = ['type', 'roadName', 'structureNr', 'width', 'constructionYear', 'spans', 'zone',
                         'circle', 'division', 'sub-division', 'EstimatedLoc']
    # Drop the specified columns
    bridge_df = bridge_df.drop(columns=columns_to_remove)

    # Identify roads that are long enough
    valid_roads = bridge_df.groupby('road')['chainage'].max() > 25
    # print("valid roads\n", valid_roads)

    # Filter the DataFrame to keep only the records where the road is in valid_roads
    df_filtered = bridge_df[bridge_df['road'].isin(valid_roads.index[valid_roads])]

    return df_filtered


def sort_and_remove_duplicates(df):
    # print("df at the begining of sort_and_remove_dupes\n", df)
    """
    This method sorts the dataframe based on the chainage, and then removes any duplicates from those columns
    """
    ordered_df = df.sort_values(by=['road', 'chainage'])

    # print(ordered_df.head(30))
    # Define custom aggregation functions
    aggregations = {
        'condition': 'max',  # Keep the worst grade
        'length': 'mean',  # Take the average
        'road': 'first',
        'LRPName': 'first',
        'chainage': 'first',
        'lat': 'mean',
        'lon': 'mean',
        'name': 'first'
    }
    # Apply groupby with custom aggregations
    dropped_df = ordered_df.groupby(['LRPName', 'road']).agg(aggregations)
    dropped_df['name'] = (dropped_df['name']
                          .str.lower()
                          .str.replace('r', 'l')
                          .str.replace(' ', '')
                          .str.replace('.', ''))

    dropped_df.reset_index(drop=True, inplace=True)
    dropped_df = dropped_df.groupby(['name', 'road']).agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    dropped_df = dropped_df.groupby(['chainage', 'road']).agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    dropped_df.drop(columns=['name'], inplace=True)
    sorted_df = dropped_df.sort_values(by=['road', 'chainage'])
    sorted_df.reset_index(drop=True, inplace=True)
    return sorted_df



def add_modeltype_name(df):
    """
    This method adds a modeltype of bridge, renames the LRPName to id, and adds a name for each bridge
    """
    # Label all bridges as a bridge
    df['model_type'] = 'bridge'
    # Add a column called 'name' filled with 'link' and a number from 1 to n
    df['name'] = 'bridge ' + (df.index + 1).astype(str)
    return df


def reorder_columns(df):
    """
    This method reorders the column so that they match the demo csv files.
    """
    # Define the desired column order
    desired_column_order = ['road', 'model_type', 'name', 'lat', 'lon', 'length', 'chainage', 'condition']
    # Reassign the DataFrame with the desired column order
    df = df[desired_column_order]
    return df


def create_source_sink(roads):
    """
    This method makes a source and a sink dataframe
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv("../data/_roads3.csv", header=0)
    # Select all rows where the column "road" is equal to "N1"
    road_df = df[['road', 'name', 'lat', 'lon', 'chainage']]
    filtered_df = road_df[road_df['road'].isin(roads)]

    # Define a function to get first and last rows for each group
    def first_last_rows(group):
        return pd.concat([group.iloc[[0]], group.iloc[[-1]]])

    # Apply the function to the DataFrame grouped by 'road'
    start_end_road_df = filtered_df.groupby('road').apply(first_last_rows).reset_index(drop=True)

    start_end_road_df['model_type'] = 'sourcesink'
    start_end_road_df['name'] = 'ss'

    # Add a length column, which is assumed to be 1
    start_end_road_df['length'] = 0
    start_end_road_df['condition'] = np.NAN
    # Put them in the correct order.
    source_sink_df = reorder_columns(start_end_road_df)

    return source_sink_df


def add_source_sink(df, source_sink_df):
    new_df = pd.DataFrame(columns=df.columns)
    prev_value = 'N1'

    # Insert a row for the first entry
    new_df.loc[0] = source_sink_df.iloc[0]
    counter = 1

    for index, row in df.iterrows():
        if row['road'] != prev_value:
            #add a sourcsink row for the beginning
            row_to_insert = source_sink_df.iloc[counter]
            new_df.loc[len(new_df)] = row_to_insert
            counter += 1
            #add a sourcsink row for the end
            row_to_insert = source_sink_df.iloc[counter]
            new_df.loc[len(new_df)] = row_to_insert
            counter += 1

        new_df.loc[len(new_df)] = df.iloc[index]
        prev_value = row['road']

    # Insert a row for the last entry
    new_df.loc[len(new_df)] = source_sink_df.iloc[-1]
    return new_df



def add_links(df):
    """
    This method adds all the links inbetween the bridges, source, and sink. The length is determined by the
    chainage of the next row, minus the chainage of the previous one.
    """
    new_dfs = []
    for i in range(len(df) - 1):
        row_before = df.iloc[i]
        row_after = df.iloc[i + 1]
        if row_before['road'] != row_after['road']:
            new_dfs.append(pd.DataFrame([row_before]))
            continue
        new_row = {
            # put the link inbetween the two bridges
            'chainage': row_before['chainage'] + (row_after['chainage'] - row_before['chainage']) / 2,
            'road': row_before['road'],
            'model_type': 'link',
            'name': 'link ' + str(i+1),
            # put the coordinates as averages of the two lats and lons
            'lat': (row_before['lat'] + row_after['lat']) / 2,
            'lon': (row_before['lon'] + row_after['lon']) / 2,
            # make the length be the difference of the cahinages of its neighbors, and multiply by 1000 to convert km->m
            # rounding is used to fix floating point rounding problems
            'length': max(0, round((row_after['chainage'] - row_before['chainage']) * 1000, 2)),
            'condition': np.NAN
        }

        new_dfs.append(pd.concat([pd.DataFrame([row_before]), pd.DataFrame([new_row])], ignore_index=True))

    # Append the last row of the original DataFrame
    new_dfs.append(pd.DataFrame([df.iloc[-1]]))
    concatted_df = pd.concat(new_dfs, ignore_index=True)
    return concatted_df


def remove_chainage_and_add_id(df):
    """
    This method removes the chainage column as it is not needed anymore, and adds an id column,
    giving each row a unique id starting from 200000
    """
    # Remove chainage
    # df = df.drop(columns=['chainage'])
    # Insert an id column
    df.insert(1, 'unique_id', range(200000, 200000 + len(df)))
    df['id'] = df['intersection_id'].fillna(df['unique_id'])
    df.drop(['intersection_id', 'unique_id', 'chainage'], axis=1, inplace=True)
    return df

def create_inverse_intersections(df, df_intersections):
    # make a copy
    # Assuming df_N_intersections is your DataFrame
    df_intersections_copy = df_intersections.copy()

    # # Swap values in 'road' and 'connects_to' columns
    df_intersections_copy['road'], df_intersections_copy['connects_to'] = df_intersections_copy['connects_to'], \
    df_intersections_copy['road']
    # Iterate over each intersection in the copy
    for index, row in df_intersections_copy.iterrows():
        road_value = row['road']
        lon_value = row['lon']

        # Filter records with the same road value from the original DataFrame
        same_road_records = df[df['road'] == road_value]

        # Calculate the absolute differences between the longitude values
        # and find the index of the record with the closest longitude
        closest_lon_index = np.argmin(np.abs(same_road_records['lon'] - lon_value))

        # Get the chainage value from the record with the closest longitude
        chainage_value = same_road_records.iloc[closest_lon_index]['chainage']

        # Update the chainage value in the copy DataFrame
        df_intersections_copy.at[index, 'chainage'] = chainage_value

    # # Set all values in the 'chainage' column to NaN
    # df_intersections_copy['chainage'] = np.nan
    return df_intersections_copy

def create_intersections(df, roads):
    roads = sorted(roads, key=len, reverse=True)


    # Read the CSV file
    read_df = pd.read_csv("../data/_roads3.csv", header=0)

    # Select all rows where the column "road" is equal to any road in the roads list
    road_df = read_df[['road', 'name', 'lat', 'lon', 'chainage', 'type']]
    filtered_df = road_df[road_df['road'].isin(roads)]

    # Define intersection names
    intersection_names = ['CrossRoad', 'SideRoad']

    # Filter the DataFrame to include only rows where the 'type' column contains any of the intersection names
    mask_type = filtered_df['type'].apply(lambda x: any(name in x for name in intersection_names))
    df_intersections = filtered_df[mask_type]

    # Filter the intersections DataFrame to include only rows where the 'name' column contains any of the road names
    mask_road = df_intersections['name'].apply(lambda x: any(name in x for name in roads))
    df_N_intersections = df_intersections[mask_road]

    print(roads)
    # Add a new column 'connects_to' that contains the substring from the 'name' column
    df_N_intersections['connects_to'] = df_N_intersections['name'].apply(
        lambda x: next((name for name in roads if name in x), None))

    df_N_intersections = df_N_intersections[df_N_intersections['road'] != df_N_intersections['connects_to']]

    df_N_intersections = df_N_intersections[~(
        df_N_intersections['name'].str.contains('N103 on Right') |
        df_N_intersections['name'].str.contains('Intersection with N105') |
        df_N_intersections['name'].str.contains('N209 / N208 to Moulovibazar')
    )]

    df_N_intersections.insert(1, 'intersection_id', range(100000, 100000 + len(df_N_intersections)))
    df_N_intersections['length'] = 0
    df_N_intersections['name'] = df_N_intersections['road'] + 'to' + df_N_intersections['connects_to']
    df_inverse = create_inverse_intersections(df, df_N_intersections)


    # # make a copy
    # # Assuming df_N_intersections is your DataFrame
    # df_N_intersections_copy = df_N_intersections.copy()
    #
    # # Swap values in 'road' and 'connects_to' columns
    # df_N_intersections_copy['road'], df_N_intersections_copy['connects_to'] = df_N_intersections_copy['connects_to'], \
    # df_N_intersections_copy['road']
    #
    # # Set all values in the 'chainage' column to NaN
    # df_N_intersections_copy['chainage'] = np.nan

    ## Here must add chainage
    #

    #concatotnate the copy with the original
    concatenated_df = pd.concat([df_N_intersections, df_inverse], ignore_index=True)

    concatenated_df['model_type'] = 'intersection'

    # Save the resulting DataFrame to a CSV file
    concatenated_df.to_csv('../data/intersectionscombined.csv', index=False)
    return concatenated_df

def add_intersections(df, df_intersectinos):
    concatenated_df = pd.concat([df, df_intersectinos], ignore_index=True)
    sorted_df = concatenated_df.sort_values(by=['road', 'chainage'])
    # compact_df = sorted_df.drop(['type', 'connects_to'], axis=1, inplace=True)
    return sorted_df.reset_index(drop=True)




# Here, all functions are called sequentially

# Get the right data, in this case: the N1 road without irrelevant columns
extracted_df = extract_data()

# Sort the data and remove the duplicates
sorted_df = sort_and_remove_duplicates(extracted_df)
# print('sorted_df\n', sorted_df)

# Add missing columns: model_type, name
full_df = add_modeltype_name(sorted_df)
# print('full_df\n', full_df)

# Reorder the columns so they match the format
reordered_df = reorder_columns(full_df)
# print('reordered_df\n', reordered_df.head(10))

roads = reordered_df['road'].unique()

start_end_of_road_df = create_source_sink(roads)
print('start_end_of_road_df\n', start_end_of_road_df.head(10))

# Format these source and sink dataframes
# formatted_start_end_of_road_df = format_source_sink(start_end_of_road_df)

# # Insert the source before the main dataframe and the sink after words
# combined_df = pd.concat([start_end_of_road_df.iloc[[0]], reordered_df,
#                          start_end_of_road_df.iloc[[1]]])

combined_df = add_source_sink(df=reordered_df, source_sink_df=start_end_of_road_df)
# print('combined_df\n', combined_df)

combined_df.to_csv('../data/combined.csv', index=False)


intersections_df = create_intersections(combined_df, roads)
with_intersections_df = add_intersections(combined_df, intersections_df)

with_links_df= add_links(with_intersections_df)

# Remove the chainage column and give each record a unique id
final_df = remove_chainage_and_add_id(with_links_df)




# Display the DataFrame
# print(final_df)
# print(final_df['length'].sum())

# Save to a csv file in the same folder as the other demos

#TODO make this pretty

# Indices of the rows you want to switch
index1 = 2564  # Index of the first row
index2 = 2562 # Index of the second row
index3 = 1351
index4 = 1353


# Get the rows to be switched
row1 = final_df.iloc[index1].copy()  # Use copy to avoid setting with copy warning
row2 = final_df.iloc[index2].copy()
row3 = final_df.iloc[index3].copy()
row4 = final_df.iloc[index4].copy()

# Swap the rows
final_df.iloc[index1], final_df.iloc[index2] = row2, row1
final_df.iloc[index3], final_df.iloc[index4] = row4, row3


final_df.to_csv('../data/N1N2.csv', index=False)
