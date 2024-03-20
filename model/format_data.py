import numpy as np
import pandas as pd

# To help with printing/debugging
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 10)


def extract_data():
    """
    This function gets called first.
    It extracts the data from the Excel file and removes all unnecessary information, and returns a dataframe.
    :return: A dataframe that contains the relevant data from the Excel file.
    """
    # Assuming _roads3.csv is in the current directory
    file_path = "../data/BMMS_overview.xlsx"
    # Read the CSV file into a DataFrame
    df_read = pd.read_excel(file_path, header=0)  # header=0 means use the first row as column names

    # Select all rows where the column "road" that start with N1 and N2
    bridge_df = df_read[df_read['road'].str.startswith(('N1', 'N2'))]

    # List of column names to remove
    columns_to_remove = ['type', 'roadName', 'structureNr', 'width', 'constructionYear', 'spans', 'zone',
                         'circle', 'division', 'sub-division', 'EstimatedLoc']

    # Drop the columns that are based on the above list
    bridge_df = bridge_df.drop(columns=columns_to_remove)

    # Identify roads that are long enough
    valid_roads = bridge_df.groupby('road')['chainage'].max() > 25

    # Filter the DataFrame to keep only the records where the road is in valid_roads
    df_filtered = bridge_df[bridge_df['road'].isin(valid_roads.index[valid_roads])]
    df_filtered = df_filtered[df_filtered['road'] != 'N106']

    return df_filtered


def sort_and_remove_duplicates(df):
    """
    This method sorts the dataframe based on the chainage, and then removes any duplicates from those columns.
    :param df: The dataframe from which the duplicates need to be removed, and that needs sorting.
    :return: A sorted dataframe without duplicates.
    """
    ordered_df = df.sort_values(by=['road', 'chainage'])

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

    # Remove bridges with the same name
    dropped_df.reset_index(drop=True, inplace=True)
    dropped_df = dropped_df.groupby(['name', 'road']).agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    # Remove bridges with the same chainage
    dropped_df = dropped_df.groupby(['chainage', 'road']).agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    # remove the name column
    dropped_df.drop(columns=['name'], inplace=True)

    # sort the dataframe by road and then chainage
    sorted_dropped_df = dropped_df.sort_values(by=['road', 'chainage'])
    sorted_dropped_df.reset_index(drop=True, inplace=True)
    return sorted_dropped_df


def add_modeltype_name(df):
    """
    This method adds a modeltype of bridge, and a name.
    :param df: The dataframe of bridges
    :return: A dataframe with a model_type column and name column
    """
    # Label all bridges as a bridge
    df['model_type'] = 'bridge'
    # Add a column called 'name' filled with 'link' and a number from 1 to n
    df['name'] = 'bridge ' + (df.index + 1).astype(str)
    return df


def create_inverse_intersections(df, df_intersections):
    """
    This helper method creates the inverse intersections. That means that if there is an intersection from the
    N1 road to the N2 road, it creates an intersection from the N2 road to the N1 road.
    :param df: The dataframe to refer to find the correct chainage for the inverse intersections
    :param df_intersections: The dataframe with the intersections that need to be flipped
    """
    df_intersections = df_intersections.copy()
    # Swap values in 'road' and 'connects_to' columns
    df_intersections['road'], df_intersections['connects_to'] = df_intersections['connects_to'], \
        df_intersections['road']

    # This loop is to find the correct chainage for each intersection, this helps determine where to place
    # the intersection on the road.
    # Iterate over each intersection
    for index, row in df_intersections.iterrows():
        road_value = row['road']
        lon_value = row['lon']

        # Filter records with the same road value from the original DataFrame, this will help determine the chainage
        same_road_records = df[df['road'] == road_value]

        # Calculate the absolute differences between the longitude values
        # and find the index of the record with the closest longitude
        closest_lon_index = np.argmin(np.abs(same_road_records['lon'] - lon_value))

        # Get the chainage value from the record with the closest longitude
        chainage_value = same_road_records.iloc[closest_lon_index]['chainage']

        # Update the chainage value in the copy DataFrame
        df_intersections.at[index, 'chainage'] = chainage_value
    df_intersections.reset_index(drop=True)
    return df_intersections


def create_intersections(df, roads):
    """
    This method creates the intersections based on the roads data. It finds all CrossRoads and SideRoads, and
    turns it into a dataframe whose rows can be inserted in the add intersections method.
    :param df: The dataframe of roads without the intersections
    :param roads: The array of roads that are in the data
    """
    # Read the CSV file
    read_df = pd.read_csv("../data/_roads3.csv", header=0)

    # Select all rows where the column "road" is equal to any road in the roads list
    road_df = read_df[['road', 'name', 'lat', 'lon', 'chainage', 'type']]

    # Sort the array of road names by reverse lengths of the string, because the filter
    # uses isin(), which would otherwise see that N102 as N1, instead of N1
    roads = sorted(roads, key=len, reverse=True)
    filtered_df = road_df[road_df['road'].isin(roads)]

    # Define intersection names
    intersection_names = ['CrossRoad', 'SideRoad']

    # Filter the DataFrame to include only rows where the 'type' column contains any of the intersection names
    mask_type = filtered_df['type'].apply(lambda x: any(name in x for name in intersection_names))
    df_intersections = filtered_df[mask_type]

    # Filter the intersections DataFrame to include only rows where the 'name' column contains any of the road names
    mask_road = df_intersections['name'].apply(lambda x: any(name in x for name in roads))
    df_n_intersections = df_intersections[mask_road]

    # Add a new column 'connects_to' that contains the substring from the 'name' column
    # df_n_intersections['connects_to'] = df_n_intersections['name'].apply(
    #     lambda x: next((name for name in roads if name in x), None))
    df_n_intersections = df_n_intersections.copy()  # Create a copy of the dataframe
    df_n_intersections['connects_to'] = df_n_intersections['name'].apply(
        lambda x: next((name for name in roads if name in x), None))

    # Remove self loops, intersections that loop with itself do not make sense
    df_n_intersections = df_n_intersections[df_n_intersections['road'] != df_n_intersections['connects_to']]

    # The algorithm takes N103 as an intersection because it reads the substring N1,  this happens 3 times, so these
    # roads are removed by hand.
    df_n_intersections = df_n_intersections[~(
        df_n_intersections['name'].str.contains('N103 on Right') |
        df_n_intersections['name'].str.contains('Intersection with N105') |
        df_n_intersections['name'].str.contains('N209 / N208 to Moulovibazar')
    )]

    # give the intersections an id
    df_n_intersections.insert(1, 'intersection_id', range(100000, 100000 + len(df_n_intersections)))

    # Make the length of each intersection 0
    df_n_intersections['length'] = 0

    # Make the model_type intersection
    df_n_intersections['model_type'] = 'intersection'

    # Give it a descriptive name
    df_n_intersections['name'] = df_n_intersections['road'] + 'to' + df_n_intersections['connects_to']

    # Call the helper function that creates 'inverse' intersections, meaning, if there is an intersection
    # going from N1 to N2, there must be an intersection from N2 to N1 as well.
    df_inverse = create_inverse_intersections(df, df_n_intersections)

    # Combine the original intersections with the inverses
    concatenated_df = pd.concat([df_n_intersections, df_inverse], ignore_index=True)
    concatenated_df.reset_index(drop=True)

    return concatenated_df


def add_intersections(df, df_intersections):
    """
    This takes the intersections that the create_intersections method made, and puts it in the rights spot.
    :param df: main dataframe without that needs to get intersections
    :param df_intersections: the intersections that need to be inserted
    :return: the dataframe with newly added intersections
    """
    print('Adding intersections...')
    print(df_intersections)
    concatenated_df = pd.concat([df, df_intersections], ignore_index=True)
    sorted_concatenated_df = concatenated_df.sort_values(by=['road', 'chainage'])
    return sorted_concatenated_df.reset_index(drop=True)


def create_source_sink(roads):
    """
    This method makes a source and a sink dataframe.
    :param roads: The string of roads that is in the data.
    :return: a dataframe with only sourcesinks
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
    # Although not a good variable name, it is short which helps the visualisation less cluttered
    start_end_road_df['name'] = 'ss'

    # Add a length column, which is assumed to be 0
    start_end_road_df['length'] = 0

    return start_end_road_df


def add_source_sink(df, source_sink_df):
    """
    Add the sources and sinks to the dataframe of bridges and intersections
    :param df: dataframe the main dataframe containing the bridges and intersections
    :param source_sink_df: the sources and sinks dataframe whose rows will be injected into df
    :return: a dataframe with bridges, intersections, and sourcesinks
    """
    # Start from scratch by making a new dataframe
    new_df = pd.DataFrame(columns=df.columns)
    prev_value = 'N1'

    # Insert a row for the first entry
    new_df.loc[0] = source_sink_df.iloc[0]
    counter = 1

    # Iterate over each row of the main dataframe
    for index, row in df.iterrows():
        # If there is a new road, add a sourcesink to the end of the previous rode
        # and one to the beginning of the next road
        if row['road'] != prev_value:
            # Add a sourcsink row for the beginning
            row_to_insert = source_sink_df.iloc[counter]
            new_df.loc[len(new_df)] = row_to_insert
            counter += 1
            # Add a sourcsink row for the end
            row_to_insert = source_sink_df.iloc[counter]
            new_df.loc[len(new_df)] = row_to_insert
            counter += 1

        new_df.loc[len(new_df)] = df.iloc[index]
        prev_value = row['road']

    # Insert a row for the last sourcesink
    new_df.loc[len(new_df)] = source_sink_df.iloc[-1]
    return new_df


def add_links(df):
    """
    This method adds all the links inbetween the bridges, source, and sink. The length is determined by the
    chainage of the next row, minus the chainage of the previous one.
    :param df: the input dataframe for which links need to be added
    :return: a dataframe with bridges, sourcesinks, intersections, and links
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
            # put the coordinates as averages of the two lattitudes and longitudes
            'lat': (row_before['lat'] + row_after['lat']) / 2,
            'lon': (row_before['lon'] + row_after['lon']) / 2,
            # make the length be the difference of the cahinages of its neighbors, and multiply by 1000 to convert km->m
            # rounding is used to fix floating point rounding problems
            'length': max(0, round((row_after['chainage'] - row_before['chainage']) * 1000, 2)),
        }

        new_dfs.append(pd.concat([pd.DataFrame([row_before]), pd.DataFrame([new_row])], ignore_index=True))

    # Append the last row of the original DataFrame
    new_dfs.append(pd.DataFrame([df.iloc[-1]]))
    concatted_df = pd.concat(new_dfs, ignore_index=True)
    return concatted_df


def remove_columns_and_add_id(df):
    """
    This method removes the chainage column as it is not needed anymore, and adds an id column,
    giving each row a unique id starting from 200000
    :param df: dataframe that needs some columns removed
    :return: a dataframe without unnecessary columns and a new id column
    """
    df.insert(1, 'unique_id', range(200000, 200000 + len(df)))
    # The id of the intersections must stay the same, the rest get a new id
    df['id'] = df['intersection_id'].fillna(df['unique_id'])
    # Define the desired column order
    desired_column_order = ['road', 'id', 'model_type', 'condition', 'name', 'lat', 'lon', 'length']
    # Reassign the DataFrame with the desired column order
    df = df[desired_column_order]

    return df


# Here, all functions are called sequentially

# Get the right data, in this case: the N1 road without irrelevant columns
extracted_df = extract_data()

# Sort the data and remove the duplicates
sorted_df = sort_and_remove_duplicates(extracted_df)

# Add missing columns: model_type, name
full_df = add_modeltype_name(sorted_df)

# make an array of strings with the names of the roads that are in the data
roads_array = full_df['road'].unique()

# create and then add intersections
intersections_df = create_intersections(full_df, roads_array)
with_intersections_df = add_intersections(full_df, intersections_df)

# create and then add sourcesinks
start_end_of_road_df = create_source_sink(roads_array)
combined_df = add_source_sink(df=with_intersections_df, source_sink_df=start_end_of_road_df)

# add links
with_links_df = add_links(combined_df)

# Remove the unnecessary columns and give each record a unique id
final_df = remove_columns_and_add_id(with_links_df)

# Save to datafile
final_df.to_csv('../data/N1N2.csv', index=False)
