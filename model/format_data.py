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
    bridge_df = df_read[df_read['road'] == 'N1']
    # List of column names to remove
    columns_to_remove = ['type', 'roadName', 'structureNr', 'width', 'constructionYear', 'spans', 'zone',
                         'circle', 'division', 'sub-division', 'EstimatedLoc']
    # Drop the specified columns
    bridge_df = bridge_df.drop(columns=columns_to_remove)

    bridge_df = bridge_df[bridge_df['chainage'] <= 241.063]

    return bridge_df


def sort_and_remove_duplicates(df):
    """
    This method sorts the dataframe based on the chainage, and then removes any duplicates from those columns
    """
    ordered_df = df.sort_values(by='chainage')
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
    dropped_df = ordered_df.groupby('LRPName').agg(aggregations)
    dropped_df['name'] = (dropped_df['name']
                          .str.lower()
                          .str.replace('r', 'l')
                          .str.replace(' ', '')
                          .str.replace('.', ''))

    dropped_df = dropped_df.groupby('name').agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    dropped_df = dropped_df.groupby('chainage').agg(aggregations)
    dropped_df.reset_index(drop=True, inplace=True)

    dropped_df.drop(columns=['name'], inplace=True)

    return dropped_df


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


def create_source_sink():
    """
    This method makes a source and a sink dataframe
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv("../data/_roads3.csv", header=0)
    # Select all rows where the column "road" is equal to "N1"
    road_df = df[df['road'] == 'N1']
    # add model types and names for the source and sink dataframes
    start_end_road_df = (road_df[road_df['name'].str
                         .startswith(('Start of Road', 'Chittagong city area ends and the survey of N1 starts again'))]
                         .copy())
    start_end_road_df['model_type'] = 'sourcesink'
    start_end_road_df['name'] = 'sourcesink'
    return start_end_road_df


def format_source_sink(source_sink_df):
    """
    This method removes unnecessary columns from the dataframe, renames lrp to id, and adds a length
    of 0. It also calls the reorder method to order to columns in the same format.
    """
    # Remove unnecessary columns
    source_sink_df.drop(columns=['gap', 'type', 'lrp'], inplace=True)
    # Add a length column, which is assumed to be 1
    source_sink_df['length'] = 0
    source_sink_df['condition'] = np.NAN
    # Put them in the correct order.
    source_sink_df = reorder_columns(source_sink_df)
    return source_sink_df


def add_links(df):
    """
    This method adds all the links inbetween the bridges, source, and sink. The length is determined by the
    chainage of the next row, minus the chainage of the previous one.
    """
    new_dfs = []
    for i in range(len(df) - 1):
        row_before = df.iloc[i]
        row_after = df.iloc[i + 1]
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

    return pd.concat(new_dfs, ignore_index=True)


def remove_chainage_and_add_id(df):
    """
    This method removes the chainage column as it is not needed anymore, and adds an id column,
    giving each row a unique id starting from 200000
    """
    # Remove chainage
    df = df.drop(columns=['chainage'])
    # Insert an id column
    df.insert(1, 'id', range(200000, 200000 + len(df)))
    return df


# Here, all functions are called sequentially

# Get the right data, in this case: the N1 road without irrelevant columns
extracted_df = extract_data()

# Sort the data and remove the duplicates
sorted_df = sort_and_remove_duplicates(extracted_df)

# Add missing columns: model_type, name
full_df = add_modeltype_name(sorted_df)

# Reorder the columns so they match the format
reordered_df = reorder_columns(full_df)

# Create dataframes for the source and sink line
start_end_of_road_df = create_source_sink()

# Format these source and sink dataframes
formatted_start_end_of_road_df = format_source_sink(start_end_of_road_df)

# Insert the source before the main dataframe and the sink after words
combined_df = pd.concat([formatted_start_end_of_road_df.iloc[[0]], reordered_df,
                         formatted_start_end_of_road_df.iloc[[1]]])


# Add all the links
with_links_df = add_links(combined_df)

# Remove the chainage column and give each record a unique id
final_df = remove_chainage_and_add_id(with_links_df)

# Display the DataFrame
print(final_df)
print(final_df['length'].sum())

# Save to a csv file in the same folder as the other demos
final_df.to_csv('../data/N1.csv', index=False)
