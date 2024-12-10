
import pandas as pd
import numpy as np
import os
import preprocessing

file_name = "month_data.csv"
# Define the file path
file_path = r"../" +"data" + "/" + file_name

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Relative path to the file
file_path = '../data/month_data.csv'

# Read the CSV
df = pd.read_csv(file_path, sep=';', encoding='utf-8')

specific_date = '2019-03-27'  # Change this to desired date

#filter the dataset to specific date and get a copy
filtered_for_date=df[(df['Datum_PAU']== specific_date)].copy()

#convert planned arrival time and planned departure time to date time format
filtered_for_date['PlanAnkTid'] = pd.to_datetime(filtered_for_date['PlanAnkTid'],errors='coerce')
filtered_for_date['PlanAvgTid'] = pd.to_datetime(filtered_for_date['PlanAvgTid'],errors='coerce')

filter_for_time_bool = False  # Set to True to filter by peak hours. Will result in some missing data for a few routes (no final station row)

if filter_for_time_bool:
    start_time = '7:00:00' #select peak hours of the day
    end_time = '19:00:00'
    #filter the dataset for the peak time
    filtered_for_time = filtered_for_date[((filtered_for_date['PlanAnkTid'].dt.time >= pd.to_datetime(start_time).time()) &
                                      (filtered_for_date['PlanAnkTid'].dt.time <= pd.to_datetime(end_time).time())) | 
                                      ((filtered_for_date['PlanAvgTid'].dt.time >= pd.to_datetime(start_time).time()) &
                                      (filtered_for_date['PlanAvgTid'].dt.time <= pd.to_datetime(end_time).time()))]
else:
    #if not filtering for peak hours, use the filtered for date dataset
    filtered_for_time = filtered_for_date


#filter for the platform
filtered_for_platform= filtered_for_time[(filtered_for_time['Ankomstplats']=='Uppsala c') | ( filtered_for_time['Avgångsplats']=='Uppsala c' )]

#List of unique mission station codes to filter by
unique_missions_for_given_station = ['U', 'Gä', 'Cst','Fln']

#Filter the DataFrame based on starting or ending stations matching the unique mission codes
filtered_for_unique_mission_for_given_station = filtered_for_platform[
    (filtered_for_platform['Förstaplatssignatur_för_Uppdrag'].isin(unique_missions_for_given_station)) &
    (filtered_for_platform['Sistaplatssignatur_för_Uppdrag'].isin(unique_missions_for_given_station))
]

# Extract the unique Train mission from the filtered dataset
unique_missions = filtered_for_unique_mission_for_given_station['Tåguppdrag'].unique()

#excluding train missions as they do not show Första or Sista
excluded_missions= [13,51,882,806,8417,8419]

# List of specific mission types interested in
mission_types = unique_missions[~np.isin(unique_missions, excluded_missions)]

# Initialize a dictionary to hold routes for each mission
all_routes = {}
all_filtered_rows = []  # List to store rows used to generate the routes

# Loop through each mission type
for mission_type in mission_types:
    # Initialize a list to hold the routes for the current mission
    routes = []

    # Iterate through the DataFrame to build routes
    for index, row in filtered_for_time.iterrows():
        if row['UppehållstypAvgång'] == 'Första' and row['Tåguppdrag'] == mission_type:
            # Start the route with the departure station
            current_route = [row['Avgångsplats']]
            all_filtered_rows.append(row)  # Save the current row to the list
            next_row = row

            # Add the arrival station of the "Första" row to the route as the next stop
            current_route.append(next_row['Ankomstplats'])
             
            # Continue to add stations until reaching 'Sista'
            while next_row['UppehållstypAnkomst'] != 'Sista':
                # Find the next station in the DataFrame that matches the current departure station
                next_station = filtered_for_time[(filtered_for_time['Avgångsplats'] == next_row['Ankomstplats']) &
                                  (filtered_for_time['Tåguppdrag'] == mission_type)]
                
                if not next_station.empty:
                    next_stop = next_station.iloc[0]['Ankomstplats']
                    current_route.append(next_stop)
                    all_filtered_rows.append(next_station.iloc[0])  # Save the row to the list
                    next_row = next_station.iloc[0]  # Move to the next row

                else:
                    break  # Exit if no further stations are found

            # If the last station is 'Sista', append it to the route
            if next_row['UppehållstypAnkomst'] == 'Sista' and next_row['Ankomstplats'] != current_route[-1]:
                current_route.append(next_row['Ankomstplats'])

            # Add the complete route to the routes list
            routes.append(" -> ".join(current_route))

    # Store the routes for the current mission in the dictionary
    all_routes[mission_type] = routes

# Save the filtered rows used to generate the routes into a new DataFrame
filtered_rows_df = pd.DataFrame(all_filtered_rows)

#fill missing values and do preprocessing
preprocessing.create_ids(filtered_rows_df)
preprocessing.missing_utfAvgTid(filtered_rows_df)

# Save the DataFrame to a CSV file
output_rows_file_name = 'test_network_U_Fln_Cst_Ga_no_time.csv'
filtered_rows_df.to_csv('../data/' + output_rows_file_name, index=False, encoding='utf-8-sig')
 

print_statisics = False # Set to True to display statistics

if print_statisics:
    # Display the routes for each mission
    for mission, routes in all_routes.items():
        print(f"\nRoutes for {mission}:")
        for route in routes:
            print("Route:", route)


    #Descriptive statistic of test network
    print('Number of unique train missions used for test network: ',len(unique_missions))

    #total records included in the test network
    print('Number of total records in the test network:',len(filtered_rows_df))

    #number of unique depature stations
    print('Number of unique depature stations in the test network:',len(filtered_rows_df['Avgångsplats'].unique()))

    #number of unique arrival stations
    print('Number of unique arrival stations in the test network:',len(filtered_rows_df['Ankomstplats'].unique()))

    #number of unique routes
    print('Number of unique routes available in the test network:',len(filtered_rows_df[['Ankomstplats','Avgångsplats']].drop_duplicates()))

    #number of depature delays
    print('Number of depature delays:',len(filtered_rows_df[filtered_rows_df['AvgFörsening'] > 0].dropna(subset=['AvgFörsening'])))

    #maximum depature delay across all routes
    print('Maximum depature delay from a station across all routes(minutes):', max(filtered_rows_df['AvgFörsening'] ))

    #early depature across all routes
    print('Maximum early depature from a station across all routes(minutes):',min(filtered_rows_df['AvgFörsening']))

    #number of arrival delays
    print('Number of arrival delays:',len(filtered_rows_df[filtered_rows_df['AnkFörsening']>0].dropna(subset=['AnkFörsening'])))

    #maximum arrival delay across all routes
    print('Maximum arrival delay to a station across all routes(minutes):', max(filtered_rows_df['AnkFörsening'] ))

    #early arrival across all routes
    print('Maximum early arrival to a station across all routes(minutes):',min(filtered_rows_df['AnkFörsening']))

    #Number of edges delay in travel time
    print('Number of edges delay in travel time: ', len(filtered_rows_df[filtered_rows_df['FörseningUppehållAvgång']>0]))

    #maximum travel time delay across edges
    print('Maximum travel time delay within edges:',max(filtered_rows_df['FörseningUppehållAvgång'].dropna()))

    #Number of trains with no delays
    no_delay_count = len(filtered_rows_df[
                (filtered_rows_df['AnkFörsening']<=0) & 
                (filtered_rows_df['AvgFörsening']<=0) & 
                (filtered_rows_df['FörseningUppehållAvgång']<=0)
                ]
                )
    print('Number of trains with no delays:',no_delay_count)

