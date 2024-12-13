
# Checking for and removing missing values in the dataset
def missing_utfAvgTid(df):
    missing_rows_Avg = df[df['UtfAvgTid'].isnull()] 
    missing_rows_Ank = df[df['UtfAnkTid'].isnull()]

    #looping through the rows "Avg" (departure) with missing values and filling them
    for row in missing_rows_Avg.iterrows():
        row = row[1] #the row element is a tuple and the row is the second element
        train_number = row['Tåguppdrag']
        date = row['Datum_PAU']
        #get the previous station to determine the missing value
        previous_station = df[(df['Tåguppdrag'] == train_number) & (df['Datum_PAU'] == date) & (df['Ankomstplats'] == row['Avgångsplats'])]
        
        if previous_station.empty:
            print('No previous station found for', train_number, date, row['Avgångsplats'])
        else:
            #update the row where id is the same as the missing row
            row_id = row['id']
            df.loc[df['id'] == row_id, 'UtfAvgTid'] = previous_station['UtfAnkTid'].values[0]
    
    # #looping through the rows of "Ank" (arrival) with missing values and filling them
    for row in missing_rows_Ank.iterrows():
        row = row[1] #the row element is a tuple and the row is the second element
        train_number = row['Tåguppdrag']
        date = row['Datum_PAU']
        # get the next station to determine the missing value
        next_station = df[(df['Tåguppdrag'] == train_number) & (df['Datum_PAU'] == date) & (df['Avgångsplats'] == row['Ankomstplats'])]
        
        if next_station.empty:
            print('No next station found for', train_number, date, row['Ankomstplats'])
        else:
            #update the row where id is the same as the missing row
            row_id = row['id']
            df.loc[df['id'] == row_id, 'UtfAnkTid'] = next_station['UtfAvgTid'].values[0]

    print_statistics = False # set to True to print statistics
    if print_statistics:
        print('\n------Missing values filled------')
        print("total missing values in UtfAvgTid", len(missing_rows_Avg))
        print("total missing values in UtfAnkTid", len(missing_rows_Ank))
        print("after update total nan avg values are ", len(df[df['UtfAvgTid'].isnull()]))
        print("after update total nan ank values are ", len(df[df['UtfAnkTid'].isnull()]))
        print("\n\n\n")
    
    return

#creates a unique id for each row
def create_ids(df):
    #save the index as the id
    df['id'] = df.index
    return