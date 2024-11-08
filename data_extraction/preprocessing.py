def missing_utfAvgTid(df):
    missing_rows_Avg = df[df['UtfAvgTid'].isnull()] 
    missing_rows_Ank = df[df['UtfAnkTid'].isnull()]
    #this Avg station should be the same as the previous Ank station
    for row in missing_rows_Avg.iterrows():
        row = row[1]
    
        train_number = row['Tåguppdrag']
        date = row['Datum_PAU']
        previous_station = df[(df['Tåguppdrag'] == train_number) & (df['Datum_PAU'] == date) & (df['Ankomstplats'] == row['Avgångsplats'])]
        if previous_station.empty:
            print('No previous station found for', train_number, date, row['Avgångsplats'])
        else:
            #update the row where id is the same as the missing row
            row_id = row['id']
            df.loc[df['id'] == row_id, 'UtfAvgTid'] = previous_station['UtfAnkTid'].values[0]
    
    for row in missing_rows_Ank.iterrows():
        train_number = row['Tåguppdrag']
        date = row['Datum_PAU']
        next_station = df[(df['Tåguppdrag'] == train_number) & (df['Datum_PAU'] == date) & (df['Avgångsplats'] == row['Ankomstplats'])]
        if next_station.empty:
            print('No next station found for', train_number, date, row['Ankomstplats'])
        else:
            #update the row where id is the same as the missing row
            row_id = row['id']
            df.loc[df['id'] == row_id, 'UtfAnkTid'] = next_station['UtfAvgTid'].values[0]
    print('\n------Missing values filled------')
    print("total missing values in UtfAvgTid", len(missing_rows_Avg))
    print("total missing values in UtfAnkTid", len(missing_rows_Ank))
    print("after update total nan avg values are ", len(df[df['UtfAvgTid'].isnull()]))
    print("after update total nan ank values are ", len(df[df['UtfAnkTid'].isnull()]))
    print("\n\n\n")
    return

def create_ids(df):
    #create a unique id for each row
    df['id'] = df.index
    return