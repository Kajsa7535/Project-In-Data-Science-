import pandas as pd


def get_dataframe(file_path, sep=';'):
    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
    return df


def create_smaller_network(df):
    network_name = "data/smaller_test_network.csv"
    
    # Creating smaller test network
    # Step 1: Read the original DataFrame (assuming 'df_network' is the original data)
    smaller_test_network = df.head(12)
    smaller_test_network.reset_index(drop=True, inplace=True)

    # Modify first row
    smaller_test_network.at[0, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[0, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[0, 'UppehållstypAnkomst'] = 'Sista'
    smaller_test_network.at[0, 'Tåguppdrag'] = 1


    # Modify second row
    smaller_test_network.at[1, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[1, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[1, 'UppehållstypAnkomst'] = 'Sista'
    smaller_test_network.at[1, 'Tåguppdrag'] = 1

    # Modify third row
    smaller_test_network.at[2, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[2, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[2, 'Tåguppdrag'] = 2
    smaller_test_network.at[2, 'UppehållstypAnkomst'] = 'Passage'


    # Modify fourth row
    smaller_test_network.at[3, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[3, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[3, 'Tåguppdrag'] = 2
    smaller_test_network.at[3, 'UppehållstypAnkomst'] = 'Passage'


    # Modify fifth row
    smaller_test_network.at[4, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[4, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[4, 'Tåguppdrag'] = 2
    smaller_test_network.at[4, 'UppehållstypAnkomst'] = 'Passage'


    # Modify sixth row
    smaller_test_network.at[5, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[5, 'Avgångsplats'] = 'Solna'
    smaller_test_network.at[5, 'Tåguppdrag'] = 2
    smaller_test_network.at[5, 'UppehållstypAnkomst'] = 'Passage'

    # Modify seventh row
    smaller_test_network.at[6, 'Ankomstplats'] = 'Stockholm'
    smaller_test_network.at[6, 'Avgångsplats'] = 'Hagalund'
    smaller_test_network.at[6, 'Tåguppdrag'] = 2
    smaller_test_network.at[6, 'UppehållstypAnkomst'] = 'Passage'
    # Modify eighth row
    smaller_test_network.at[7, 'Ankomstplats'] = 'Stockholm'
    smaller_test_network.at[7, 'Avgångsplats'] = 'Hagalund'
    smaller_test_network.at[7, 'Tåguppdrag'] = 2
    smaller_test_network.at[7, 'UppehållstypAnkomst'] = 'Passage'

    # Modify ninth row
    smaller_test_network.at[8, 'Ankomstplats'] = 'Stockholm'
    smaller_test_network.at[8, 'Avgångsplats'] = 'Hagalund'
    smaller_test_network.at[8, 'Tåguppdrag'] = 2
    smaller_test_network.at[8, 'UppehållstypAnkomst'] = 'Passage'
    # Modify tenth row
    smaller_test_network.at[9, 'Ankomstplats'] = 'Stockholm'
    smaller_test_network.at[9, 'Avgångsplats'] = 'Hagalund'
    smaller_test_network.at[9, 'Tåguppdrag'] = 2
    smaller_test_network.at[9, 'UppehållstypAnkomst'] = 'Passage'

    smaller_test_network.at[10, 'Ankomstplats'] = 'Hagalund'
    smaller_test_network.at[10, 'Avgångsplats'] = 'Stockholm'
    smaller_test_network.at[10, 'Tåguppdrag'] = 3
    smaller_test_network.at[10, 'UppehållstypAnkomst'] = 'Passage'

    smaller_test_network.at[11, 'Ankomstplats'] = 'Solna'
    smaller_test_network.at[11, 'Avgångsplats'] = 'Hagalund'
    smaller_test_network.at[11, 'Tåguppdrag'] = 3
    smaller_test_network.at[11, 'UppehållstypAnkomst'] = 'Passage'

    smaller_test_network
    # Step 3: Save the modified DataFrame to a new CSV file
    smaller_test_network.to_csv(network_name, index=False)
    return network_name


def main():
    print("in main")
    month_data = "month_data.csv"
    file_path = r"data" + "/" + month_data
    df = get_dataframe(file_path)
    smaller_network_name = create_smaller_network(df)
    print("done with smaller network creation: ", smaller_network_name)
    
if __name__ == "__main__":
    main()
