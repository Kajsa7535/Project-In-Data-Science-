
# Network init calculations
# =========================
# - A (adjacency matrix): for each station, put 1 if it is connected to another station, otherwise 0.
# - B = for each station, calculate the turnover rate
# - fij = for each edge, calculate the train frequency from i to j
# - tij = for each edge, calculate the time it took to travel, and average it 
# - pij = for all trains that have gone to i, calculate the probability of going to j (in a fraction)
#     Note: If a train has i as first station it will not be counted in pij even if its going to j. Only trains that are going to i and then j are included
# - rij =  for all trains that do not have i as end station, calculate the probability of those trains going to j (in a fraction)
#     Note: If a train has i as first station it will not be counted in rij even if its going to j. Only trains that are going to i and then j are included
# - sj = for each train that pass through station j, calculate the number of trains that end at j. (On that edge)
# - T(i,j) = extract from data
#


import pandas as pd
import numpy as np


class Station:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.delay = 0 # total delay at the station at the current time step, maybe this needs to be a list so we store the delay for each time step
        self.T_ij = []  # set of trains moving to station i at time t #TODO
        self.N_out = [] # set of stations to which there is a edge from station i (neightbours out)
        self.N_in = [] # set of stations from which there is an edge to station i (neighbours in)
        self.Bis = None # turnover rate dict of Bi for each hour of the day
        self.si = None #fraction of trains on the edge towards this station that end at this station

    def initiate_station(self, df, network_start_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]
        self.si = self.get_si(rows)
        self.delay = self.initiate_delay(df, network_start_time)
        return

    # Function that finds the intial delay of the station
    def initiate_delay(self, df, network_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]        
        rows = rows[(rows['UtfAnkTid'] >= network_time) & (rows['UtfAvgTid'] <= network_time)]
        if len(rows) == 0:
            return 0
        
        #sum of all of the incoming trains delays
        delay = rows['AvgFörsening'].sum()
        return delay


    # Function that finds fraction of trains on the edges towards this station that end at this station
    def get_si(self, rows):
        #rows are all rows that are going to this station
        if len(rows) == 0:
            return 0
        final_station_rows = rows[rows['UppehållstypAnkomst'] == 'Sista']
        total_rows = len(rows)
        final_rows = len(final_station_rows)
        return final_rows/total_rows
    
    # Creates and initiates Bi dictionary, for now creates one Bi for every hour (00 - 23)
    def initiate_Bis(self, edges): 
        bi_dict = {}

        for i in range(24): # creates 24 intervals, one for every hour
            start_time = pd.to_datetime(f"2019-03-31 {i}:00:00.000").time()
            if i == 23:
                end_time = pd.to_datetime(f"2019-03-31 00:00:00.000").time()
            else:
                end_time = pd.to_datetime(f"2019-03-31 {i+1}:00:00.000").time()
         
            bi_dict[(start_time, end_time)] = 0
        
        for time_span in bi_dict.keys(): # For every hour, calculate the individual Bi
            bi_dict[time_span] = self.calculate_Bi(edges, time_span)
            
        self.Bis = bi_dict
        return

    
    #Function that takes all edges that are incoming to this station, divides the sum of frequencies by the sum of frequencies times the average time
    def calculate_Bi(self, edges, time_span):
        incomming_edges = [value for key, value in edges.items() if key[1] == self.name]
        sum_of_freq = 0
        sum_of_freq_and_time = 0

        # Goes over every edge of the incomming edges towards this station
        for edge in incomming_edges:
            freq = edge.fetch_fij(time_span) # gets the frequency of that edge on the current time interval (1 hour right now)
            avg_time = edge.fetch_tij(time_span) # gets the avg time of that edge (calculated on that time interval)
            sum_of_freq += freq
            sum_of_freq_and_time += freq * avg_time
            
        if sum_of_freq_and_time == 0: # check that the denominator will not be 0
            return 0
        else:
            return  sum_of_freq/sum_of_freq_and_time

    def set_N_in(self, neighbours_in):
        self.N_in = neighbours_in
    
    def set_N_out(self, neighbours_out):
        self.N_out = neighbours_out

    def fetch_Bi(self, time_span):
        value = self.Bis[time_span]
        return value


#####################################################################################################

class Edge: 
    def __init__(self, id, start, end, adj_number):
        self.id = id #edge id
        self.i = start #station i = start
        self.j = end # station j = end
        self.Aij = adj_number #always 1 for edged that exists, might be redundant
        self.fijs = None # #dict of frquencies for each hour of the day
        self.tijs = None # dict of average travel times for each hour of the day
        self.pij = None # fraction of trains to i that continues to j. It is a probability.
        self.rij = None # fraction of trains to i that continue to j if they do not end at i
    
    #function to fetch the frequency of trains from i to j at a specific time span
    #time span is a tuple of two times in Timestamp format, needs to in format of (0,1), (1,2) etc
    def fetch_tij(self, time_span):
        value = self.tijs[time_span]
        return value
    
    #function to fetch the frequency of trains from i to j at a specific time span
    #time span is a tuple of two times in Timestamp format, needs to in format of (0,1), (1,2) etc
    def fetch_fij(self, time_span):
        value = self.fijs[time_span]
        return value
    
    #initiate the static values of the edge
    def initiate_edge(self, df):
        rows = df[(df['Ankomstplats'] == self.i) | (df['Avgångsplats'] == self.i)] #trains that either arrive at i or depart at i. TODO: Should do some check on timings also?
        self.pij = self.get_pij(rows)
        self.rij = self.get_rij(df)
        
        rows = df[(df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)] #trains that depart from i and arrive at j
        #getting values for the frequency and average travel time for each hour of the day
        frequency_dict, time_span_dict = self.get_time_span_dicts(rows)

        self.fijs = frequency_dict
        self.tijs = time_span_dict
        return

    #calculates the average travel time for all trains that travel from station i to j during the input time span
    #input rows should be the trains that travel from station i to j
    def get_average_travel_time(self, rows, time_span): 
        rows = rows.dropna(subset=['UtfAnkTid', 'UtfAvgTid']) #remove rows that do not have a departure time or arrival time
        rows = rows[(rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1], inclusive='left'))] # filtering to only include trains that depart during the time span
              
        if len(rows) == 0: #return 0 if there are no 
            return 0
        
        time_diff = rows['UtfAnkTid'] - rows['UtfAvgTid'] #calculate the time difference between arrival and departure
        time_diff = time_diff.dt.total_seconds() / 60 
        
        mean_time_diff = time_diff.mean()
        rounded = np.round(mean_time_diff, 2) #round to two decimals

        return rounded #in minutes
    
    #calculates the probability of a train that has gone to i, to continue to j
    #rows should be the trains that either arrive at i or depart at i
    def get_pij(self, rows): 
        trains_to_i = rows[rows['Ankomstplats'] == self.i]#all of the train errands that are going to i
        
        train_errands = trains_to_i['Tåguppdrag'].unique() #getting the unique errands from the trains that are going to i
        incoming_trains_count = 0
        outgoing_i_to_j_count = 0
        
        for errand in train_errands:
            #all trains that are going to i with the errand from the list
            current_errand_train = trains_to_i[trains_to_i['Tåguppdrag'] == errand] 
            incoming_trains_count += len(current_errand_train)

            current_errand_train_from_i_to_j = rows[(rows['Tåguppdrag'] == errand) & (rows['Avgångsplats'] == self.i) & (rows['Ankomstplats'] == self.j)]
            outgoing_i_to_j_count += len(current_errand_train_from_i_to_j)

        if incoming_trains_count == 0: #to avoid division by 0
            return 0
        return outgoing_i_to_j_count/incoming_trains_count

    # calculates the probability of a train that has gone to i, to continue to j if they do not end at i
    # df should be the whole dataframe
    def get_rij(self, df): 
        trains_that_pass_i = df[(df['Ankomstplats'] == self.i) & (df['UppehållstypAnkomst'] != 'Sista')]
        if len(trains_that_pass_i) == 0: # to avoid division by 0
            return 0
        
        train_errands = trains_that_pass_i['Tåguppdrag'].unique() #getting the unique errands from the trains that are going to i
        trains_to_j_count = 0

        for errand in train_errands:
            #all trains that are going to i with the errand from the list
            current_errand_train = df[(df['Tåguppdrag'] == errand) & (df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)]
            trains_to_j_count += len(current_errand_train)
        
        rij = trains_to_j_count/len(trains_that_pass_i)
        return rij
    
    #function to calculate the frequency of trains from i to j at a specific time span, in minutes
    #minutes should be the time span in minutes
    #rows should be all trains that depart from station i to station j
    def get_fij(self, rows, time_span, minutes):
        rows = rows[(rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1], inclusive='left'))]
        
        freq = len(rows)/minutes
        return freq

    #function to get the frequency of trains from i to j at a specific time span, stores the frequency in a dict with the time span as key (start, end) and the frequency as value
    def get_time_span_dicts(self, rows):
        frequency_dict = {}
        avg_time_dict = {}
        
        minutes = 60 #minutes between each time span, right now it is set to 60 minutes
        
        for i in range(24): # If you change this, also change variable "minutes" to the correct time span
            start_time = pd.to_datetime(f"2019-03-31 {i}:00:00.000").time()
            if i == 23:
                end_time = pd.to_datetime(f"2019-03-31 00:00:00.000").time()
            else:
                end_time = pd.to_datetime(f"2019-03-31 {i+1}:00:00.000").time()
         
            #initiate values as 0
            frequency_dict[(start_time, end_time)] = 0 
            avg_time_dict[(start_time, end_time)] = 0
        
        #populating frequency dict
        for time_span in frequency_dict.keys():
            freq = self.get_fij(rows, time_span, minutes)
            frequency_dict[time_span] = freq
        
        #populating time span dict
        for time_span in avg_time_dict.keys():
            avg_time = self.get_average_travel_time(rows, time_span)
            avg_time_dict[time_span] = avg_time
        
        return frequency_dict, avg_time_dict
    
############################################################################################################
        
class Network:
    def __init__(self):
        self.N = 0 # number of stations
        self.stations = {} # dictionary of stations {name: Station}
        self.edges = {} #dict of edges {(start, end): Edge}
        self.station_indicies = None  # station_indicies = {station: idx}
        self.A_matrix = None # adjacency matrix for the network
        self.G_matrices = None #dict of G matrices for each hour of the day
        self.D_matrix = None #delay matrix for the network currently at time step
        self.current_time = None #the current time of the network
        self.time_step = None #time step of the network, delta t
    
    #Initiates the network.
    def initate_network(self, df, time_step = 1):
        
        # Comverts the times from string to datetime
        df['UtfAnkTid'] = pd.to_datetime(df['UtfAnkTid'])
        df['UtfAvgTid'] = pd.to_datetime(df['UtfAvgTid'])
    
        self.time_step = time_step
        network_start_time = self.extract_start_time(df) 
        self.current_time = network_start_time
        self.extract_stations(df)
        self.extract_edges(df)

        for station in self.stations:
            self.stations[station].initiate_Bis(self.edges)
        self.extract_G_matrices()
        self.extract_D_matrix()
        return
    
    # Extract the first time that exists in the data
    def extract_start_time(self, df):
        all_times = df['UtfAvgTid'].dropna()
        first_time_stamp = all_times.min()
        return pd.to_datetime(first_time_stamp)

    # Calculates the D matrix by going through the station and extracts the delays
    def extract_D_matrix(self):
        D_matrix = np.zeros((self.N,1))
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            #station delay matrix is the delay at the station at the start time
            D_matrix[row_index] = station.delay
        self.D_matrix = D_matrix
        return

    # Function that executes a time step 
    def call_time_step(self):
        
        matrix_keys = list(self.G_matrices.keys()) # Get all the time intervals 
        current_time = self.current_time.time()
        
        for i in range(len(matrix_keys)): # Goes through all the time intervalls
            if current_time >= matrix_keys[i][0] and current_time < matrix_keys[i][1]: # checks which interval current time is in 
                G_matrix = self.G_matrices[matrix_keys[i]]#  Extracts the correct G matrix that is in to the correct interval
                break
            
        current_delday = self.D_matrix
    
        #matrix multiplication of G and D
        difference = np.matmul(G_matrix, current_delday)
       
        #adds the difference to the current delay
        self.D_matrix = current_delday + difference

        # Goes throigh all all stations and updates the new delay
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            station.delay = self.D_matrix[row_index]
            
        #add time step to the current time
        self.current_time += pd.DateOffset(minutes = self.time_step,)
        return

    # Creates and add station to the network
    def add_station(self, name, id, df):
        station = Station(name, id)
        station.initiate_station(df, self.current_time)
        self.stations[name] = station
        self.N += 1
        return

    # Creates and add edge to the network
    def add_edge(self, id, start, end, df, adj_matrix):
        edge = Edge(id, start, end, adj_matrix)
        edge.initiate_edge(df)
        key = (start,end)
        self.edges[key] = edge
        return
    
    # Find all stations in the data and add them to the network
    def extract_stations(self, df):
        stations_depart = df['Avgångsplats'].unique()
        stations_arrive = df['Ankomstplats'].unique()
        stations = set(stations_depart).union(set(stations_arrive))
        
        # Goes through all stationss in the data and add them to the network
        for i, station in enumerate(stations):
            self.add_station(station, i, df)
        
        # For every station find all the incoming and outgoing neighbours
        for station_name in self.stations:
            #add neighbours
            neighbours_in_names = df[df['Ankomstplats'] == station_name]
            neighbours_in_names = neighbours_in_names['Avgångsplats'].unique()
            neighbours_in = [self.stations[name] for name in neighbours_in_names]

            neighbours_out_names = df[df['Avgångsplats'] == station_name]
            neighbours_out_names = neighbours_out_names['Ankomstplats'].unique()
            neighbours_out = [self.stations[name] for name in neighbours_out_names]
        
            station = self.stations[station_name]
            station.set_N_in(neighbours_in)
            station.set_N_out(neighbours_out)
        return

    # Creates the adjacency matrix of the network, that is 1 if the stations are connected with an edge, 0 otherwise
    def create_adjacency_matrix(self, edges):
        station_indices = {station: idx for idx, station in enumerate(self.stations)} # All stations in the network
        self.station_indicies = station_indices
        n = self.N
        adj_matrix = np.zeros((n, n), dtype=int) # Creates an empty matrix 

        for edge in edges:
            depart_station = edge[0] 
            arrive_station = edge[1]

            # Finds the index of the depart and arrival stations
            i = station_indices[depart_station]
            j = station_indices[arrive_station]

            # Inputs 1 at the correct index
            adj_matrix[i][j] = 1
        
        adj_matrix_df = pd.DataFrame(adj_matrix, index=self.stations, columns=self.stations)

        return adj_matrix_df
    
    # Extract all edges from the data and add them to the network
    def extract_edges(self, df):
        unique_edges = df[['Avgångsplats', 'Ankomstplats']].drop_duplicates()
        
        # Convert to a list of lists 
        edges = unique_edges.values.tolist()
        
        adj_matrix = self.create_adjacency_matrix(edges)
        self.A_matrix = adj_matrix

        # for every edge, add it to the network
        for i, edge in enumerate(edges):
            self.add_edge(i, edge[0], edge[1], df, adj_matrix.loc[edge[0]][edge[1]])
        return

    # Function that extracts the G matrices for each hour of the day
    def extract_G_matrices(self):
        G_matrices = {}
        n = self.N
        
        for i in range(24): 
            start_time = pd.to_datetime(f"2019-03-31 {i}:00:00.000").time()
            if i == 23:
                end_time = pd.to_datetime(f"2019-03-31 00:00:00.000").time()
            else:
                end_time = pd.to_datetime(f"2019-03-31 {i+1}:00:00.000").time()
         
            #initiates the G matrix for the current time span, all values are 0 at the start
            G_matrices[(start_time, end_time)] = np.zeros((n, n), dtype=float)
        
        #populating the G matrices
        for time_span in G_matrices.keys():
            G_matrices[time_span] = self.calculate_G_matrix(G_matrices[time_span], time_span, n)
       
        
        self.G_matrices = G_matrices
        return

    # Function that calculates the G matrix for a specific time span
    # G_matrix should be an nxn empty matrix, time span should be a tuple of two times in Timestamp format only (0,1), (1,2) etc
    def calculate_G_matrix(self, G_matrix, time_span, n):
        #looping through all stations
        for station_name, row_index in self.station_indicies.items():
            i_station = self.stations[station_name] #station object of the current row
            #going throgh all stations again for the values of the G matrix columns for the current row
            for col_index in range(n):
                Aji = self.A_matrix.iloc[col_index][row_index]
                j_station_name = [key for key, value in self.station_indicies.items() if value == col_index][0]
                j_station = self.stations[j_station_name] #stati
                pji = 0
                if Aji == 1: #if there is an edge from j to i, we extraact the values of the edge
                    edge_ji = self.edges[(j_station.name, i_station.name)]
                    pji = edge_ji.pij
                    
                Bj = j_station.fetch_Bi(time_span)
                
                #kroneker value should only be 1 if the row and col stations are the same
                kronecker = 1 if col_index == row_index else 0
                #calculating the value of the G matrix at the current row and column
                value = Aji * pji * Bj - Bj * kronecker
                G_matrix[row_index][col_index] = value
        return G_matrix

    # Function that prints the station information of a specific station
    def print_station_info(self, station_name):
        station = self.stations[station_name]
        print(f"Station: {station_name}")
        print(f"Neighbours in: {[neighbour.name for neighbour in station.N_in]}")
        print(f"Neighbours out: {[neighbour.name for neighbour in station.N_out]}")
        print(f"si: {station.si}")
        print(f"Bi: {station.Bis}")
        print(" ")
        return

    # Function that prints the edge information
    def print_edge_info(self, start, end):
        key = (start,end)
        edge = self.edges[key]
        print(f"Edge from {start} to {end}")
        print(f"Travel time: {edge.tijs}")
        print(f"pij: {edge.pij}")
        print(f"rij: {edge.rij}")
        print(f"fij: {edge.fijs}")
        print(f"Aij: {edge.Aij}")
        print(" ")
        return

    def print_delay_matrix(self):
        delay_matrix= self.D_matrix
        print("Delay matrix at time: ", self.current_time)
        
        for station_name, row_index in self.station_indicies.items(): 
            print(f"{station_name}: {delay_matrix[row_index][0]}")
        return
    
    # Function that prints the network information
    def print_network_info(self):
        for station_name in self.stations:
            self.print_station_info(station_name)
        for edge in self.edges:
            start = edge[0]
            end = edge[1]
            self.print_edge_info(start, end)
        print(self.A_matrix)
        self.print_delay_matrix()
        return
