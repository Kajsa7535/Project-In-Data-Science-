
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
import matplotlib.pyplot as plt
import seaborn as sns
import pygraphviz as pgv

pd.set_option('display.max_rows', None)  # Show all rows


class Station:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.delay = 0 # total delay at the station at the current time step, maybe this needs to be a list so we store the delay for each time step
        self.delay_origins = [] # list of delays for each train that is going to this station
        self.T_ij = []  # set of trains moving to station i at time t #TODO
        self.N_out = [] # set of stations to which there is a edge from station i (neightbours out)
        self.N_in = [] # set of stations from which there is an edge to station i (neighbours in)
        self.Bis = None # turnover rate dict of Bi for each hour of the day
        self.si = None #fraction of trains on the edge towards this station that end at this station

    def initiate_station(self, df, network_start_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]
        self.si = self.get_si(rows)
        self.delay = self.calculate_delay(df, network_start_time)
        return

    # Function that finds the delay of the station at the current network time
    def calculate_delay(self, df, network_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]        
        rows = rows[(rows['UtfAnkTid'] > network_time) & (rows['UtfAvgTid'] <= network_time)]
        if len(rows) == 0:
            return 0
        #sum of all of the incoming trains delays
        delay = rows['AvgFörsening'].sum()
        if abs(delay) > 0:
            delayed_rows = rows[abs(rows['AvgFörsening']) > 0]
            for row in delayed_rows.iterrows():
                delay_item = ((row[1]['Avgångsplats'], row[1]['Ankomstplats']), row[1]['AvgFörsening']) #((start, end), delay)
                if delay_item not in self.delay_origins:
                    self.delay_origins.append(delay_item)
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
    
    def print_Bi(self):
        print("Bi:")
        for key, value in self.Bis.items():
            start_time = key[0].hour  # Extract the starting hour
            end_time = key[1].hour    # Extract the ending hour
            print(f"{start_time}-{end_time} Bi: {value}")


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

    #calculates the average travel time at a station for all trains that travel from station i to j during the input time span
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
        if rounded == 0: #if the mean time difference is 0, return 1 to avoid losing the edge
            return 1
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
    
    #function to print the frequencies of the edge
    def print_frequencies(self):
        print("Frequencies:")
        for key, value in self.fijs.items():
            start_time = key[0].hour  # Extract the starting hour
            end_time = key[1].hour    # Extract the ending hour
            print(f"{start_time}-{end_time} fij: {value}")
        return
    
    #function to print the travel times of the edge
    def print_travel_times(self):
        print("Travel times:")
        for key, value in self.tijs.items():
            start_time = key[0].hour
            end_time = key[1].hour
            print(f"{start_time}-{end_time} tij: {value}")
        return
    
############################################################################################################
        
class Network:
    def __init__(self):
        self.N = 0 # number of stations
        self.stations = {} # dictionary of stations {name: Station}
        self.edges = {} #dict of edges {(start, end): Edge}
        self.station_indicies = None  # station_indicies = {station: idx}
        self.A_matrix = None # adjacency matrix for the network
        self.G_matrices = None #dict of G matrices for each hour of the day
        self.D_matrix = None #delay matrix for the network at the current time step
        self.D_matrices = None #list that holds each delay matrix with its corresponding G_matrices for the network currently at time step [(D_matrix, [G_matrices]), (D_matrix, [G_matrices])...]
        self.current_time = None #the current time of the network
        self.time_step = None #time step of the network, delta t
    
    #Initiates the network.
    def initate_network(self, df, time_step = 1, network_start_time = None):
        
        # Comverts the times from string to datetime
        df['UtfAnkTid'] = pd.to_datetime(df['UtfAnkTid'])
        df['UtfAvgTid'] = pd.to_datetime(df['UtfAvgTid'])
    
        self.time_step = time_step
        if network_start_time is None:
            network_start_time = self.extract_start_time(df) 
        self.current_time = network_start_time
        self.extract_stations(df)
        self.extract_edges(df)

        for station in self.stations:
            self.stations[station].initiate_Bis(self.edges)
        self.extract_G_matrices()
        self.extract_D_matrix()
        self.extract_D_matrices()
        return
    
    # Extract the first time that exists in the data
    def extract_start_time(self, df):
        #TODO add a check for the first time that includes delay
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
    
    # Calculates the D matrix by going through the station and extracts the delays
    def extract_D_matrices(self):
        D_matrices = []
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            #station delay matrix is the delay at the station at the start time
            if len(station.delay_origins) > 0:
                delays = station.delay_origins
                for delay_item in delays:
                    current_thread_D_matrix = np.zeros((self.N,1))
                    delay = delay_item[1]
                    current_thread_D_matrix[row_index] = delay
                    delay_edge = delay_item[0]
                    print("DELAY EDGE: ", delay_edge)
                    directed_A_matrix, removed_edges = self.create_directed_A_matrix(delay_edge)
                    G_matrices = self.create_directed_G_matrices(directed_A_matrix, removed_edges)
                    D_matrices.append([current_thread_D_matrix, G_matrices])     
        self.D_matrices = D_matrices
        return
    
    
    # Calculates and returns the D matrix for the current time from the data in df
    def fetch_D_matrix(self, df):
        D_matrix = np.zeros((self.N,1))

        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            station_delay = station.calculate_delay(df, self.current_time)
            #station delay matrix is the delay at the station at the start time
            D_matrix[row_index] = station_delay
        return D_matrix

    # Function that creates a directed A matrix from the original A matrix
    def create_directed_A_matrix(self, directed_edge):
        # Method
        # Remove the outgoing edge that the delay was on 
        # Remove all incoming edges on the current station
        # Move to all the connected stations except the station where the delay came from, and do the same to them. 
        #step 1 remove the edge ( start, end)
        #step 2 for each item in frontier: remove all 1:s in end column expcet for the start row 
        # add all outgoing edges from current item to frontier (add all 1:s in the row for the current item end station)
        removed_edges = []
        removed_edges_with_count_dict = {}
        directed_A_matrix = self.A_matrix.copy()
        current_edge = directed_edge
        directed_A_matrix.loc[current_edge[1],current_edge[0]] = 0

        removed_first_edge = (current_edge[1], current_edge[0])
        removed_edges.append(removed_first_edge)
        froniter = [current_edge] #list of nodes to visit
        visited_notes= [] #list of nodes that have been visited
        while len(froniter) > 0:
            current_edge = froniter.pop(0) #(start, end)
            visited_notes.append(current_edge[1])
            
            #change all 1:s in the end column to 0 except for the start row
            prev_incomming_edge_value = directed_A_matrix.loc[current_edge[0],current_edge[1]] 
            current_col=  directed_A_matrix.loc[:, current_edge[1]]
            #get the row index wherre value is >0
            incomming_edges = current_col[current_col >= 1].index
            #remove index of the start station
            current_node_removed_edges = [x for x in incomming_edges if x != current_edge[0]]
            test_item = [x for x in removed_edges if x[0] == current_edge[1]]
            for item in test_item:
                removed_edges.remove(item)
                removed_edges_with_count_dict[item] = len(current_node_removed_edges)

            directed_A_matrix.loc[:, current_edge[1]] = 0
            directed_A_matrix.loc[current_edge[0],current_edge[1]] = prev_incomming_edge_value

            outgoings_row = directed_A_matrix.loc[current_edge[1]] # The row for the current endstation
            
            outgoings_indicies = outgoings_row[outgoings_row >= 1].index 
            outgoings_edges_count = len(outgoings_indicies)

            for station_name in current_node_removed_edges:
                current_removed_edge = (station_name, current_edge[1])
                removed_edges.append(current_removed_edge)

            if outgoings_edges_count != 0:
                weight = 1/outgoings_edges_count
            else: 
                weight = 0
            for index in outgoings_indicies:
                directed_A_matrix.loc[current_edge[1], index] += weight
        

            new_edges = [(current_edge[1], x) for x in outgoings_indicies if x != current_edge[0]] 
            new_edges = [x for x in new_edges if x[1] not in visited_notes]
            new_edges = [x for x in new_edges if x[1] not in [y[1] for y in froniter]]
            froniter += new_edges
        return directed_A_matrix, removed_edges_with_count_dict

    # Function that executes a time step 
    def predict_time_step(self):
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

     # Function that executes a time step 
    def predict_time_step_with_direction(self):
        matrix_keys = list(self.G_matrices.keys()) # Get all the time intervals 
        current_time = self.current_time.time()
        
        for delay_thread in self.D_matrices: 
            current_delay = delay_thread[0]
            G_matrices = delay_thread[1]
        
            for i in range(len(matrix_keys)): # Goes through all the time intervalls
                if current_time >= matrix_keys[i][0] and current_time < matrix_keys[i][1]: # checks which interval current time is in 
                    G_matrix = G_matrices[matrix_keys[i]]#  Extracts the correct G matrix that is in to the correct interval
                    break
                
            G_matrix = pd.DataFrame(G_matrix, index=self.stations, columns=self.stations)
            difference = np.matmul(G_matrix, current_delay)
            difference = np.array(difference)

            delay_thread[0] = current_delay + difference

        total_delay = np.zeros((self.N,1))
        for delay_thread in self.D_matrices: #sum all individual delay matrices
            total_delay += delay_thread[0]
            
        # Goes throigh all all stations and updates the new delay
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            station.delay = total_delay[row_index]

        self.D_matrix = total_delay #update the delay matrix
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
        adj_matrix = np.zeros((n, n), dtype=float) # Creates an empty matrix 

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
    
    # Function that extracts the G matrices for each hour of the day
    def create_directed_G_matrices(self, A_matrix, removed_edges):
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
            G_matrices[time_span] = self.calculate_G_matrix(G_matrices[time_span], time_span, n, A_matrix = A_matrix, removed_edges = removed_edges)
       
        return G_matrices

    # Function that calculates the G matrix for a specific time span
    # G_matrix should be an nxn empty matrix, time span should be a tuple of two times in Timestamp format only (0,1), (1,2) etc
    def calculate_G_matrix(self, G_matrix, time_span, n, A_matrix = None, removed_edges = None):
        if A_matrix is None:
            A_matrix = self.A_matrix
        #looping through all stations
        if removed_edges:
            removed_edges_keys = removed_edges.keys()
        else:
            removed_edges_keys = []
        for station_name, row_index in self.station_indicies.items():

            i_station = self.stations[station_name] #station object of the current row
            #going throgh all stations again for the values of the G matrix columns for the current row
            for col_index in range(n):
                Aji = A_matrix.iloc[col_index,row_index]
                j_station_name = [key for key, value in self.station_indicies.items() if value == col_index][0]
                j_station = self.stations[j_station_name] #stati
                pji = 0
                value_to_add = 0
                if Aji >= 1: #if there is an edge from j to i, we extraact the values of the edge
                    Bj = j_station.fetch_Bi(time_span)
                    edge_ji = self.edges[(j_station.name, i_station.name)]
                    pji = edge_ji.pij
                    Aji = 1 # set the value to 1 to make sure the node is weighted as 1
                    
                    #gå igenom removed edges och hitta edges som har samma start som j_station_name
                    #rmoved edeges keys [(start, end), (start, end)]
                    
                    edges_to_add = [x for x in removed_edges_keys if x[0] == j_station_name]
                    for edge in edges_to_add:
                        keys_edges = self.edges.keys()
                        if edge in keys_edges:
                            edge_object = self.edges[edge]
                            value_to_add += edge_object.Aij * edge_object.pij * Bj
        
                Bj = j_station.fetch_Bi(time_span)
                
                #kroneker value should only be 1 if the row and col stations are the same
                kronecker = 1 if col_index == row_index else 0
                #calculating the value of the G matrix at the current row and column
                value = Aji * pji * Bj - Bj * kronecker + value_to_add
                G_matrix[row_index][col_index] = value
        return G_matrix
    
    # Evaluates the network against the actual data.
    def evaluate_network(self, df, time_steps, visualize = False):
        print("Evaluating network")
        print("Time steps: ", time_steps)
        print("Network start time: ", self.current_time)
        print("-------------------------")

        for step in range(time_steps):

            true_delay = self.fetch_D_matrix(df) #matrix that holds the true delay of the data in df
            true_delay = np.round(true_delay, 3)#round the delay matrix to 3 decimals
            
            self.predict_time_step_with_direction() # predict the delay matrix with directions
            predicted_delay = self.D_matrix #predicted delay matrix
            predicted_delay = np.round(predicted_delay, 3) #round the delay matrix to 3 decimals

            if visualize:
                self.visualize_comparative_delays(true_delay, predicted_delay, step, title="Comparative Delay Visualization", cap=0.25)

            comparison = np.concatenate((true_delay, predicted_delay), axis=1) # Prints the delaymatrix so both values are side by side 
            comparison = np.round(comparison, 3) #round the comparison matrix to 3 decimals
            comparison_matrix = np.zeros((self.N,1))
            for i in range(self.N):
                comparison_matrix[i] = true_delay[i] - predicted_delay[i]
            comparison = np.concatenate((comparison, comparison_matrix), axis=1)
            self.print_comparison_delay_matrix(comparison, print_all=False)
            print(" ")
        return


    # Function that prints the station information of a specific station
    def print_station_info(self, station_name):
        station = self.stations[station_name]
        print(f"Station: {station_name}")
        print(f"Neighbours in: {[neighbour.name for neighbour in station.N_in]}")
        print(f"Neighbours out: {[neighbour.name for neighbour in station.N_out]}")
        print(f"si: {station.si}")
        station.print_Bi()
        print(" ")
        return

    # Function that prints the edge information
    def print_edge_info(self, start, end):
        key = (start,end)
        edge = self.edges[key]
        print(f"Edge from {start} to {end}")
        edge.print_travel_times()
        print(f"pij: {edge.pij}")
        print(f"rij: {edge.rij}")
        edge.print_frequencies()
        print(f"Aij: {edge.Aij}")
        print(" ")
        return

    # Print the delay matrix
    def print_delay_matrix(self, print_all = True, delay_matrix = None, cap=0.25):
        if delay_matrix is None:
            delay_matrix = self.D_matrix
        print("Delay matrix at time: ", self.current_time)
        if print_all: 
            for station_name, row_index in self.station_indicies.items(): 
                print(f"{station_name}: {delay_matrix[row_index][0]}")
        else: 
            for station_name, row_index in self.station_indicies.items(): 
                if(abs(delay_matrix[row_index][0]) >= cap):
                    print(f"{station_name}: {delay_matrix[row_index][0].round(3)}")
            
        return
    
    # Print the comparison delay matrix. That is the matrix that includes the actual delay, the predicted delay and the difference between them. 
    def print_comparison_delay_matrix(self, comparison_delay_matrix, print_all = True,cap=0.25):
        print("Delay matrices at time: ", self.current_time)
        print("Station: True delay, Predicted delay, Difference")
        
        name_array = np.array(list(self.station_indicies.keys())) #array of station names
        comparison_df = pd.DataFrame(comparison_delay_matrix, index = name_array, columns = ["True delay", "Predicted delay", "Difference"]) #put in a dataframe for easier visualization
        comparison_df= comparison_df.sort_index()        
        
        if print_all: 
           print(comparison_df)
        else: 
            #filtering out the rows that have 0 in both columns
            comparison_df = comparison_df[(comparison_df["True delay"] != 0) | (abs(comparison_df["Predicted delay"]) >= cap)]
            print(comparison_df)
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
    
   
    #Visualizes actual vs. predicted delays using network graphs.
    def visualize_comparative_delays(self, actual_delay, predicted_delay, step, title="Comparative Delay Visualization", cap=0.25):
        # Function to map delays to colors
        def get_color(value, cap):

            #If the value is below the cap, returns a neutral color.
            if abs(value) <= cap:
                return "#D3D3D3"  # Neutral color for small delays (gray)

            # Color mapping for significant delays
            color_map = {
                -15: "#0D47A1", -10: "#1976D2", -5: "#42A5F5", 
                -3: "#90CAF9", -1: "#E3F2FD", 1: "#FFEBEE", 
                3: "#FFCDD2", 5: "#E57373", 10: "#D32F2F", 
                15: "#B71C1C"
            }
            
            if value < 0:
                for key in sorted(color_map.keys(), reverse=True):
                    if value >= key:
                        return color_map[key]
            else:
                for key in sorted(color_map.keys()):
                    if value <= key:
                        return color_map[key]

            return color_map[key]

        # Create two separate graphs: one for actual delays and one for predicted delays
        graphs = {"Actual Delay": actual_delay, "Predicted Delay": predicted_delay}
        
        for graph_title, delay_array in graphs.items():
            G = pgv.AGraph(directed=True)
            # Add nodes with colors based on actual or predicted delays
            for station_name, row_index in self.station_indicies.items():
                delay = delay_array[row_index][0]
                color = get_color(delay, cap)  # Use delay to determine color'
                G.add_node(station_name, style="filled", fillcolor=color, fontsize=10, margin="0.1,0.1")

            # Add edges from the network
            for (start, end), _ in self.edges.items():
                G.add_edge(start, end, penwidth=2, color="gray")

            # Adjust layout for aesthetics
           # G.graph_attr.update(rankdir="LR", nodesep="2.0", ranksep="1.5", splines="true", dpi="400")
            
            # Save the graph to a file
            output_path = f"images/{graph_title.lower().replace(' ', '_')}_step_{step + 1}.png"
            G.layout(prog="fdp")
            G.draw(output_path, format="png")
            print(f"{graph_title} graph saved to {output_path}")

        # Display the graphs side by side using matplotlib
        _, axes = plt.subplots(1, 2, figsize=(16, 8))
        for idx, (graph_title, _) in enumerate(graphs.items()):
            img_path = f"images/{graph_title.lower().replace(' ', '_')}_step_{step + 1}.png"
            img = plt.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(graph_title, fontsize=14)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        #save the plot as a png      
        plt.savefig(f"images/compared_delays_step_{step + 1}.png", dpi=300, bbox_inches='tight')
        return

