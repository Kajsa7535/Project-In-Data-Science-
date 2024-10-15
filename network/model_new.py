# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
class Station:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.delay = 0 # total delay at the station at the current time step, maybe this needs to be a list so we store the delay for each time step
        self.T_ij = []  # set of trains moving to station i at time t #TODO
        self.N_out = [] # set of stations to which there is a edge from station i (neightbours out)
        self.N_in = [] # set of stations from which there is an edge to station i (neighbours in)
        self.Bi = None # turnover rate
        self.si = None #fraction of trains on the edge towards this station that end at this station

    def initiate_station(self, df, edges, network_start_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]
        self.si = self.get_si(rows)
        self.delay = self.update_delay(df, network_start_time)

    def update_delay(self, df, network_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]        
        rows = rows[(rows['UtfAnkTid'] >= network_time) & (rows['UtfAvgTid'] <= network_time)]
        columns_to_print = ['UtfAnkTid', 'UtfAvgTid', 'AvgFörsening']
        filtered_rows = rows[columns_to_print]
        if len(rows) == 0:
            return 0
        #sum of all of the incoming trains delays
        delay = rows['AvgFörsening'].sum()
        return delay


    def get_si(self, rows):
        #rows are all rows that are going to this station
        if len(rows) == 0:
            return 0
        final_station_rows = rows[rows['UppehållstypAnkomst'] == 'Sista']
        total_rows = len(rows)
        final_rows = len(final_station_rows)
        return final_rows/total_rows
    
    def initiate_Bi(self, edges):
        #takes all edges that are incoming to this station, divides the sum of frequencies by the sum of frequencies times the average time
        incomming_edges = [value for key, value in edges.items() if key[1] == self.name]
        sum_of_freq = 0
        sum_of_freq_and_time = 0
        for edge in incomming_edges:
            freq = edge.fetch_fij()
            avg_time = edge.fetch_tij()
            sum_of_freq += freq
            sum_of_freq_and_time += freq * avg_time
        if sum_of_freq_and_time == 0:
            self.Bi =  0
        else:
            self.Bi = sum_of_freq/sum_of_freq_and_time

    def set_N_in(self, neighbours_in):
        self.N_in = neighbours_in
    
    def set_N_out(self, neighbours_out):
        self.N_out = neighbours_out
    

# -

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

# +
import pandas as pd
import numpy as np

class Edge: 
    def __init__(self, id, start, end, adj_number):
        self.id = id
        self.i = start #station i = start
        self.j = end # station j = end
        self.Aij = adj_number #adjacency matrix for all edges in network
        self.fij = None # frequency of trains on this edge i to j
        self.tij = None # travel time on this edge average i to j
        self.pij = None # fraction of trains to i that continues to j. It is a probability.
        self.rij = None # fraction of trains to i that continue to j if they do not end at i
    
    def fetch_tij(self):
        return self.tij
    
    def fetch_fij(self):
        return self.fij
    
    def initiate_edge(self, df, time_span):
        #average travel time on edge i to j
        rows = df[(df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)]
        
        average_travel_time = self.get_average_travel_time(rows, time_span)
        self.tij = average_travel_time

        #fraction of trains that continue from i to j
        rows = df[(df['Ankomstplats'] == self.i) | (df['Avgångsplats'] == self.i)] #trains that arrive at i and depart at i. TODO: Should do some check on timings also?
        self.pij = self.get_pij(rows)

        # fraction of trains that continue from i to j if they do not end at i
        self.rij = self.get_rij(df)
        rows = df[(df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)]
        self.fij = self.get_fij(rows, time_span)
    

    def get_average_travel_time(self, rows, time_span): 
        #rows are the trains that travel from station i to j
        rows = rows.dropna(subset=['UtfAnkTid', 'UtfAvgTid'])

        rows = rows[(rows['UtfAnkTid'].dt.time.between(time_span[0], time_span[1])) & 
                (rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1]))]
        #print UtfAnkTid and UtfAvgTid
        time_diff = rows['UtfAnkTid'] - rows['UtfAvgTid']

        time_diff = time_diff.dt.total_seconds() / 60
        mean_time_diff = time_diff.mean()
        rounded = np.round(mean_time_diff, 2)

        return rounded #in minutes
    
    def get_pij(self, rows): #for all trains that have gone to i, calculate the probability of going to j. 
        #rows are the trains that arrive  at station i. 
        trains_to_i = rows[rows['Ankomstplats'] == self.i]
        #all of the train errands that are going to i
        train_errands = trains_to_i['Tåguppdrag'].unique()
        incoming_trains_count = 0
        outgoing_i_to_j_count = 0

        for errand in train_errands:
            #all trains that are going to i with the errand from the list
            current_errand_train = trains_to_i[trains_to_i['Tåguppdrag'] == errand] 
            incoming_trains_count += len(current_errand_train)

            current_errand_train_from_i_to_j = rows[(rows['Tåguppdrag'] == errand) & (rows['Avgångsplats'] == self.i) & (rows['Ankomstplats'] == self.j)]
            outgoing_i_to_j_count += len(current_errand_train_from_i_to_j)

        if incoming_trains_count == 0:
            return 0
        return outgoing_i_to_j_count/incoming_trains_count

    def get_rij(self, df): #for all trains that do not have i as end station, calculate the probability of those trains going to j (in a fraction)
        trains_that_pass_i = df[(df['Ankomstplats'] == self.i) & (df['UppehållstypAnkomst'] != 'Sista')]
        if len(trains_that_pass_i) == 0:
            return 0
        train_errands = trains_that_pass_i['Tåguppdrag'].unique()
        trains_to_j_count = 0

        for errand in train_errands:
            #all trains that are going to i with the errand from the list
            current_errand_train = df[(df['Tåguppdrag'] == errand) & (df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)]
            trains_to_j_count += len(current_errand_train)
        
        rij = trains_to_j_count/len(trains_that_pass_i)
        return rij

    def get_fij(self, rows, time_span):
        time_span_minute_conut = (time_span[1].hour - time_span[0].hour) * 60

        rows = rows[(rows['UtfAnkTid'].dt.time.between(time_span[0], time_span[1])) & 
                (rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1]))]

        #right now the freq is only calculated for one time unit which is now all the time we have
        #rows should be all trains that depart from station i to station j
        freq = len(rows)/time_span_minute_conut
        return freq
    

# +
import pandas as pd

class Network:
    def __init__(self):
        self.N = 0 # number of stations
        self.stations = {} # dictionary of stations {name: Station}
        self.edges = {} #dict of edges {(start, end): Edge}
        self.station_indicies = None
        self.A_matrix = None # adjacency matrix for the network
        self.G_matrix = None # topological structure matrix for the network
        self.D_matrix = None #delay matrix for the network
        self.current_time = None
        self.time_step = None
        self.time_span = None
    
    def initate_network(self, df, time_span=("2019-03-31 15:00:00.000","2019-03-31 19:00:00.000"), time_step = 1):
        #convert the columns UtfAnkTid and UtfAvgTid to datetime
        df['UtfAnkTid'] = pd.to_datetime(df['UtfAnkTid'])
        df['UtfAvgTid'] = pd.to_datetime(df['UtfAvgTid'])
        time_span = (pd.to_datetime(time_span[0]), pd.to_datetime(time_span[1]))
        time_span = (time_span[0].time(), time_span[1].time())
        self.time_span = time_span

        self.time_step = time_step
        network_start_time = self.extract_start_time(df)
        self.current_time = network_start_time
        self.extract_stations(df)
        self.extract_edges(df)

        for station in self.stations:
            self.stations[station].initiate_Bi(self.edges)
        self.extract_G_matrix()
        self.extract_D_matrix(df)
    
    def extract_start_time(self, df):
        all_times = df['UtfAvgTid'].dropna()
        first_time_stamp = all_times.min()
        return pd.to_datetime(first_time_stamp)


    def extract_D_matrix(self, df):
        # self.station_indicies = {station: idx}
        D_matrix = np.zeros((self.N,1))
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            #station delay matrix is the delay at the station at the start time
            D_matrix[row_index] = station.delay
            print("Station name and idex: ", station_name, row_index)
        self.D_matrix = D_matrix
        #loop through all stations and calculate the delay

    def call_time_step(self):
        current_delday = self.D_matrix
        G_matrix = self.G_matrix
        #do matrix multiplication of G and D
        difference = np.matmul(G_matrix, current_delday)
        #add the difference to the current delay
        self.D_matrix = current_delday + difference
        #print the delays
        for station_name, row_index in self.station_indicies.items():
            station = self.stations[station_name]
            station.delay = self.D_matrix[row_index]
        #add time step to the current time
        self.current_time += pd.DateOffset(minutes = self.time_step,)
        print("Delay matrix at time: ", self.current_time)
        print(self.D_matrix)
        print(" --------------------------------------------")
        print(" ")

        

    def add_station(self, name, id, df, edges):
        station = Station(name, id)
        station.initiate_station(df, edges, self.current_time)
        self.stations[name] = station
        self.N += 1

    def add_edge(self, id, start, end, df, adj_matrix):
        edge = Edge(id, start, end, adj_matrix)
        edge.initiate_edge(df, self.time_span)
        key = (start,end)
        self.edges[key] = edge
    
    def extract_stations(self, df):
        stations_depart = df['Avgångsplats'].unique()
        stations_arrive = df['Ankomstplats'].unique()
        stations = set(stations_depart).union(set(stations_arrive))
        for i, station in enumerate(stations):
            self.add_station(station, i, df, self.edges)
        for station_name in self.stations:
            #station is the value from the dictionary, which is the sation object
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
    

    def extract_edges(self, df):
        unique_edges = df[['Avgångsplats', 'Ankomstplats']].drop_duplicates()
        # Convert to a list of lists 
        edges = unique_edges.values.tolist()
        
        adj_matrix = self.create_adjacency_matrix(edges)
        self.A_matrix = adj_matrix

        for i, edge in enumerate(edges):
            self.add_edge(i, edge[0], edge[1], df, adj_matrix.loc[edge[0]][edge[1]])


    def extract_G_matrix(self):
        #create a N x N matrix
        n = self.N
        G_matrix = np.zeros((n, n), dtype=float)
        for station_name, row_index in self.station_indicies.items():
            i_station = self.stations[station_name]
            for col_index in range(n):
                Aji = self.A_matrix.iloc[col_index][row_index]
                j_station_name = [key for key, value in self.station_indicies.items() if value == col_index][0]
                j_station = self.stations[j_station_name]
                pji = 0
                if Aji == 1:
                    edge_ji = self.edges[(j_station.name, i_station.name)]
                    pji = edge_ji.pij
                    
                Bj = j_station.Bi
                #print all variables
                kronecker = 1 if col_index == row_index else 0
                value = Aji * pji * Bj - Bj * kronecker
                G_matrix[row_index][col_index] = value
        self.G_matrix = G_matrix

    def print_station_info(self, station_name):
        station = self.stations[station_name]
        print(f"Station: {station_name}")
        print(f"Neighbours in: {[neighbour.name for neighbour in station.N_in]}")
        print(f"Neighbours out: {[neighbour.name for neighbour in station.N_out]}")
        print(f"si: {station.si}")
        print(f"Bi: {station.Bi}")
        print(" ")

    
    def print_edge_info(self, start, end):
        key = (start,end)
        edge = self.edges[key]
        print(f"Edge from {start} to {end}")
        print(f"Travel time: {edge.tij}")
        print(f"pij: {edge.pij}")
        print(f"rij: {edge.rij}")
        print(f"fij: {edge.fij}")
        print(f"Aij: {edge.Aij}")
        print(" ")

    def print_network_info(self):
        for station_name in self.stations:
            self.print_station_info(station_name)
        for edge in self.edges:
            start = edge[0]
            end = edge[1]
            self.print_edge_info(start, end)
        print(self.A_matrix)
        print(self.G_matrix)
        print("Delay matrix at time: ", self.current_time)
        print(self.D_matrix)

