
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
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import copy
from network.utils import create_24_hour_dict, remove_initial_edge, process_current_edge, get_new_edges_to_frontier, calculate_G_matrix_values, find_current_time_G_matrix, get_color

pd.set_option('display.max_rows', None)  # Show all rows when printing dataframe, remove this if you want to limit the output


class Station:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.delay = 0 # total delay at the station at the current time step
        self.delay_origins = [] # list of delays for each train that is going to this station
        self.N_out = [] # set of stations to which there is a edge from station i (neightbours out)
        self.N_in = [] # set of stations from which there is an edge to station i (neighbours in)
        self.Bis = None # turnover rate dict of Bi for each hour of the day
        self.si = None #fraction of trains on the edge towards this station that end at this station, currently not used but it exists for future work.

    # Function that initiates the station with the data from the dataframe
    def initiate_station(self, df, network_start_time):
        self.delay = self.calculate_delay(df, network_start_time)
        return

    # Function that finds the delay of the station at the current network time
    def calculate_delay(self, df, network_time):
        #rows are all trains that are going to this station
        rows = df[df['Ankomstplats'] == self.name]
        #filtering to only include trains that arrive after the network time and before or during the network time. That is: the trains that are in movement at the network time       
        rows = rows[(rows['UtfAnkTid'] > network_time) & (rows['UtfAvgTid'] <= network_time)]
        if len(rows) == 0:
            return 0

        #remove negative delays 
        rows = rows[rows['AvgFörsening'] >= 0]
        #sum of all of the incoming trains delays
        delay = rows['AvgFörsening'].sum()

        #if any delay exists
        if delay > 0:
            delayed_rows = rows[rows['AvgFörsening'] > 0]
            for row in delayed_rows.iterrows():
                #get each individual delay and the origin of the delay
                delay_item = ((row[1]['Avgångsplats'], row[1]['Ankomstplats']), row[1]['AvgFörsening']) #((start, end), delay)
                if delay_item not in self.delay_origins:
                    #add the delay to the list of delays
                    self.delay_origins.append(delay_item)
        return delay
    
    # Creates and initiates Bi (the turnover rate) dictionary, for now creates one Bi for every hour (00 - 23)
    def initiate_Bis(self, edges): 
        #create a dictionary with 24 hours as keys and 0 as values
        bi_dict = create_24_hour_dict()
        for time_span in bi_dict.keys(): # For every hour, calculate the individual Bi
            bi_dict[time_span] = self.calculate_Bi(edges, time_span)
        
        #setting the Bi dict
        self.Bis = bi_dict
        return

    #Function that takes all edges that are incoming to this station, divides the sum of frequencies by the sum of frequencies times the average time
    def calculate_Bi(self, edges, time_span):
        # Get all edges that are incoming to this station
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

    # Function that finds fraction of trains on the edges towards this station that end at this station
    # Currently not used, but is created for future use.
    def calculate_si(self, rows):
        #rows are all rows that are going to this station
        if len(rows) == 0:
            return 0
        final_station_rows = rows[rows['UppehållstypAnkomst'] == 'Sista']
        total_rows = len(rows)
        final_rows = len(final_station_rows)
        return final_rows/total_rows

    def set_N_in(self, neighbours_in):
        self.N_in = neighbours_in
    
    def set_N_out(self, neighbours_out):
        self.N_out = neighbours_out

    def fetch_Bi(self, time_span):
        value = self.Bis[time_span]
        return value
    
    def fetch_Si(self):
        return self.si
    
    def print_Bi(self):
        print("Bi:")
        for key, value in self.Bis.items():
            start_time = key[0].hour  # Extract the starting hour
            end_time = key[1].hour    # Extract the ending hour
            print(f"{start_time}-{end_time} Bi: {value}")


#####################################################################################################

class Edge: 
    def __init__(self, id, start, end, adj_number):
        self.id = id # edge id
        self.i = start # station i = start
        self.j = end # station j = end
        self.Aij = adj_number # always 1 for edges that exists
        self.fijs = None # dict of frquencies for each hour of the day
        self.tijs = None # dict of average travel times for each hour of the day
        self.pij = None # fraction of trains to i that continues to j. It is a probability.
        self.rij = None # fraction of trains to i that continue to j if they do not end at i. Currently not used, but exists for future use
    
    #initiate the static values of the edge
    def initiate_edge(self, df):
        rows = df[(df['Ankomstplats'] == self.i) | (df['Avgångsplats'] == self.i)] #trains that either arrive at i or depart at i.
        self.pij = self.calculate_pij(rows)
        
        rows = df[(df['Avgångsplats'] == self.i) & (df['Ankomstplats'] == self.j)] #trains that depart from i and arrive at j
        #getting values for the frequency and average travel time for each hour of the day
        self.initiate_frequency_dict(rows)
        self.initiate_avg_travel_time_dict(rows)
        return
    
    #function to get the frequency of trains from i to j at a specific time span, stores the frequency in a dict with the time span as key (start, end) and the frequency as value
    def initiate_frequency_dict(self, rows):
        frequency_dict = create_24_hour_dict()
        
        #populating frequency dict
        for time_span in frequency_dict.keys():
            freq = self.calculate_fij(rows, time_span)
            frequency_dict[time_span] = freq
        
        self.fijs = frequency_dict
        return 
    
    #function to get the averiage travel time of trains from i to j at a specific time span, stores the time in a dict with the time span as key (start, end) and the frequency as value
    def initiate_avg_travel_time_dict(self, rows):
        avg_time_dict =  create_24_hour_dict()
        
        #populating time span dict
        for time_span in avg_time_dict.keys():
            avg_time = self.calculate_average_travel_time(rows, time_span)
            avg_time_dict[time_span] = avg_time
        
        self.tijs = avg_time_dict
        return

    #calculates the average travel time at a station for all trains that travel from station i to j during the input time span
    #input rows should be the trains that travel from station i to j
    def calculate_average_travel_time(self, rows, time_span): 
        rows = rows.dropna(subset=['UtfAnkTid', 'UtfAvgTid']) #remove rows that do not have a departure time or arrival time
        rows = rows[(rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1], inclusive='left'))] # filtering to only include trains that depart during the time span
              
        if len(rows) == 0: #return 0 if there are no 
            return 0
        
        time_diff = rows['UtfAnkTid'] - rows['UtfAvgTid'] #calculate the time difference between arrival and departure
        time_diff = time_diff.dt.total_seconds() / 60 #convert to minutes
        
        mean_time_diff = time_diff.mean()
        rounded = np.round(mean_time_diff, 2) #round to two decimals
        
        if rounded == 0: #if the mean time difference is 0, return 1 to avoid losing the edge - some rows have the same arrival and departure time which results in 0, but rounding to 1 to avoid losing the edge
            return 1
        
        return rounded #in minutes
    
    #calculates the probability of a train that has gone to i, to continue to j
    #rows should be the trains that either arrive at i or depart at i
    def calculate_pij(self, rows): 
        trains_to_i = rows[rows['Ankomstplats'] == self.i]#all of the trains that are going to i
        
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

    #function to calculate the frequency of trains from i to j at a specific time span, in minutes
    #rows should be all trains that depart from station i to station j
    #currently checking frequency per hour (60 min)
    def calculate_fij(self, rows, time_span, minutes=60): 
        rows = rows[(rows['UtfAvgTid'].dt.time.between(time_span[0], time_span[1], inclusive='left'))]
        freq = len(rows)/minutes
        return freq

    # calculates the probability of a train that has gone to i, to continue to j if they do not end at i
    # df should be the whole dataframe
    # currently not used, but is created for furture use
    def calculate_rij(self, df): 
        #getting the trains that pass i, meaning they arrive at i but do not end there
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
        self.station_indices = None  # station_indices = {station: idx}
        self.A_matrix = None # adjacency matrix for the network
        self.G_matrices = None #dict of G matrices for each hour of the day
        self.D_matrix = None #delay matrix for the network at the current time step
        self.D_matrices = None #list that holds each delay matrix with its corresponding G_matrices for the network currently at time step [(D_matrix, [G_matrices]), (D_matrix, [G_matrices])...]
        self.current_time = None #the current time of the network
        self.time_step_size = None #time step of the network, delta t
    
    #Initiates the network.
    def initiate_network(self, df, time_step_size = 1, network_start_time = None):
        
        # Converts the times from string to datetime
        df['UtfAnkTid'] = pd.to_datetime(df['UtfAnkTid'])
        df['UtfAvgTid'] = pd.to_datetime(df['UtfAvgTid'])
    
    
        self.time_step_size = time_step_size
        # If there is not input time, extract the first time in the network that has a delay
        if network_start_time is None: 
            network_start_time = self.extract_start_time(df) 

        #set the start time of the network and extract all stations and edges    
        self.current_time = network_start_time
        self.extract_stations(df)
        self.extract_edges_and_A_matrix(df)
        
        #initiate the Bi values for each station with the data from the edtes
        for station in self.stations:
            self.stations[station].initiate_Bis(self.edges)
        
        #initiate the D matrix and the D matrices
        self.extract_G_matrices()
        self.extract_D_matrix()
        self.extract_D_matrices()
        return
    
    # Extract the first time that exists in the data with a delay
    def extract_start_time(self, df):
        
        all_times = df['UtfAvgTid'].dropna()
        first_time_stamp = all_times.min()

        #get the rows with the first time stamp
        first_time_rows = df[df['UtfAvgTid'] == first_time_stamp]
        # If the delay is 0 or negative, find the next time minimum time stamp until a positive delay is found
        while first_time_rows['AvgFörsening'].sum() <=0 :
            #getting the minimum time stamp that is larger than the current first time stamp
            first_time_stamp = all_times[all_times > first_time_stamp].min()
            first_time_rows = df[df['UtfAvgTid'] == first_time_stamp]

        return pd.to_datetime(first_time_stamp)

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

        # Create a dictionary with all stations and their index in the network
        station_indices = {station: idx for idx, station in enumerate(self.stations)} # All stations in the network
        self.station_indices = station_indices
        return
    
    # Creates and add station to the network
    def add_station(self, name, id, df):
        station = Station(name, id)
        station.initiate_station(df, self.current_time)
        self.stations[name] = station
        self.N += 1
        return
    
    # Extract all edges from the data and add them to the network
    def extract_edges_and_A_matrix(self, df):
        unique_edges = df[['Avgångsplats', 'Ankomstplats']].drop_duplicates()
        
        # Convert to a list of lists 
        edges = unique_edges.values.tolist()
        
        # Create the undirected adjacency matrix of the network with the data from the edges
        adj_matrix = self.create_adjacency_matrix(edges) 
        self.A_matrix = adj_matrix

        # for every edge, add it to the network
        for i, edge in enumerate(edges):
            self.add_edge(i, edge[0], edge[1], df, adj_matrix.loc[edge[0]][edge[1]])
            
        return
    
    # Creates and add edge to the network
    def add_edge(self, id, start, end, df, adj_matrix):
        edge = Edge(id, start, end, adj_matrix)
        edge.initiate_edge(df)
        key = (start,end)
        self.edges[key] = edge
        return
    

    # Calculates the D matrix by going through the station and extract the delays
    def extract_D_matrix(self):
        D_matrix = np.zeros((self.N,1))
        
        for station_name, row_index in self.station_indices.items():
            station = self.stations[station_name]
            #station delay matrix is the delay at the station at the time
            D_matrix[row_index] = station.delay
            
        self.D_matrix = D_matrix
        return
    
    # Calculates the D matrices for each individual delay by going through the station and extract the delays
    def extract_D_matrices(self):
        D_matrices = []
        #loop through all stations
        for station_name, row_index in self.station_indices.items():
            station = self.stations[station_name]
            #if there is any delay at the station
            if len(station.delay_origins) > 0:
                #fetching the delay origins from the sataions and saving it in delays variable, each item in the list looks like this: ((start, end), delay)
                delays = station.delay_origins
                for delay_item in delays:
                    #create a D matrix for the current delay
                    current_thread_D_matrix = np.zeros((self.N,1))
                    delay = delay_item[1] #delay value is the second item in the tuple
                    current_thread_D_matrix[row_index] = delay #set the delay at the current station (row_index)
                    delay_edge = delay_item[0] #delay edge is the first item in the tuple, this is the edge that the delay came from

                    #create a directed A matrix and the G matrices for the current delay
                    directed_A_matrix, removed_edges = self.create_directed_A_matrix(delay_edge)
                    G_matrices = self.get_directed_G_matrices(directed_A_matrix, removed_edges)

                    D_matrices.append([current_thread_D_matrix, G_matrices]) #  will be in this format: [(D_matrix, [G_matrices]), (D_matrix, [G_matrices])...] 
                    
        self.D_matrices = D_matrices
        return

        # Creates the adjacency matrix of the network, that is 1 if the stations are connected with an edge, 0 otherwise
    def create_adjacency_matrix(self, edges):
        station_indices = self.station_indices
        n = self.N
        adj_matrix = np.zeros((n, n), dtype=float) # Creates an empty matrix to fill with 1 if the stations are connected

        # goes through edges and add 1 in the correct corresponding position in adj_matrix
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
    
    # Function that creates a directed A matrix from the original A matrix based on a directed delay. 
    # That is, it follows the direction the delay is spreading in. 
    # The functon follows this method to create the direted A matrix: 
        # Remove the outgoing edge that the delay was on 
        # Remove all incoming edges on the current station
        # Move to all the connected stations except the station where the delay came from, and do the same to them. 
        # step 1 remove the edge (start, end)
        # step 2 for each item in frontier: remove all 1:s in end column expcet for the start row 
        # add all outgoing edges from current item to frontier (add all 1:s in the row for the current item end station)
    def create_directed_A_matrix(self, directed_edge):
        removed_edges = []  #list to store the removed edges, this is a temporary list that the edges will be added to before they are added to the count dict
        removed_edges_with_count = {} #dict to store the removed edges with the count of how many edges that were removed from the end station
        directed_A_matrix = copy.deepcopy(self.A_matrix)

        # Initialize by removing the edge in the opposite direction of the delay
        directed_A_matrix = remove_initial_edge(directed_A_matrix, directed_edge, removed_edges)

        frontier = [directed_edge]
        visited_nodes = []

        while frontier: # while there are nodes to visit, bfs algorithm
            current_edge = frontier.pop(0)
            visited_nodes.append(current_edge[1])

            # Process the current edge and get the updated directed A matrix, send in the removed edges and the removed edges with count dict
            directed_A_matrix = process_current_edge(directed_A_matrix, current_edge, removed_edges, removed_edges_with_count)

            # Get new edges to be added to the frontier
            new_edges = get_new_edges_to_frontier(directed_A_matrix, current_edge, visited_nodes, frontier)
            frontier.extend(new_edges)

        return directed_A_matrix, removed_edges_with_count
    
    
    # Function that extracts the G matrices for each hour of the day
    # sets the G matrices in the network, does not return anything
    def extract_G_matrices(self):
        n = self.N
        G_matrices = create_24_hour_dict(matrix_element=True, n=n) # Creates a dictionary with 24 hours as keys and an empty nxn matrix as values
        
        #populating the G matrices
        for time_span in G_matrices.keys():
            G_matrices[time_span] = self.calculate_G_matrix(G_matrices[time_span], time_span, n)

        self.G_matrices = G_matrices
        return
    
    # Function that extracts the G matrices for each hour of the day with a directed delay
    # extra inputs compared to the normal G matrix is the A matrix and the removed edges
    # returns the G matrices
    def get_directed_G_matrices(self, A_matrix, removed_edges):
        n = self.N
        G_matrices = create_24_hour_dict(matrix_element=True, n=n)
        
        #populating the G matrices
        for time_span in G_matrices.keys():
            G_matrices[time_span] = self.calculate_G_matrix(G_matrices[time_span], time_span, n, A_matrix = A_matrix, removed_edges = removed_edges)
       
        return G_matrices
    
    # Function that calculates the G matrix for a specific time span
    # G_matrix should be an nxn empty matrix, time span should be a tuple of two times in Timestamp format only (0,1), (1,2) etc
    def calculate_G_matrix(self, G_matrix, time_span, n, A_matrix = None, removed_edges = None):
        if A_matrix is None: #if no A matrix is given, use the network undirected A matrix
            A_matrix = self.A_matrix
            
        # if there are removed edges, get the keys of the removed edges, otherwise set the removed edges keys to an empty list
        if removed_edges:
            removed_edges_keys = removed_edges.keys()
        else:
            removed_edges_keys = []
        
        #looping through all stations
        for station_name, row_index in self.station_indices.items():

            i_station = self.stations[station_name] #station object of the current row
            #going throgh all stations again for the values of the G matrix columns for the current row
            for col_index in range(n):
                value = calculate_G_matrix_values(time_span, A_matrix, removed_edges_keys, row_index, i_station, col_index, self.station_indices, self.stations, self.edges)
                G_matrix[row_index][col_index] = value
        return G_matrix

    # Evaluates the network against the real data in df.
    def evaluate_network(self, df, time_steps, visualize = False, directed_delay = True):
        print("Evaluating network")
        print("Time steps: ", time_steps)
        print("Network start time: ", self.current_time)
        if directed_delay:
            print("Using directed delay")
        else:
            print("Using undirected delay")
        print("-------------------------")

        # Taks as many time steps as the user sent in
        for step in range(time_steps):

            #fetching the true delay of the system from the data in df
            true_delay = self.get_D_matrix(df) #matrix that holds the true delay of the data in df
            true_delay = np.round(true_delay, 3)#round the delay matrix to 3 decimals
            
            #take a time step, either with directed delay or undirected delay
            if directed_delay:
                self.predict_time_step_with_direction() 
            else: 
                self.predict_time_step()

            #fetching the predicted delay of the system
            predicted_delay = self.D_matrix #predicted delay matrix
            predicted_delay = np.round(predicted_delay, 3) #round the delay matrix to 3 decimals

            # visualizing the prediction and actual data by creating maps of the network. These can  b
            if visualize:
                self.visualize_comparative_delays(true_delay, predicted_delay, step, title="Comparative Delay Visualization", cap=0.25)

            #creating a comparison array to hold the true, predicted and the difference between the two
            comparison = np.concatenate((true_delay, predicted_delay), axis=1)
            comparison = np.round(comparison, 3) #round the comparison matrix to 3 decimals
            #difference array for the difference between the true and predicted delay
            difference_array = np.zeros((self.N,1))
            #populating the difference array with the difference between the true and predicted delay for each station
            for i in range(self.N):
                difference_array[i] = true_delay[i] - predicted_delay[i]
            # add the difference array to the comparison array
            comparison = np.concatenate((comparison, difference_array), axis=1)
            #print the comparison delay matrix, default is to not print all stations, but can be changed to print_all=True to print all stations
            self.print_comparison_delay_matrix(comparison, print_all=False)
            print(" ")

        return
    
    # Calculates and returns the D matrix for the current time from the data in df
    def get_D_matrix(self, df):
        D_matrix = np.zeros((self.N,1)) # Creates an empty matrix to fill with the delay of the stations
        # Goes through all stations and add the delay to the D matrix
        for station_name, row_index in self.station_indices.items():
            station = self.stations[station_name]
            station_delay = station.calculate_delay(df, self.current_time)
            #station delay matrix is the delay at the station at the start time
            D_matrix[row_index] = station_delay
        return D_matrix


    # Function that executes an undirected time step 
    def predict_time_step(self):
        matrix_keys = list(self.G_matrices.keys()) # Get all the time intervals 
        current_time = self.current_time.time()
        
        G_matrix = find_current_time_G_matrix(matrix_keys, current_time, self.G_matrices)
            
        current_delay = self.D_matrix
    
        #matrix multiplication of G and D to get the difference between this time step and the next
        difference = np.matmul(G_matrix, current_delay)
       
        #adds the difference to the current delay
        self.D_matrix = current_delay + difference

        # Goes through all stations and updates the new delay
        for station_name, row_index in self.station_indices.items():
            station = self.stations[station_name]
            station.delay = self.D_matrix[row_index]
            
        #add one time step to the current time
        self.current_time += pd.DateOffset(minutes = self.time_step,)
        return

     # Function that executes a directed delay time step 
    def predict_time_step_with_direction(self):
        matrix_keys = list(self.G_matrices.keys()) # Get all the time intervals 
        current_time = self.current_time.time()
        
        #go through each individual delay matrix and update the delay
        for delay_thread in self.D_matrices: 
            current_delay = delay_thread[0]
            G_matrices = delay_thread[1]
        
            G_matrix = find_current_time_G_matrix(matrix_keys, current_time, G_matrices)
            #matrix multiplication of G and D to get the difference between this time step and the next
            difference = np.matmul(G_matrix, current_delay)
            delay_thread[0] = current_delay + difference #update the delay matrix in the delay thread

        # create a total delay matrix that will hold the sum of all individual delay matrices
        total_delay = np.zeros((self.N,1))
        for delay_thread in self.D_matrices: #sum all individual delay matrices
            total_delay += delay_thread[0]
            
        # Goes through all stations and updates the new delay
        for station_name, row_index in self.station_indices.items():
            station = self.stations[station_name]
            station.delay = total_delay[row_index]

        self.D_matrix = total_delay #update the delay matrix
        #add one time step to the current time
        self.current_time += pd.DateOffset(minutes = self.time_step_size,)
        return
    
    #Visualizes actual vs. predicted delays using network graphs.
    def visualize_comparative_delays(self, actual_delay, predicted_delay, step, title="Comparative Delay Visualization", cap=0.25):
        # Create two separate graphs: one for actual delays and one for predicted delays
        graphs = {"Actual Delay": actual_delay, "Predicted Delay": predicted_delay}
        
        for graph_title, delay_array in graphs.items():
            # Create a graph using the graphviz library
            G = pgv.AGraph(directed=True)
            # go through all stations and add nodes to the graph 
            for station_name, row_index in self.station_indices.items():
                delay = delay_array[row_index][0]
                color = get_color(delay, cap) #get the color of the node based on the delay value
                G.add_node(station_name, style="filled", fillcolor=color, fontsize=10, margin="0.1,0.1")

            # Add edges from the network into the graph
            for (start, end), _ in self.edges.items():
                G.add_edge(start, end, penwidth=2, color="gray")

            # Adjust layout for aesthetics, uncomment this line if you want to adjust the layout of the graph according to the graphviz documentation
            #G.graph_attr.update(rankdir="LR", nodesep="2.0", ranksep="1.5", splines="true", dpi="400")
            
            # Save the graph to a file
            output_path = f"images/{graph_title.lower().replace(' ', '_')}_step_{step + 1}.png"
            G.layout(prog="fdp") # can change this to change the layout of the graph
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

    # Function that prints the station information of a specific station
    def print_station_info(self, station_name):
        station = self.stations[station_name]
        print(f"Station: {station_name}")
        print(f"Neighbours in: {[neighbour.name for neighbour in station.N_in]}")
        print(f"Neighbours out: {[neighbour.name for neighbour in station.N_out]}")
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
            for station_name, row_index in self.station_indices.items(): 
                print(f"{station_name}: {delay_matrix[row_index][0]}")
        else: 
            for station_name, row_index in self.station_indices.items(): 
                if(abs(delay_matrix[row_index][0]) >= cap):
                    print(f"{station_name}: {delay_matrix[row_index][0].round(3)}")
        return
    
    # Print the comparison delay matrix. That is the matrix that includes the actual delay, the predicted delay and the difference between them. 
    def print_comparison_delay_matrix(self, comparison_delay_matrix, print_all = True,cap=0.25):
        print("Delay matrices at time: ", self.current_time)
        print("Station: True delay, Predicted delay, Difference")
        
        name_array = np.array(list(self.station_indices.keys())) #array of station names
        comparison_df = pd.DataFrame(comparison_delay_matrix, index = name_array, columns = ["True delay", "Predicted delay", "Difference"]) #put in a dataframe for easier visualization
        comparison_df= comparison_df.sort_index()        
        
        if print_all: 
           print(comparison_df)
        else: 
            #filtering out the rows that have 0 in both columns
            comparison_df = comparison_df[(comparison_df["True delay"] != 0) | (abs(comparison_df["Predicted delay"]) >= cap)]
            print(comparison_df)
        return

