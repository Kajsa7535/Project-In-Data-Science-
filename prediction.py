
import pandas as pd
import os
import glob
import argparse
from network.model import Network

def remove_old_images():
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

def create_img_folder():
    if not os.path.exists("images"):
        os.makedirs("images")

def print_initial_delay_matrix(network):
    print("initial delay matrix")
    network.print_delay_matrix(print_all=False)
    print(" --------------------------------------------")
    print(" ")

def extract_random_start_time(df, cap = 7):
    delay_rows = df[df['AvgFörsening'] > cap]
    random_start_time = delay_rows.sample(1)['UtfAvgTid'].values[0]
    return random_start_time


def main(network_name, network_start_time = None, time_steps=10, time_step_size = 1, visualize = True, directed_delay = True):
    #creating image folder if it does not exist
    create_img_folder()
    #removing old images if there are any
    remove_old_images()

    #reading the network data
    data_folder_name = "data"
    network_file_name = f"{network_name}.csv"
    network_path = os.path.join(data_folder_name, network_file_name)
    print("network path", network_path)

    #creating a network
    df_network = pd.read_csv(network_path, sep=',', encoding='utf-8')
    network = Network()

    #if there is a start time, convert it to datetime
    if network_start_time:
        network_start_time_string = f"{network_start_time}:00"
        network_start_time = pd.to_datetime(network_start_time_string)

    #initiating the network and printing the initial delay matrix
    network.initiate_network(df_network, time_step_size, network_start_time=network_start_time)
    print_initial_delay_matrix(network)

    #evaluating the network
    print("directed delay", directed_delay)
    network.evaluate_network(df_network, time_steps, visualize=visualize, directed_delay=directed_delay)

    return
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the network simulation")

    #Positional arguments
    parser.add_argument("network_name", help="Name of the network CSV file")

    #Optional arguments
    parser.add_argument("--network_start_time", default=None, help="Start time of the network (format: YYYY-MM-DD HH:MM)")
    parser.add_argument("--time_steps", type=int, default=10, help="Number of time steps to evaluate the network")
    parser.add_argument("--time_step_size", type=int, default=1, help="Size of each time step")
    parser.add_argument("--visualize", action="store_true", default=False,  help="Visualize the network during evaluation")
    parser.add_argument("--directed_delay", action="store_true", default=False, help="Consider directed delay in the evaluation")

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        network_name=args.network_name,
        network_start_time=args.network_start_time,
        time_steps=args.time_steps,
        time_step_size=args.time_step_size,
        visualize=args.visualize,
        directed_delay=args.directed_delay
    )
    
