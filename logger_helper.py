import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import os
from datetime import datetime
import json

def initialize_graph_folder(params):
  
  # Get the current date and time as a string
  current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

  # Create the folder name with "graphs_" prefix
  folder_name = f"graphs_{current_datetime}"

  # Specify the path where you want to create the folder
  base_directory = ""  # Replace with your desired directory path

  # Combine the base directory and folder name to create the full path
  folder_path = os.path.join(base_directory, folder_name)

  # Check if the folder already exists before creating it
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_name}' created at '{folder_path}'")
  else:
    print(f"Folder '{folder_name}' already exists at '{folder_path}'")
    
  params_file_path = os.path.join(folder_path, "params.txt")

  # Write the dictionary to a JSON file
  def convert_arrays(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return obj

  with open(params_file_path, 'w') as json_file:
    json.dump(params, json_file, default=convert_arrays, indent=4)


  print(f"Dictionary saved to '{params_file_path}'")
    
  return folder_name

def save_sim_data(folder_name, sim_data):
  # Get path to new file
  folder_path = os.path.join(folder_name, "sim_data.json")

  # Convert NumPy arrays to lists for JSON serialization
  for key, value in sim_data.items():
    sim_data[key] = value.tolist()
  
  # Write data in file
  with open(folder_path, 'w') as json_file:
    json.dump(sim_data, json_file)
    json_file.write('\n')


def load_sim_data(folder_name):
  # Get path to simulation data file
  folder_path = os.path.join(folder_name, "sim_data.json")
    
  # Initialize data dictionary
  sim_data = {}
   
  # Load data from file
  with open(folder_path, 'r') as json_file:
    loaded_data = json.load(json_file)
  
  # Convert lists back to NumPy arrays
  for key, value in loaded_data.items():
    sim_data[key] = np.array(value)
    
  return sim_data






