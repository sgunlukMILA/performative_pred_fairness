#!/usr/bin/env python
# coding: utf-8


# IMPORT LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
get_ipython().system(' pip install lime')
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
import simpy as sp
import math


# # Data Generating Process


# DATA GENERATING CLASS

class ScalarLinearDecisionModel():
  """
    A class for generating simulated data based on our scalar linear decision model.

    Attributes:
        n_samples (int): Number of samples to generate.
        s_a_const (float): Coefficient for S in generating A (alpha).
        a_var (float): Variance for A noise (epsilon_A^2).
        a_noise (numpy.ndarray): Noise added to A based on a_var (u_A).
        s_c_const (float): Coefficient for S in generating C (omega_S).
        a_c_const (float): Coefficient for A in generating C (omega_A).
        c_var (float): Variance for C noise (epsilon_C^2).
        c_noise (numpy.ndarray): Noise added to C based on c_var (u_C).
        s_y_const (float): Coefficient for S in generating Y (m_S).
        c_y_const (float): Coefficient for C in generating Y (m_C).
        y_var (float): Variance for Y noise (epsilon_Y^2).
        y_noise (numpy.ndarray): Noise added to Y based on y_var (u_Y).
        S (numpy.ndarray): Binary variable indicating the treatment {0, 1}.
        S_sym (numpy.ndarray): Binary variable converted to {-1, 1}.
        A (numpy.ndarray): Simulated values of variable A.
        C (numpy.ndarray): Simulated values of variable C.
        Y (numpy.ndarray): Simulated values of binary outcome Y.

    Methods:
        __init__(params): Initialize the ScalarLinearDecisionModel using params, a dictionary containing model parameters.
        generate_basic_data(): Generates simulated data based on the model.
        generate_do_A_data(new_A): Generates data for improved set of ancestral featues.
        generate_improve_A_data(diff_A): Generates data by improving A by diff_A.
        generate_do_C_data(new_C): Generates data for improved set of causal featues.
        generate_improve_C_data(diff_C): Generates data by improving C by diff_C.
        ts_to_df(describe=False): Converts simulated data to a DataFrame with describe being a printer attribute.
  """
  def __init__(self, params):
    #n_samples, s_a_const = 1, a_var = 1, s_c_const = 0,  a_c_const = 1, c_var = 1, s_y_const = 1, c_y_const = 1, y_var = 0.1):
    self.n_samples = params['n_samples']
    self.s_a_const = params['s_a_const']
    self.a_var = params['a_var']
    self.a_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.a_var*np.ones(self.n_samples))
    self.s_c_const = params['s_c_const']
    self.a_c_const = params['a_c_const']
    self.c_var = params['c_var']
    self.c_noise = np.random.normal(loc = np.zeros(self.n_samples), scale=self.c_var*np.ones(self.n_samples))
    self.s_y_const = params['s_y_const']
    self.c_y_const = params['c_y_const']
    self.y_var = params['y_var']
    self.y_noise = np.random.normal(loc = np.zeros(self.n_samples), scale = self.y_var*np.ones(self.n_samples))
    self.S = np.random.binomial(n=1, p = 0.75, size=self.n_samples)
    #make labels s = {-1, 1} by doing (2S-1)
    self.S_sym = 2*self.S - 1
    self.A = np.empty((self.n_samples,1))
    self.C = np.empty((self.n_samples,1))
    self.Y = np.empty((self.n_samples,1))
  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))
  def generate_basic_data(self):
    self.A = self.S_sym*self.s_a_const + self.a_noise
    self.C = self.A*self.a_c_const + self.S_sym*self.s_c_const + self.c_noise
    y_logit = self.sigmoid(self.C*self.c_y_const + self.S_sym*self.s_y_const + self.y_noise)
    self.Y = np.random.binomial(n=1, p = y_logit)
  def generate_do_A_data(self, new_A):
    temp_C = new_A*self.a_c_const + self.S_sym*self.s_c_const + self.c_noise
    y_logit = self.sigmoid(self.C*self.c_y_const + self.S_sym*self.s_y_const + self.y_noise)
    temp_Y = np.random.binomial(n=1, p = y_logit)
    return temp_C, temp_Y
  def generate_improve_A_data(self, diff_A):
    temp_C = (self.A+diff_A)*self.a_c_const + self.S_sym*self.s_c_const + self.c_noise
    y_logit = self.sigmoid(self.C*self.c_y_const + self.S_sym*self.s_y_const + self.y_noise)
    temp_Y = np.random.binomial(n=1, p = y_logit)
    return temp_C, temp_Y
  def generate_do_C_data(self, new_C):
    y_logit = self.sigmoid(new_C*self.c_y_const + self.S_sym*self.s_y_const + self.y_noise)
    temp_Y = np.random.binomial(n=1, p = y_logit)
    return temp_Y
  def generate_improve_C_data(self, diff_C):
    y_logit = self.sigmoid((self.C+diff_C)*self.c_y_const + self.S_sym*self.s_y_const + self.y_noise)
    temp_Y = np.random.binomial(n=1, p = y_logit)
    return temp_Y
  def ts_to_df(self, describe = False):
    sim_data = pd.DataFrame()
    sim_data["S"] = self.S
    sim_data["A"] = self.A
    sim_data["C"] = self.C
    sim_data["Y"] = self.Y
    if describe:
      print(sim_data.describe())
      print(sim_data.loc[sim_data['S'] == 0].describe())
      print(sim_data.loc[sim_data['S'] == 1].describe())
    return sim_data



# Data generation test call

default_params = {
    'n_samples': 1000, 
    's_a_const': 1, 
    'a_var': 1, 
    's_c_const': 0,  
    'a_c_const': 1, 
    'c_var': 1, 
    's_y_const': 1, 
    'c_y_const': 1, 
    'y_var': 0.1 
}

sample_dm = ScalarLinearDecisionModel(default_params)
sample_dm.generate_basic_data()
sample_data = sample_dm.ts_to_df()

printer = True

if printer:
  print(sample_data.describe())
  print(sample_data.loc[sample_data['S'] == 0].describe())
  print(sample_data.loc[sample_data['S'] == 1].describe())


# # Criteria: $\hat{Y}(c) - \hat{Y} \perp S$



#Helper Functions
def calculate_joint_distribution(data):
  """
    Calculate the joint distribution of sensitive feature, S, and label/prediction, y, based on given data.

    Args:
        data (pandas.DataFrame): Input data containing S and y values.

    Returns:
        pandas.Series: Calculated joint distribution.
  """
  joint_distribution = data.groupby(['S', 'y']).size() / len(data)
  return joint_distribution

def calculate_expected_joint_distribution(marginal_probabilities_S, marginal_probabilities_y):
  """
    Calculate the expected joint distribution, assuming independence.

    Args:
        marginal_probabilities_S (pandas.Series): Marginal probabilities of sensitive attribute, S.
        marginal_probabilities_y (pandas.Series): Marginal probabilities of label/prediction, y.

    Returns:
        pandas.DataFrame: Calculated expected joint distribution.
  """
  expected_joint_distribution = pd.DataFrame(index=marginal_probabilities_S.index, columns=marginal_probabilities_y.index)
  expected_joint_distribution = expected_joint_distribution.fillna(0)

  for s, p_s in marginal_probabilities_S.items():
    for y, p_y in marginal_probabilities_y.items():
      expected_joint_distribution.loc[s, y] = p_s * p_y

  return expected_joint_distribution

def printer_joint_marg_diff(temp_data):
  """
    Calculate and print joint distribution, expected joint distribution, and their difference.

    Args:
        temp_data (pandas.DataFrame): Input data that includes sensative feature, S, and label/prediction, y.

    Prints:
        Joint distribution, expected joint distribution (assuming independence), and their differences for 
        each combination of 'S' and 'y'.
  """
  marginal_probabilities_S = temp_data['S'].value_counts() / len(temp_data)
  marginal_probabilities_diff = temp_data['y'].value_counts() / len(temp_data)

  joint_distribution = calculate_joint_distribution(temp_data)

  expected_joint_distribution = calculate_expected_joint_distribution(marginal_probabilities_S, marginal_probabilities_diff)

  for (s, y), joint_prob in joint_distribution.items():
      expected_prob = expected_joint_distribution.loc[s, y]
      print(f"S: {s}, y: {y}")
      print(round(joint_prob, 4), ": Joint Distribution")
      print(round(expected_prob, 4), ": Expected Joint Distribution (Assuming Independence)")
      print(round(joint_prob - expected_prob, 4), ": diff")
      print()


#P(Y'(c) - Y' & S)

def criteria_measure(dm, trainer, improvement):
  data_A = dm.A.reshape((dm.n_samples,1))
  data_C = dm.C.reshape((dm.n_samples,1))
  data_AC = np.concatenate((data_A, data_C), axis=1)
  data_Y = dm.Y
  data_improve_C = data_C + improvement
  data_improve_AC = np.concatenate((data_A, data_improve_C), axis=1)
  data_improve_Y = dm.generate_improve_C_data(improvement)
  model = trainer.fit(data_AC, data_Y)
  y_improve_pred = model.predict(data_improve_AC)
  #print(data_improve_AC.shape)
  #print(data_improve_Y)
  print("score: ", model.score(data_improve_AC, data_improve_Y))

  temp_data = pd.DataFrame()
  temp_data['y'] = model.predict(data_improve_AC) - model.predict(data_AC)
  temp_data['S'] = dm.S

  printer_joint_marg_diff(temp_data)

criteria_measure(sample_dm, LogisticRegression(), 1)


#P(Y'(c) - Y' & S | Y = 0)

def alt_criteria(data, trainer, improvement):
  data_A = data.A.reshape((data.n_samples,1))
  data_C = data.C.reshape((data.n_samples,1))
  data_AC = np.concatenate((data_A, data_C), axis=1)
  data_Y = data.Y
  data_improve_C = data_C + improvement
  data_improve_AC = np.concatenate((data_A, data_improve_C), axis=1)
  data_improve_Y = data.generate_improve_C_data(improvement)
  model = trainer.fit(data_AC, data_Y)

  y_improve_pred = model.predict(data_improve_AC)
  print("score: ", model.score(data_improve_AC, data_improve_Y))

  temp_data = pd.DataFrame()
  indices = np.where(data_Y == 0)[0]

  temp_data['y'] = model.predict(data_improve_AC[indices]) - model.predict(data_AC[indices])
  temp_data['S'] = data.S[indices]

  printer_joint_marg_diff(temp_data)

alt_criteria(sample_dm, LogisticRegression(), 0.5)


# # Equal Improvability


# Helper Function
def ei_calc(y_data, S_data, printer=False):
  """
    Calculate our Equal Improvability Measure (EI) for based on observed data.

    Args:
        y_data (numpy.ndarray or list): Data of binary labels/predictions, y.
        S_data (numpy.ndarray or list): Data of sensative attribute, S.
        printer (bool, optional): If True, prints the joint distribution, expected distribution assuming independence, 
                                  and the difference (default is False).

    Returns:
        tuple of two floats or None: EI for S=0, EI for S=1.
  """
  temp_data = pd.DataFrame()
  temp_data['y'] = y_data
  temp_data['S'] = S_data
  # Calculate marginal probabilities of 'S' and 'y'
  marginal_probabilities_S = temp_data['S'].value_counts() / len(temp_data)
  marginal_probabilities_diff = temp_data['y'].value_counts() / len(temp_data)

  # Calculate the joint distribution
  joint_distribution = calculate_joint_distribution(temp_data)

  # Calculate the expected joint distribution assuming independence
  expected_joint_distribution = calculate_expected_joint_distribution(marginal_probabilities_S, marginal_probabilities_diff)

  jp_s0_d1 = 0
  jp_s1_d1 = 0
  # Print the joint distribution and expected joint distribution together
  for (s, y), joint_prob in joint_distribution.items():
    if printer:
        expected_prob = expected_joint_distribution.loc[s, diff]
        print(f"S: {s}, y: {y}")
        print(round(joint_prob, 4), ": Joint Distribution")
        print(round(expected_prob, 4), ": Expected Joint Distribution (Assuming Independence)")
        print(round(joint_prob - expected_prob, 4), ": diff")
        print()
    if diff == 1:
      if s == 0:
        jp_s0_d1 = joint_prob
      else:
        jp_s1_d1 = joint_prob
  #print("MARG ", marginal_probabilities_S)
  try:
    p_improve_s0 = jp_s0_d1/marginal_probabilities_S[0]
    p_improve_s1 = jp_s1_d1/marginal_probabilities_S[1]
  except:
    p_improve_s0 = None
    p_improve_s1 = None
  return p_improve_s0, p_improve_s1




# ## GRAPH HELPERS


#grapher helper functions
def line_plots(ei_data, consts, deltas, arrow_str = ''):
  """
    Generate and display line plots for Equal Improvability (EI) data.

    Args:
        ei_data (numpy.ndarray): EI data for varying cofficients and deltas.
        consts (list): List of values of the varied coefficient with which EI data was collected.
        deltas (list): List of delta values with which EI data was collected.
        arrow_str (str, optional): String for the legent to indicate the varied coefficient (default is '').
  """
  names = sorted(
            mcolors.TABLEAU_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
  names = [elem.removeprefix('tab:') for elem in names]
  for i, const in enumerate(consts):
    plt.plot(deltas, ei_data[i,:,0], color=names[i], linestyle='dashed',
             label = "protected for const = "+ str(round(const, 4)))
    plt.plot(deltas, ei_data[i,:,1], color=names[i],
             label = "not protected for const = "+ str(round(const, 4)))
  plt.xlabel('effort ($\Delta$)')
  plt.ylabel('probability of improvability')
  plt.title('Plot of P(Y(c+$\Delta$) = 1| Y(c) = 0, S = s) vs $\Delta$ effort')
  plt.legend()
  plt.show()

  for i, const in enumerate(consts):
    plt.plot(deltas, ei_data[i,:,1] - ei_data[i,:,0], color=names[i],
             label = arrow_str + " coeff = "+ str(round(const, 4)))
  plt.xlabel('effort ($\Delta$)')
  plt.ylabel('difference in probability of improving')
  plt.title('Plot of P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 1) - P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 0) vs $\Delta$ effort')
  plt.legend()
  plt.show()


def all_improve_heat_map_effort(results, mask, var_I, str_var_I, var_J, str_var_J, effort_str="None", type_sim="NA", delta=0, save=False):
  """
    Generate and display a heatmap for improvement probability data across different variables.

    Args:
        results (numpy.ndarray): EI data while varying two coefficients.
        mask (numpy.ndarray): Mask to indicate missing data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        effort_str (str, optional): Description of effort regime based on delta (default is 'None').
        type_sim (str, optional): Type of experiment for the filename (default is 'NA').
        delta (float, optional): Effort change delta (default is 0).
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  mask = np.flip(mask, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  map = sns.heatmap(results, cmap='viridis', mask=mask, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
  #, cbar_kws={'label': 'P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 0) - P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 1)'})

  # Set axis labels and plot title
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(effort_str + "-Effort regime ($\Delta$ = " + str(delta) + ")")
  plt.margins(x=0)

  # Customize the tick labels on the x and y axes
  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  # Show the heatmap
  if save:
    plt.savefig("graphs/"+type_sim+effort_str+"effort")
  plt.show()


#Feature Importance Heatmaps
    
def importance_heat_map(results, var_I, str_var_I, var_J, str_var_J, title_importance="None", type_importance="NA", save=False):
  """
    Generate and display a heatmap for the importance of a single feature.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  map = sns.heatmap(results, cmap='viridis')#, mask=mask)#, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
    
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(title_importance+ " Importance")

  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
      plt.savefig("graphs/"+type_importance+"_importance")
  plt.show()

def diff_importance_heatmap(results, var_I, str_var_I, var_J, str_var_J, title_importance="None", type_importance="NA", save=False):
  """
    Generate and display a heatmap for the difference in importance of features A and C.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  limit = max(np.max(results), np.min(results)*-1)
  palette = (sns.color_palette("Spectral_r", as_cmap=True))
  map = sns.heatmap(results, cmap=palette, vmin=-1*limit, vmax=limit)#, mask=mask)#, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
  
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(title_importance+ " Importance")

  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
      plt.savefig("graphs/"+type_importance+"_importance")
  plt.show()

def categorical_heatmap(results, var_I, str_var_I, var_J,  str_var_J, type_importance="None", title_importance="NA", save=False):
  """
    Generates and displays a categorical heatmap for feature importance, i.e. when coefficient of A is greater than
    coefficient of C and vice versa.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  results = np.where(results < 0, -1, 1)
  var_I = np.flip(var_I, axis = 0)
  category_cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
    
  sns.heatmap(results, fmt='', cmap=sns.color_palette("Spectral_r", as_cmap=True), cbar_kws={"ticks": [-1, 0, 1]},
               vmin=-1.25, vmax=1.25)

  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(type_importance)
  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
    plt.savefig("graphs/"+type_importance+"_pos_importance")
  plt.show()


testing = False
if testing: 
  A = np.array([1, 2, 3])
  B =  np.array([1, 2, 3, 4])
  results = np.empty((A.shape[0],B.shape[0])) #[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  for i, a in enumerate(A):
    for j, b in enumerate(B):
      results[i,j] = a*-0.4 + b*-0.04

  print(results)
  masker = np.zeros_like(results, dtype=bool)
  all_improve_heat_map_effort(results, masker, A, 'I', B, 'J')


def ei_violation_all_improve(trainer, trials, params, deltas, str_var_1, vars_1, str_var_2, vars_2):
  """
    Calculate various metrics related to Equal Improvability (EI), varying two parameters in order to make heatmaps.

    Args:
        trainer: The trained model used for prediction.
        trials (int): Number of trials to simulate to collect data.
        params (dict): Parameters for the data generation model.
        deltas (numpy.ndarray): Array of different effort change, delta.
        str_var_1 (str): String representing the variable name for the first coefficient that is varied.
        vars_1 (numpy.ndarray): Array of variable values to be used to vary the first coefficient.
        str_var_2 (str): String representing the variable name for the second coefficient that is varied.
        vars_2 (numpy.ndarray): Array of variable values to be used to vary the second coefficient.

    Returns:
        dict: A dictionary containing the following calculated metrics:
            - ei_data: Array of EI data for different parameters.
            - mask: Mask indicating missing data.
            - best_improve: Array of 1 where C was better to improve -1 where A was better improve, and 0 if equal.
            - pred_accuracy: Array of prediction accuracy for each set of parameters.
            - feature_weights_AC: Array of feature weights for each set of parameters.
  """
  print(trials)
  ei_data = np.zeros((3, deltas.shape[0], vars_1.shape[0], vars_2.shape[0]))
  mask = np.zeros((3, deltas.shape[0], vars_1.shape[0], vars_2.shape[0]), dtype=bool)
  best_improve = np.zeros((deltas.shape[0], vars_1.shape[0], vars_2.shape[0]))
  pred_accuracy = np.zeros((vars_1.shape[0], vars_2.shape[0]))
  feature_weights = np.zeros((2, vars_1.shape[0], vars_2.shape[0]))
  for i, var_1 in enumerate(vars_1):
    for j, var_2 in enumerate(vars_2):
      A_failed_trials = np.zeros(deltas.shape[0])
      C_failed_trials = np.zeros(deltas.shape[0])
      AC_failed_trials = np.zeros(deltas.shape[0])
      for trial in range(trials):
        #DATA
        params[str_var_1] = var_1
        params[str_var_2] = var_2
        #print("var 1: ", var_1, " and var 2: ", var_2)
        #print(params)
        data = ScalarLinearDecisionModel(params)
        data.generate_basic_data()
        data_A = data.A.reshape((data.n_samples,1))
        data_C = data.C.reshape((data.n_samples,1))
        data_S = data.S
        data_AC = np.concatenate((data_A, data_C), axis=1)
        data_Y = data.Y.reshape(data.n_samples)
        #print(np.mean(data_Y))
        model = trainer.fit(data_AC, data_Y)
        pred_y = model.predict(data_AC)
        pred_accuracy[i, j] += model.score(data_AC, data_Y)/trials
        feature_weights[:, i, j] += (model.coef_[0])/trials
        for d, delta in enumerate(deltas):
            data_improve_A_A = data_A[pred_y == 0] + delta
            data_improve_A_C, data_improve_A_Y  = data.generate_improve_A_data(delta)
            data_improve_A_C = data_improve_A_C.reshape((data.n_samples,1))
            data_improve_A_AC = np.concatenate((data_improve_A_A, data_improve_A_C[pred_y == 0]), axis=1)
            data_improve_C_AC = np.concatenate((data_A[pred_y == 0], data_C[pred_y == 0] + delta), axis=1)
            try:
              temp_A_y = model.predict(data_improve_A_AC)
              A_temp_0, A_temp_1 = ei_calc(temp_A_y, data_S[pred_y == 0])
              if (A_temp_0 == None or A_temp_1 == None):
                A_failed_trials[d] += 1
              else:
                ei_data[0, d, i, j] += (A_temp_1-A_temp_0)/trials
            except:
              A_failed_trials[d] += 1
        
            #DATA FOR JUST CHANGING C
            try:
              temp_C_y = model.predict(data_improve_C_AC)
              C_temp_0, C_temp_1 = ei_calc(temp_C_y, data_S[pred_y == 0])
            #print("Group 0: ", C_temp_0, " and Group 1: ", C_temp_1)
              if (C_temp_0 == None or C_temp_1 == None):
                C_failed_trials[d] += 1
              else:
                ei_data[1, d, i, j] += (C_temp_1-C_temp_0)/trials
            except:
              C_failed_trials[d] += 1
        
            #DATA FOR CHANGING A OR C
            try:
              temp_y = np.max((temp_A_y, temp_C_y), axis=0)
              AC_temp_0, AC_temp_1 = ei_calc(temp_y, data_S[pred_y == 0])
              if (AC_temp_0 == None or AC_temp_1 == None):
                AC_failed_trials[d] += 1
              else:
                ei_data[2, d, i, j] += (AC_temp_1-AC_temp_0)/trials
              best_improve[d, i, j] += np.mean(np.where(temp_C_y > temp_A_y, 1, np.where(temp_C_y < temp_A_y, -1, 0)))/trials
            except:
              AC_failed_trials[d] += 1

      for d in range(deltas.shape[0]): 
        if A_failed_trials[d] >= trials*3/4:
          mask[0, d, i, j] = True
        elif A_failed_trials[d] > 0:
          ei_data[0, d, i, j] = ei_data[0, d, i, j]*trials/(trials-A_failed_trials[d])
        if C_failed_trials[d] >= trials*3/4:
          mask[1, d, i, j] = True
        elif C_failed_trials[d] > 0:
          ei_data[1, d, i, j] = ei_data[1, d, i, j]*trials/(trials-C_failed_trials[d])
        if AC_failed_trials[d] >= trials*3/4:
          mask[2, d, i, j] = True
        elif AC_failed_trials[d] > 0:
          ei_data[2, d, i, j] = ei_data[2, d, i, j]*trials/(trials-AC_failed_trials[d])
  results = {
      "ei_data": ei_data, 
      "mask": mask, 
      "best_improve": best_improve, 
      "pred_accuracy": pred_accuracy, 
      "feature_weights_AC": feature_weights
  }
  return results


def find_best_improve_lin_cost(cost_fn, pred_fn, data):
  """
    Find the best improvement for each individual in the dataset considering cost constraints.

    Args:
        cost_fn (dict): Cost function with weights and bias.
        pred_fn (dict): Prediction function with weights and bias.
        data (dict): Dictionary containing data of sensitive features, S, and observed features, A and C,
        that the predictor observed.

    Returns:
        numpy.ndarray: Array of the necessary cost improvements for each individual.
  """
  features = data["AC"]
  n, d = features.shape
  best_improvement = np.zeros_like(features)
  delta_necessary = np.zeros(n)
  for individual in range(n):
    try:
      model = gp.Model("best_improvement")
      model.Params.LogToConsole = 0
      improve_vector = model.addMVar(shape=(1,d), name="added_vector")
      model.setObjective(improve_vector@cost_fn['w'] + cost_fn['b'], GRB.MINIMIZE)
      # Add the constraint for less than delta cost
      model.addConstr((features[individual]+improve_vector)@pred_fn['w'] + pred_fn['b'] >= 0)
      #model.addConstr( <= delta)
       # Optimize the model
      model.optimize()
      # Get the optimal solution
      if model.status == GRB.OPTIMAL:
        best_improvement[individual] = improve_vector.X
        #print(model.getObjective().getValue())
        delta_necessary[individual] = model.getObjective().getValue()

    except gp.GurobiError as e:
      print("Error code " + str(e.errno) + ": " + str(e))
    #print("solved!")

  return delta_necessary

def ei_violation(trainer, trials, params, deltas, cost_fn, str_var_1, vars_1, str_var_2, vars_2):
  """
    Calculate various metrics related to Equal Improvability (EI) measure under cost constraints.

    Args:
        trainer: The trained model used for prediction.
        trials (int): Number of trials to simulate to collect data.
        params (dict): Parameters for the data generation model.
        deltas (numpy.ndarray): Array of different effort change, delta.
        cost_fn (dict): Cost function with weights and bias.
        str_var_1 (str): String representing the variable name for the first coefficient that is varied.
        vars_1 (numpy.ndarray): Array of variable values to be used to vary the first coefficient.
        str_var_2 (str): String representing the variable name for the second coefficient that is varied.
        vars_2 (numpy.ndarray): Array of variable values to be used to vary the second coefficient.

    Returns:
        dict: A dictionary containing the following calculated metrics:
            - ei_data: Array of EI data for different parameters.
            - mask: Mask indicating missing data.
            - best_improve: Array of 1 where C was better to improve -1 where A was better improve, and 0 if equal.
            - pred_accuracy: Array of prediction accuracy for each set of parameters.
            - feature_weights_AC: Array of feature weights for each set of parameters.
  """
  ei_data = np.zeros((deltas.shape[0], vars_1.shape[0], vars_2.shape[0]))
  mask = np.zeros((deltas.shape[0], vars_1.shape[0], vars_2.shape[0]), dtype=bool)
  best_improve = np.zeros((deltas.shape[0], 2, vars_1.shape[0], vars_2.shape[0]))
  pred_accuracy = np.zeros((vars_1.shape[0], vars_2.shape[0]))
  feature_weights = np.zeros((2, vars_1.shape[0], vars_2.shape[0]))
  for i, var_1 in enumerate(vars_1):
    for j, var_2 in enumerate(vars_2):
      failed_trials = np.zeros(deltas.shape[0])
      for trial in range(trials):
        #DATA
        #print(trial)
        params[str_var_1] = var_1
        params[str_var_2] = var_2
        #print("var 1: ", var_1, " and var 2: ", var_2)
        #print(params)
        data = ScalarLinearDecisionModel(params)
        data.generate_basic_data()
        data_A = data.A.reshape((data.n_samples,1))
        data_C = data.C.reshape((data.n_samples,1))
        data_S = data.S
        data_AC = np.concatenate((data_A, data_C), axis=1)
        data_Y = data.Y.reshape(data.n_samples)
        #print(np.mean(data_Y))
        model = trainer.fit(data_AC, data_Y)
        pred_y = model.predict(data_AC)
        pred_accuracy[i, j] += model.score(data_AC, data_Y)/trials
        feature_weights[:, i, j] += (model.coef_[0])/trials
        pred_fn = {'w': model.coef_.flatten(), 'b': model.intercept_[0]} 
        data_unqualified = {
            "S": data_S[pred_y == 0],
            "AC": np.concatenate((data_A[pred_y == 0], data_C[pred_y == 0]), axis = 1)
        }

        '''def pred_fn1(features, added_vector):
          improved_features = (features+added_vector)
          lin_combo = np.dot(improved_features, coefficients) + intercept
          return (lin_combo >= 0).astype(int)
        # my attempt at keeping some dependenance, so changing A also improves C, 
        # but not sure how appropriate this is
        def pred_fn2(features, added_vector):
          A_change, C_change = added_vector
          improved_features = data.generate_improve_A_data(A_change)
          improved_features[1] += C_change 
          improved_features = improved_features.reshape(1, -1)
          lin_combo = np.dot(improved_features, coefficients) + intercept
          logits = 1 / (1 + np.exp(-lin_combo))
          return (probabilities >= 0.5).astype(int)'''
        best_delta_improve = find_best_improve_lin_cost(cost_fn, pred_fn, data_unqualified)
        for d, delta in enumerate(deltas):
          temp_y =  np.where(best_delta_improve <= delta, 1, 0)
          #print("delta: ", delta)
          #print("mean: ", temp_y.mean())
          temp_0, temp_1 = ei_calc(temp_y, data_S[pred_y == 0])
          #print("temp_0: ", temp_0, " temp_1: ", temp_1)
          if (temp_0 == None or temp_1 == None):
            failed_trials[d] += 1
          else:
            ei_data[d, i, j] += (temp_1-temp_0)/trials
          #except:
           # failed_trials[d] += 1

      for d in range(deltas.shape[0]): 
        if failed_trials[d] >= trials*3/4:
          mask[d, i, j] = True
        elif failed_trials[d] > 0:
          ei_data[d, i, j] = ei_data[d, i, j]*trials/(trials-failed_trials[d])
  results = {
      "ei_data": ei_data, 
      "mask": mask, 
      "best_improve": best_improve, 
      "pred_accuracy": pred_accuracy, 
      "feature_weights_AC": feature_weights
  }
  return results



# # CURRENT RUNS

# ##  Experiment C: A->C vs. S->C


simulation_C = {
    "trials": 10,
    "consts_ac": np.arange(0, 10, 0.25),
    "consts_sc": np.arange(0, 10, 0.25),
    "deltas": np.array([0.5, 1, 2.5, 5]),
    "type_effort": ["Low", "Mid", "High", "Higher"],
    "type_sims": ["C_A_", "C_C_", "C_AC_"],
    "imprtnc_type_sims": ["C_A", "C_C", "C_Difference_between_A_and_C", "C_Improve_A_or_C"],
    "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}


ei_data_C = ei_violation_all_improve(LogisticRegression(), trials = simulation_C["trials"],
                                     params = default_params.copy(),
                                     deltas = simulation_C["deltas"], 
                                     str_var_1 = "a_c_const",
                                     vars_1 = simulation_C["consts_ac"], 
                                     str_var_2 = "s_c_const",
                                     vars_2 = simulation_C["consts_sc"])



#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_C["deltas"]):
    if simulation_C["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_C["ei_data"][i, d], ei_data_C["mask"][i, d], simulation_C["consts_ac"], r"$\omega_a$ (A$\rightarrow$C)", simulation_C["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", simulation_C["type_effort"][d], simulation_C["type_sims"][i], delta)
  



#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_C["feature_weights_AC"][0] - ei_data_C["feature_weights_AC"][1], 
                        simulation_C["consts_ac"], r"$\omega_a$ (A$\rightarrow$C)", 
                        simulation_C["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", 
                        type_importance=simulation_C["imprtnc_type_sims"][i], 
                        title_importance=simulation_C["imprtnc_title_sims"][i])
  elif i == 3:
    print(np.max(ei_data_C["feature_weights_AC"][0] - ei_data_C["feature_weights_AC"][1]))
    print(np.min(ei_data_C["feature_weights_AC"][0] - ei_data_C["feature_weights_AC"][1]))
    categorical_heatmap(ei_data_C["feature_weights_AC"][0] - ei_data_C["feature_weights_AC"][1], 
                        simulation_C["consts_ac"], r"$\omega_a$ (A$\rightarrow$C)", 
                        simulation_C["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", 
                        type_importance=simulation_C["imprtnc_type_sims"][i], 
                        title_importance=simulation_C["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_C["feature_weights_AC"][i], 
                        params_C["consts_ac"], r"$\omega_a$ (A$\rightarrow$C)", 
                        params_C["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", 
                        type_importance=simulation_C["imprtnc_type_sims"][i], 
                        title_importance=simulation_C["imprtnc_title_sims"][i])
  


# ##  Experiment S: S->A vs. S->C


simulation_S = {
  "trials": 5,
  "consts_sa": np.arange(0, 10, 0.25),
  "consts_sc": np.arange(0, 10, 0.25),
  "deltas": np.array([0.1, 0.5, 1, 2.5]),
  "type_effort": ["Lowest", "Low", "Mid", "High"],
  "type_sims": ["S_A_", "S_C_", "S_AC_"],
  "imprtnc_type_sims": ["S_A", "S_C", "S_Difference_between_A_and_C", "S_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}


ei_data_S = ei_violation_all_improve(LogisticRegression(), trials = simulation_S["trials"],
                                             params = default_params.copy(),
                                             deltas = simulation_S["deltas"], 
                                             str_var_1 = 's_a_const',
                                             vars_1 = simulation_S["consts_sa"], 
                                             str_var_2 = 's_c_const',
                                             vars_2 = simulation_S["consts_sc"])




#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_S["deltas"]):
    if simulation_S["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_S["ei_data"][i, d], ei_data_S["mask"][i, d], 
                                  simulation_S["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                                  simulation_S["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", 
                                  simulation_S["type_effort"][d], simulation_S["type_sims"][i], delta)



#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_S["feature_weights_AC"][0] - ei_data_S["feature_weights_AC"][1], 
                        simulation_S["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_S["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)",  
                        type_importance=simulation_S["imprtnc_type_sims"][i], 
                        title_importance=simulation_S["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_S["feature_weights_AC"][0] - ei_data_S["feature_weights_AC"][1], 
                        simulation_S["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_S["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)",  
                        type_importance=simulation_S["imprtnc_type_sims"][i], 
                        title_importance=simulation_S["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_S["feature_weights_AC"][i], 
                        simulation_S["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_S["consts_sc"], r"$\omega_s$ (S$\rightarrow$C)", 
                        type_importance=simulation_S["imprtnc_type_sims"][i], 
                        title_importance=simulation_S["imprtnc_title_sims"][i])
  


# ##  Experiment SY: S->A vs. S->Y



simulation_SY = {
  "trials": 5,
  "consts_sa": np.arange(0, 10, 0.5),
  "consts_sy": np.arange(0, 10, 0.5),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["SY_A_", "SY_C_", "SY_AC_"],
  "imprtnc_type_sims": ["SY_A", "SY_C", "SY_Difference_between_A_and_C", "SY_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}

cost_fn ={'w': np.array([100000, 1]),'b': 0}

ei_data_SY_test = ei_violation(LogisticRegression(), simulation_SY["trials"], 
                               params = default_params.copy(),
                               deltas = simulation_SY["deltas"], 
                               cost_fn = cost_fn,
                               str_var_1 = 's_a_const',
                               vars_1 = simulation_SY["consts_sa"], 
                               str_var_2 = 's_y_const',
                               vars_2 = simulation_SY["consts_sy"])

ei_data_SY = ei_violation_all_improve(LogisticRegression(), simulation_SY["trials"], 
                                               params = default_params.copy(),
                                               deltas = simulation_SY["deltas"], 
                                               str_var_1 = 's_a_const',
                                               vars_1 = simulation_SY["consts_sa"], 
                                               str_var_2 = 's_y_const',
                                               vars_2 = simulation_SY["consts_sy"])


#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_SY["deltas"]):
    if simulation_SY["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_SY["ei_data"][i, d], ei_data_SY["mask"][i, d], 
                                  simulation_SY["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                                  simulation_SY["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                                  simulation_SY["type_effort"][d], simulation_SY["type_sims"][i], delta)
  



#DIFF IN IMPROVABILITY GRAPHS
for d, delta in enumerate(simulation_SY["deltas"]):
  #print(ei_data_SY_test["ei_data"][d])
  #print(ei_data_SY_test["mask"][d])
  all_improve_heat_map_effort(ei_data_SY_test["ei_data"][d], ei_data_SY_test["mask"][d], 
                              simulation_SY["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                              simulation_SY["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                              simulation_SY["type_effort"][d], "SY_TEST_AC", delta)
  


#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_SY["feature_weights_AC"][0] - ei_data_SY["feature_weights_AC"][1], 
                        simulation_SY["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_SY["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_SY["imprtnc_type_sims"][i], 
                        title_importance=simulation_SY["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_SY["feature_weights_AC"][0] - ei_data_SY["feature_weights_AC"][1], 
                        simulation_SY["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_SY["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_SY["imprtnc_type_sims"][i], 
                        title_importance=simulation_SY["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_SY["feature_weights_AC"][i], 
                        simulation_SY["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_SY["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_SY["imprtnc_type_sims"][i], 
                        title_importance=simulation_SY["imprtnc_title_sims"][i])
  



print(default_params)


# ##  Experiment SC: S->C vs. S->Y


simulation_SC = {
  "trials": 5,
  "consts_sc": np.arange(0, 10, 0.25),
  "consts_sy": np.arange(0, 10, 0.25),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["SC_A_", "SC_C_", "SC_AC_"],
  "imprtnc_type_sims": ["SC_A", "SC_C", "SC_Difference_between_A_and_C", "SC_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}

ei_data_SC = ei_violation_all_improve(LogisticRegression(), simulation_SC["trials"], 
                                               params = default_params.copy(),
                                               deltas = simulation_SC["deltas"], 
                                               str_var_1 = 's_c_const',
                                               vars_1 = simulation_SC["consts_sc"], 
                                               str_var_2 = 's_y_const',
                                               vars_2 = simulation_SC["consts_sy"])



#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_SC["deltas"]):
    if simulation_SC["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_SC["ei_data"][i, d], ei_data_SC["mask"][i, d], 
                                  simulation_SC["consts_sc"], r"$\omega_S$ (S$\rightarrow$C)", 
                                  simulation_SC["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                                  simulation_SC["type_effort"][d], simulation_SC["type_sims"][i], delta)
  


#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    importance_heat_map(ei_data_SC["feature_weights_AC"][0] - ei_data_SC["feature_weights_AC"][1], 
                        simulation_SC["consts_sc"], r"$\omega_S$ (S$\rightarrow$C)", 
                        simulation_SC["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_SC["imprtnc_type_sims"][i], 
                        title_importance=simulation_SC["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_SC["feature_weights_AC"][0] - ei_data_SC["feature_weights_AC"][1], 
                        simulation_SC["consts_sc"], r"$\omega_S$ (S$\rightarrow$C)", 
                        simulation_SC["consts_sy"], r"$m_S$ (S$\rightarrow$Y)",  
                        type_importance=simulation_SC["imprtnc_type_sims"][i], 
                        title_importance=simulation_SC["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_SC["feature_weights_AC"][i], 
                        simulation_SC["consts_sc"], r"$\omega_S$ (S$\rightarrow$C)", 
                        simulation_SC["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_SC["imprtnc_type_sims"][i], 
                        title_importance=simulation_SC["imprtnc_title_sims"][i])
  


# ## Experiment Y: C->Y vs S->Y



simulation_Y = {
  "trials": 5,
  "consts_cy": np.arange(0, 10, 0.25),
  "consts_sy": np.arange(0, 10, 0.25),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["Y_A_", "Y_C_", "Y_AC_"],
  "imprtnc_type_sims": ["Y_A", "Y_C", "Y_Difference_between_A_and_C", "Y_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}

ei_data_Y = ei_violation_all_improve(LogisticRegression(), 
                                             trials = simulation_Y["trials"],
                                             params = default_params.copy(),
                                             deltas = simulation_Y["deltas"], 
                                             str_var_1 = 'c_y_consts',
                                             vars_1 = simulation_Y["consts_cy"], 
                                             str_var_2 = 's_y_const',
                                             vars_2 =  simulation_Y["consts_sy"])



#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_Y["deltas"]):
    if simulation_Y["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_Y["ei_data"][i, d], ei_data_Y["mask"][i, d], 
                                  simulation_Y["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                                  simulation_Y["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                                  simulation_Y["type_effort"][d], simulation_Y["type_sims"][i], delta)
  



#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_Y["feature_weights_AC"][0] - ei_data_Y["feature_weights_AC"][1], 
                        simulation_Y["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        simulation_Y["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_Y["imprtnc_type_sims"][i], 
                        title_importance=simulation_Y["imprtnc_title_sims"][i], save=True)
  elif i == 3:
    categorical_heatmap(ei_data_Y["feature_weights_AC"][0] - ei_data_Y["feature_weights_AC"][1], 
                        simulation_Y["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        simulation_Y["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_Y["imprtnc_type_sims"][i], 
                        title_importance=simulation_Y["imprtnc_title_sims"][i], save=True)
  else:
    importance_heat_map(ei_data_Y["feature_weights_AC"][i], 
                        simulation_Y["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        simulation_Y["consts_sy"], r"$m_S$ (S$\rightarrow$Y)", 
                        type_importance=simulation_Y["imprtnc_type_sims"][i], 
                        title_importance=simulation_Y["imprtnc_title_sims"][i], save=True)
  


# ## Experiment ACY: A->C vs. C->Y



simulation_ACY = {
  "trials": 5,
  "consts_cy": np.arange(0, 10, 0.5),
  "consts_ac": np.arange(0, 10, 0.5),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["C_A_", "C_C_", "C_AC_"],
  "imprtnc_type_sims": ["C_A", "C_C", "C_Difference_between_A_and_C", "C_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}

ACY_params = default_params.copy()
ACY_params['s_a_const'] = 2.2
ACY_params['s_y_const'] = 2.5

ei_data_ACY = ei_violation_all_improve(LogisticRegression(), 
                                             trials= simulation_ACY["trials"],
                                             params = ACY_params,
                                             deltas = simulation_ACY["deltas"], 
                                             str_var_1 = 'a_c_const',
                                             vars_1 = simulation_ACY["consts_ac"], 
                                             str_var_2 = 'c_y_const',
                                             vars_2 = simulation_ACY["consts_cy"])



#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_ACY["deltas"]):
    if simulation_ACY["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_ACY["ei_data"][i, d], ei_data_ACY["mask"][i, d], 
                                  simulation_ACY["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                                  simulation_ACY["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                                  simulation_ACY["type_effort"][d], simulation_ACY["type_sims"][i], delta)
  



#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_ACY["feature_weights_AC"][0] - ei_data_ACY["feature_weights_AC"][1], 
                        simulation_ACY["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        simulation_ACY["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        type_importance=simulation_ACY["imprtnc_type_sims"][i], 
                        title_importance=simulation_ACY["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_ACY["feature_weights_AC"][0] - ei_data_ACY["feature_weights_AC"][1], 
                        simulation_ACY["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        simulation_ACY["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        type_importance=simulation_ACY["imprtnc_type_sims"][i], 
                        title_importance=simulation_ACY["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_ACY["feature_weights_AC"][i], 
                        simulation_ACY["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        simulation_ACY["consts_cy"], r"$m_C$ (C$\rightarrow$Y)", 
                        type_importance=simulation_ACY["imprtnc_type_sims"][i], 
                        title_importance=simulation_ACY["imprtnc_title_sims"][i])
    


# ## Experiment A: S->A vs. A->C


simulation_A = {
  "trials": 10,
  "consts_sa": np.arange(0, 10, 0.25),
  "consts_ac": np.arange(0, 10, 0.25),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["A_A_", "A_C_", "A_AC_"],
  "imprtnc_type_sims": ["A_A", "A_C", "A_Difference_between_A_and_C", "A_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}


ei_data_A = ei_violation_all_improve(LogisticRegression(), 
                                     trials= simulation_A['trials'],
                                     params = default_params.copy(),
                                     deltas = simulation_A["deltas"], 
                                     str_var_1 = 's_a_const',
                                     vars_1 = simulation_A["consts_sa"], 
                                     str_var_2 = 'a_c_const',
                                     vars_2 = simulation_A["consts_ac"])




#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_A["deltas"]):
    if simulation_A["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_A["ei_data"][i, d], ei_data_A["mask"][i, d], 
                                  simulation_A["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                                  simulation_A["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)",
                                  simulation_A["type_effort"][d], simulation_A["type_sims"][i], delta)
  


#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_A["feature_weights_AC"][0] - ei_data_A["feature_weights_AC"][1], 
                        simulation_A["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_A["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        type_importance=simulation_A["imprtnc_type_sims"][i], 
                        title_importance=simulation_A["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_A["feature_weights_AC"][0] - ei_data_A["feature_weights_AC"][1], 
                        simulation_A["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_A["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        type_importance=simulation_A["imprtnc_type_sims"][i], 
                        title_importance=simulation_A["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_A["feature_weights_AC"][i], 
                        simulation_A["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_A["consts_ac"], r"$\omega_A$ (A$\rightarrow$C)", 
                        type_importance=simulation_A["imprtnc_type_sims"][i], 
                        title_importance=simulation_A["imprtnc_title_sims"][i])
  


# ## Experiment O1: S->A vs. C->Y


simulation_O1 = {
  "trials": 10,
  "consts_sa": np.arange(0, 10, 0.25),
  "consts_cy": np.arange(0, 10, 0.25),
  "deltas": np.array([0.1, 0.5, 1, 2.5, 5, 10]),
  "type_effort": ["Lowest", "Low", "Mid", "High", "Higher", "Highest"],
  "type_sims": ["O1_A_", "O1_C_", "O1_AC_"],
  "imprtnc_type_sims": ["O1_A", "O1_C", "O1_Difference_between_A_and_C", "O1_Improve_A_or_C"],
  "imprtnc_title_sims": ["A", "C", "Difference between A and C", "Categorical Heatmap of A vs. C Importance"]
}


ei_data_O1 = ei_violation_all_improve(LogisticRegression(), 
                                      trials = simulation_O1['trials'],
                                      params = default_params.copy(),
                                      deltas = simulation_O1["deltas"], 
                                      str_var_1 = 's_a_const',
                                      vars_1 = simulation_O1["consts_sa"], 
                                      str_var_2 = 'c_y_const',
                                      vars_2 = simulation_O1["consts_cy"])


#DIFF IN IMPROVABILITY GRAPHS
for i in range(3):
  for d, delta in enumerate(simulation_O1["deltas"]):
    if simulation_O1["type_effort"][d] != None:
      all_improve_heat_map_effort(ei_data_O1["ei_data"][i, d], ei_data_O1["mask"][i, d], 
                                  simulation_O1["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                                  simulation_O1["consts_cy"], r"$m_C$ (C$\rightarrow$Y)",
                                  simulation_O1["type_effort"][d], simulation_O1["type_sims"][i], delta)
  

#IMPORTANCE GRAPHS
for i in range(4):
  if i == 2:
    diff_importance_heatmap(ei_data_O1["feature_weights_AC"][0] - ei_data_O1["feature_weights_AC"][1], 
                        simulation_O1["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_O1["consts_cy"], r"$m_C$ (C$\rightarrow$Y)",
                        type_importance=simulation_O1["imprtnc_type_sims"][i], 
                        title_importance=simulation_O1["imprtnc_title_sims"][i])
  elif i == 3:
    categorical_heatmap(ei_data_O1["feature_weights_AC"][0] - ei_data_O1["feature_weights_AC"][1], 
                        simulation_O1["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_O1["consts_cy"], r"$m_C$ (C$\rightarrow$Y)",
                        type_importance=simulation_O1["imprtnc_type_sims"][i], 
                        title_importance=simulation_O1["imprtnc_title_sims"][i])
  else:
    importance_heat_map(ei_data_O1["feature_weights_AC"][i], 
                        simulation_O1["consts_sa"], r"$\alpha$ (S$\rightarrow$A)", 
                        simulation_O1["consts_cy"], r"$m_C$ (C$\rightarrow$Y)",
                        type_importance=simulation_O1["imprtnc_type_sims"][i], 
                        title_importance=simulation_O1["imprtnc_title_sims"][i])
  


# # Improve Rate

def line_plot_improv_rate(data, deltas, rate = 0.05):
  data_rate = (data[1:] - data[:-1])/rate
  plt.plot((deltas[1:] + deltas[:-1])/2, data_rate)
  plt.xlabel('Delta')
  plt.ylabel('Rate of Improvability')
  plt.title('Rate of Improvability vs. Delta')
  plt.show()

def test_improv_rate(rate = 0.05):
  n = 1000
  a = np.random.normal(loc=0.0, scale=1.0, size=n)
    #np.random.uniform(low=0.0, high=1.0, size=n)
    #
  #print(a)
  deltas = np.arange(0, 2, rate)

  data = np.ones_like(deltas)*-1

  for i, delta in enumerate(deltas):
    temp_a = a + delta
    data[i] = np.mean((temp_a >= 1.1) & (a <= 1.1))
  
  graph_improv_rate(data, deltas, rate)
    
    
def heat_map_improve_rate(results, mask, var_delta, var_J,  str_var_J, group = "NA", improve_type = "NA", type_sim="None", save=False):
  results = np.flip(results, axis = 0)
  var_delta = np.flip(var_delta, axis = 0)
  #if mask != None:
  mask = np.flip(mask, axis = 0)
  map = sns.heatmap(results, cmap='viridis', mask=mask)#, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
  #, cbar_kws={'label': 'P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 0) - P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 1)'})

  # Set axis labels and plot title
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + r"$\Delta$")
  if group == "01":
    plt.title("Difference in rate of improvability (improving "+improve_type+")")
  else:
    plt.title("Rate of Improvability for group " + group + " (improving "+improve_type+")")
    

  # Customize the tick labels on the x and y axes
  step_size_I = max(len(var_delta) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_delta[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_delta), step_size_I) + 0.5, ytick_labels)
  # Show the heatmap
  if save:
      plt.savefig("graphs/"+type_sim+group+"_improve_rate")
  plt.show()


def improv_rate_calc_heat_map(trainer, params, str_var, var_consts, trials, rate, deltas):
  
  data_improve = np.zeros((3, 2, deltas.shape[0], var_consts.shape[0]))
  mask = np.zeros((3, deltas.shape[0], var_consts.shape[0]))
  improve_rate = np.zeros((3, 2, deltas.shape[0]-1, var_consts.shape[0]))
  
  #print(a)
  for i, var_const in enumerate(var_consts):
    A_failed_trials = np.zeros(deltas.shape[0])
    C_failed_trials = np.zeros(deltas.shape[0])
    AC_failed_trials = np.zeros(deltas.shape[0])
    for t in range(trials):
      params[str_var] = var_const
      data = ScalarLinearDecisionModel(params)
      data.generate_basic_data()
      data_A = data.A.reshape((data.n_samples,1))
      data_C = data.C.reshape((data.n_samples,1))
      data_S = data.S
      data_AC = np.concatenate((data_A, data_C), axis=1)
      data_Y = data.Y.reshape(data.n_samples)
      model = trainer.fit(data_AC, data_Y)
      pred_y = model.predict(data_AC)
      data_S_y = data_S[pred_y == 0]
      for d, delta in enumerate(deltas):
        data_improve_A_A = data_A[pred_y == 0] + delta
        data_improve_A_C, data_improve_A_Y  = data.generate_improve_A_data(delta)
        data_improve_A_C = data_improve_A_C.reshape((data.n_samples,1))
        data_improve_A_AC = np.concatenate((data_improve_A_A, data_improve_A_C[pred_y == 0]), axis=1)
        data_improve_C_AC = np.concatenate((data_A[pred_y == 0], data_C[pred_y == 0] + delta), axis=1)
        #print(data.s_y_const)
        #print(data_improve_A_A[data_S[pred_y == 0] == 0].shape)
        #print(data_improve_A_A[data_S[pred_y == 0] == 1].shape)
        #DATA FOR JUST CHANGING A
        temp_A_y = model.predict(data_improve_A_AC)
        if len(temp_A_y[data_S_y == 1]) != 0:
          data_improve[0, 0, d, i] += np.mean(temp_A_y[data_S_y == 0])/trials
          data_improve[0, 1, d, i] += np.mean(temp_A_y[data_S_y == 1])/trials
        else:
          A_failed_trials[d] += 1
        temp_C_y = model.predict(data_improve_C_AC)
        if len(temp_C_y[data_S_y == 1]) != 0:
          data_improve[1, 0, d, i] += np.mean(temp_C_y[data_S_y == 0])/trials
          data_improve[1, 1, d, i] += np.mean(temp_C_y[data_S_y == 1])/trials
        else:
          C_failed_trials[d] += 1
        temp_y = np.max((temp_A_y, temp_C_y), axis=0)
        if len(temp_y[data_S_y == 1]) != 0:
          data_improve[2, 0, d, i] += np.mean(temp_y[data_S_y == 0])/trials
          data_improve[2, 1, d, i] += np.mean(temp_y[data_S_y == 1])/trials
        else:
          AC_failed_trials[d] += 1
    for d, delta in enumerate(deltas):
      if A_failed_trials[d] >= trials*3/4:
        mask[0, d, i] = True
      elif A_failed_trials[d] > 0:
        data_improve[0, 0, d, i] = data_improve[0, 0, d, i]*trials/(trials - A_failed_trials[d])
        data_improve[0, 1, d, i] = data_improve[0, 1, d, i]*trials/(trials - A_failed_trials[d])
      if C_failed_trials[d] >= trials*3/4:
        mask[1, d, i] = True
      elif C_failed_trials[d] > 0:
         data_improve[1, 0, d, i] = data_improve[1, 0, d, i]*trials/(trials - C_failed_trials[d])
         data_improve[1, 1, d, i] = data_improve[1, 1, d, i]*trials/(trials - C_failed_trials[d])
      if AC_failed_trials[d] >= trials*3/4:
        mask[2, d, i] = True
      elif AC_failed_trials[d] > 0:
         data_improve[2, 0, d, i] = data_improve[2, 0, d, i]*trials/(trials - AC_failed_trials[d])
         data_improve[2, 1, d, i] = data_improve[2, 1, d, i]*trials/(trials - AC_failed_trials[d])
        
  improve_rate = (data_improve[:, :, 1:, :] - data_improve[:, :, :-1, :])/rate
  improve_mask = np.logical_and(mask[:, 1:, :], mask[:, :-1, :])
  improve_deltas = (deltas[1:] + deltas[:-1])/2
  #print(np.max(improve_rate[:,:,1], axis = 0))
  #print(np.argmax(improve_rate[:,:,1], axis = 0))
  results = {
      'improve_rate': improve_rate, 
      'improve_deltas': improve_deltas,
      'mask': improve_mask
  }
  return results


SY_ir_sim = {
    'improved': ["A", "C", "A and C"],
    'labels': ["SY_A_", "SY_C_", "SY_AC_"],
    'groups': ["0", "1", "01"],
    'str_var': 's_y_const', 
    'var_consts': np.arange(0, 5, 0.5), 
    'trials': 5, 
    'rate': 0.5, 
    'delta_min': 0, 
    'delta_max': 5
}


ir_SY = improv_rate_calc_heat_map(trainer = LogisticRegression(), 
                                  params = default_params.copy(),
                                  str_var = SY_ir_sim['str_var'],
                                  var_consts = SY_ir_sim['var_consts'], 
                                  trials = SY_ir_sim['trials'], 
                                  rate = SY_ir_sim['rate'], 
                                  deltas = np.arange(SY_ir_sim['delta_min'], SY_ir_sim['delta_max'], SY_ir_sim['rate'])
                                 )







print(ir_SY['improve_rate'].shape)
print(ir_SY['mask'].shape)
for i, label in enumerate(SY_ir_sim['labels']):
  for j, group in enumerate(SY_ir_sim['groups']):
    if j == 2:
      heat_map_improve_rate(ir_SY['improve_rate'][i, 1] - ir_SY['improve_rate'][i, 0], ir_SY['mask'][i], 
                            ir_SY['improve_deltas'], SY_ir_sim['var_consts'],  r"$m_S$ (S$\rightarrow$Y)", 
                            group = group, improve_type = SY_ir_sim['improved'][i], type_sim=label)
    else:
      heat_map_improve_rate(ir_SY['improve_rate'][i, j], ir_SY['mask'][i], 
                            ir_SY['improve_deltas'], SY_ir_sim['var_consts'],  r"$m_S$ (S$\rightarrow$Y)", 
                            group = group, improve_type = SY_ir_sim['improved'][i], type_sim=label)

SC_ir_sim = {
    'improved': ["A", "C", "A and C"],
    'labels': ["SC_A_", "SC_C_", "SC_AC_"],
    'groups': ["0", "1", "01"],
    'str_var': 's_c_const', 
    'var_consts': np.arange(0, 5, 0.25), 
    'trials': 5, 
    'rate': 0.1, 
    'delta_min': 0, 
    'delta_max': 5
}


ir_SC = improv_rate_calc_heat_map(trainer = LogisticRegression(), 
                                  params = default_params.copy(),
                                  str_var = SC_ir_sim['str_var'],
                                  var_consts = SC_ir_sim['var_consts'], 
                                  trials = SC_ir_sim['trials'], 
                                  rate = SC_ir_sim['rate'], 
                                  deltas = np.arange(SC_ir_sim['delta_min'], SC_ir_sim['delta_max'], SC_ir_sim['rate'])
                                 )
 

for i, label in enumerate(SC_ir_sim['labels']):
  for j, group in enumerate(SC_ir_sim['groups']):
    if j == 2:
      heat_map_improve_rate(ir_SC['improve_rate'][i, 1] - ir_SC['improve_rate'][i, 0], ir_SC['mask'][i], 
                            ir_SC['improve_deltas'], SC_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = SC_ir_sim['improved'][i], type_sim=label)
    else:
      heat_map_improve_rate(ir_SC['improve_rate'][i, j], ir_SC['mask'][i], 
                            ir_SC['improve_deltas'], SC_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = SC_ir_sim['improved'][i], type_sim=label)


SA_ir_sim = {
    'improved': ["A", "C", "A and C"],
    'labels': ["SA_A_", "SA_C_", "SA_AC_"],
    'groups': ["0", "1", "01"],
    'str_var': 's_c_const', 
    'var_consts': np.arange(0, 5, 0.25), 
    'trials': 5, 
    'rate': 0.1, 
    'delta_min': 0, 
    'delta_max': 5
}


ir_SA = improv_rate_calc_heat_map(trainer = LogisticRegression(), 
                                  params = default_params.copy(),
                                  str_var = SA_ir_sim['str_var'],
                                  var_consts = SA_ir_sim['var_consts'], 
                                  trials = SA_ir_sim['trials'], 
                                  rate = SA_ir_sim['rate'], 
                                  deltas = np.arange(SA_ir_sim['delta_min'], SA_ir_sim['delta_max'], SA_ir_sim['rate'])
                                 )
 

for i, label in enumerate(SA_ir_sim['labels']):
  for j, group in enumerate(SA_ir_sim['groups']):
    if j == 2:
      heat_map_improve_rate(ir_SA['improve_rate'][i, 1] - ir_SA['improve_rate'][i, 0], ir_SA['mask'][i], 
                            ir_SA['improve_deltas'], SA_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = SA_ir_sim['improved'][i], type_sim=label)
    else:
      heat_map_improve_rate(ir_SA['improve_rate'][i, j], ir_SA['mask'][i], 
                            ir_SA['improve_deltas'], SA_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = SA_ir_sim['improved'][i], type_sim=label)


AC_ir_sim = {
    'improved': ["A", "C", "A and C"],
    'labels': ["AC_A_", "AC_C_", "AC_AC_"],
    'groups': ["0", "1", "01"],
    'str_var': 's_c_const', 
    'var_consts': np.arange(0, 5, 0.25), 
    'trials': 5, 
    'rate': 0.1, 
    'delta_min': 0, 
    'delta_max': 5
}


ir_AC = improv_rate_calc_heat_map(trainer = LogisticRegression(), 
                                  params = default_params.copy(),
                                  str_var = AC_ir_sim['str_var'],
                                  var_consts = AC_ir_sim['var_consts'], 
                                  trials = AC_ir_sim['trials'], 
                                  rate = AC_ir_sim['rate'], 
                                  deltas = np.arange(AC_ir_sim['delta_min'], AC_ir_sim['delta_max'], AC_ir_sim['rate'])
                                 )
 

for i, label in enumerate(AC_ir_sim['labels']):
  for j, group in enumerate(AC_ir_sim['groups']):
    if j == 2:
      heat_map_improve_rate(ir_AC['improve_rate'][i, 1] - ir_AC['improve_rate'][i, 0], ir_AC['mask'][i], 
                            ir_AC['improve_deltas'], AC_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = AC_ir_sim['improved'][i], type_sim=label)
    else:
      heat_map_improve_rate(ir_AC['improve_rate'][i, j], ir_AC['mask'][i], 
                            ir_AC['improve_deltas'], AC_ir_sim['var_consts'],  r"$\omega_S$ (S$\rightarrow$C)", 
                            group = group, improve_type = AC_ir_sim['improved'][i], type_sim=label)


# # Variance of rare samples

#VARIANCE

def calculate_variance(models, X):
    predictions = []
    for model in models:
        prediction = model.predict_proba(X)[:, 1]
        predictions.append(prediction)
    predictions = np.array(predictions)
    return np.var(predictions, axis=0)

X_train, X_test, y_train, y_test = train_test_split(data_AC, data_Y, test_size=0.33, random_state=42)



# Number of logistic regression models to train
num_models = 10

# Train the logistic regression models
models = []
for i in range(num_models):
    model = MLPClassifier(max_iter=500, solver = 'sgd')#LogisticRegression()
    model.fit(X_train, y_train)
    models.append(model)

# Calculate the variance of the output function
variance = calculate_variance(models, X_temp)

print("Variance of the output function:", np.round(variance, 4))
print(sum(variance))

