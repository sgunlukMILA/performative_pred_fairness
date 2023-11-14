import gurobipy as gp
from gurobipy import GRB
import numpy as np

def find_pred_improve_lin_cost(dgp, cost_fn, pred_fn, data):
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
  import numpy as np
  n = data["A"].shape[0]
  d = dgp.intervention_dim
  best_improvement = np.zeros((n,d))
  which_improve = np.zeros(n)
  delta_necessary = np.zeros(n)
  for individual in range(n):
    try:
      model = gp.Model("best_improvement")
      model.Params.LogToConsole = 0
      improvement_vector = model.addMVar(shape=(1,d), name="added_vector", lb=0)
      improved_features = dgp.generate_improve_data(data = data.iloc[individual], diff_vec = improvement_vector[0, :])
      # Add objective for minimal effort
      model.setObjective(improvement_vector@cost_fn['w'] + cost_fn['b'], GRB.MINIMIZE)
      # Add the constraint for improved prediction label 
      model.addConstr((improved_features)@pred_fn['w'] + pred_fn['b'] >= 0.000000001)
       # Optimize the model
      model.optimize()
      # Get the optimal solution
      if model.status == GRB.OPTIMAL:
        best_improvement[individual] = improvement_vector.X
        #print(model.getObjective().getValue())
        delta_necessary[individual] = model.getObjective().getValue()
      else:
        # Print constraints
        print("Constraints:")
        print(f"  {improved_features} @ {pred_fn['w']} + {pred_fn['b']} >= 0.000000001")

        # Print objective
        print(f"Objective: {improvement_vector} @ {cost_fn['w']} + {cost_fn['b']} (MINIMIZE)")

        print("ERROR: OPT FAILED")
        print(improvement_vector.X)
        print(model.getObjective().getValue())

    except gp.GurobiError as e:
      print("Error code " + str(e.errno) + ": " + str(e))
    
    which_improve[individual] = np.sign(best_improvement[individual, 0] - best_improvement[individual, 1])
   
  return delta_necessary, best_improvement, which_improve


def find_real_improve_lin_cost(dgp, cost_fn, data):
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
  n = data["A"].shape[0]
  d = dgp.intervention_dim

  import numpy as np
  best_improvement = np.zeros((n,d))
  which_improve = np.zeros(n)
  delta_necessary = np.zeros(n)
  for individual in range(n):
    try:
      model = gp.Model("best_improvement")
      model.Params.LogToConsole = 0
      improvement_vector = model.addMVar(shape=(1,d), name="added_vector", lb=0)
      # Add objective for minimal effort
      model.setObjective(improvement_vector@cost_fn['w'] + cost_fn['b'], GRB.MINIMIZE)
      # Add the constraint for better real label
      temp_Y_logit = dgp.generate_y_logit(data = data.iloc[individual], diff_vec = improvement_vector)
      model.addConstr(temp_Y_logit >= 0.0000001)
       # Optimize the model
      model.optimize()
      # Get the optimal solution
      if model.status == GRB.OPTIMAL:
        best_improvement[individual] = improvement_vector.X
        #print(model.getObjective().getValue())
        delta_necessary[individual] = model.getObjective().getValue()
      else:
        print("ERROR")

    except gp.GurobiError as e:
      print("Error code " + str(e.errno) + ": " + str(e))
    
    which_improve[individual] = np.sign(best_improvement[individual, 0] - best_improvement[individual, 1])

  return delta_necessary, best_improvement, which_improve


