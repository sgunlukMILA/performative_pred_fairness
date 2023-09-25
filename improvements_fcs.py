import numpy as np
from gurobipy import GRB
import gurobipy as gp


def find_pred_improve_lin_cost(dt, cost_fn, pred_fn, data):
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
    which_improve = np.zeros(n)
    delta_necessary = np.zeros(n)
    for individual in range(n):
        try:
            model = gp.Model("best_improvement")
            model.Params.LogToConsole = 0
            improve_vector = model.addMVar(shape=(1, d), name="added_vector")
            # improve_A = model.addVar(lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="improve_A")
            # improve_C = model.addVar(lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="improve_C")
            # improve_vector = np.array([improve_A, improve_C])
            # print(improve_vector.shape)
            temp_C = dt.generate_improve_data(S=data['S'][individual], A=features[individual][0],
                                              diff_A=improve_vector[:, 0],
                                              diff_C=improve_vector[:, 1], c_noise=data['c_noise'][individual])
            improved_features = np.array([improve_vector[:, 0] + features[individual][0], temp_C])
            # print(improved_features)
            model.setObjective(improve_vector @ cost_fn['w'] + cost_fn['b'], GRB.MINIMIZE)
            # Add the constraint for less than delta cost
            model.addConstr((improved_features) @ pred_fn['w'] + pred_fn['b'] >= 0.0000001)
            # model.addConstr( <= delta)
            # Optimize the model
            model.optimize()
            # Get the optimal solution
            if model.status == GRB.OPTIMAL:
                best_improvement[individual] = improve_vector.X
                # print(model.getObjective().getValue())
                delta_necessary[individual] = model.getObjective().getValue()

        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))
        # print("solved!")
        # print(best_improvement[individual, 0])
        # print(np.sign(best_improvement[individual, 0] - best_improvement[individual, 1]))
        which_improve[individual] = np.sign(best_improvement[individual, 0] - best_improvement[individual, 1])
        # print(which_improve.shape)
        # print(which_improve[individual])
    return delta_necessary, best_improvement, which_improve


def find_best_real_improve_lin_cost(dt, cost_fn, data):
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
    which_improve = np.zeros(n)
    delta_necessary = np.zeros(n)
    for individual in range(n):
        try:
            model = gp.Model("best_improvement")
            model.Params.LogToConsole = 0
            improve_vector = model.addMVar(shape=(1, d), name="added_vector")
            # improve_A = model.addVar(lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="improve_A")
            # improve_C = model.addVar(lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="improve_C")
            # improve_vector = np.array([improve_A, improve_C])
            # print(improve_vector.shape)
            temp_C = dt.generate_improve_data(S=data['S'][individual],
                                              A=features[individual][0],
                                              diff_A=improve_vector[:, 0],
                                              diff_C=improve_vector[:, 1],
                                              c_noise=data['c_noise'][individual])
            improved_features = np.array([improve_vector[:, 0] + features[individual][0], temp_C])
            # print(improved_features)
            model.setObjective(improve_vector @ cost_fn['w'] + cost_fn['b'], GRB.MINIMIZE)
            # Add the constraint for less than delta cost
            model.addConstr(temp_C * dt.c_y_const + (2 * data['S'][individual] - 1) * dt.s_y_const >= 0.0000001)
            # model.addConstr( <= delta)
            # Optimize the model
            model.optimize()
            # Get the optimal solution
            if model.status == GRB.OPTIMAL:
                best_improvement[individual] = improve_vector.X
                # print(model.getObjective().getValue())
                delta_necessary[individual] = model.getObjective().getValue()

        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))
        # print("solved!")
        # print(best_improvement[individual, 0])
        # print(np.sign(best_improvement[individual, 0] - best_improvement[individual, 1]))
        which_improve[individual] = np.sign(best_improvement[individual, 0] - best_improvement[individual, 1])
        # print(which_improve.shape)
        # print(which_improve[individual])
    return delta_necessary, best_improvement, which_improve


if __name__ == "__main__":
    from SCM import ScalarLinearDecisionModel
    from sklearn.linear_model import LogisticRegression

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

    data_train = ScalarLinearDecisionModel(default_params)
    data_train.generate_basic_data()
    data_train_AC = np.stack((data_train.A, data_train.C), axis=1)
    model = LogisticRegression().fit(data_train_AC, data_train.Y)
    pred_fn = {'w': model.coef_.flatten(), 'b': model.intercept_[0]}

    cost_fn = {'w': np.array([1, 1]), 'b': 0}

    data_summary = {
        "c_noise": data_train.c_noise,
        "S": data_train.S,
        "AC": np.stack((data_train.A, data_train.C), axis=1),
        "Y_logit": data_train.Y_logit,
        "Y": data_train.Y
    }
    pred_best_delta, pred_best_improvement, pred_which_improve = find_pred_improve_lin_cost(data_train, cost_fn,
                                                                                            pred_fn, data_summary)
    real_best_delta, real_best_improvement, real_which_improve = find_best_real_improve_lin_cost(data_train, cost_fn,
                                                                                                 data_summary)
    print('bye')
