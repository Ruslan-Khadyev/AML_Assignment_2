import torch
import numpy as np
import matplotlib.pyplot as plt


def loss_function(output: torch.Tensor, data_point: torch.Tensor):
    data_length = data_point.shape[1]
    x_data = data_point[:, :data_length // 2]
    indicator = data_point[:, data_length // 2:]

    return torch.norm(indicator * x_data - indicator * output)


def plot_explained_variance(x, datalen):
    #Calculating Eigenvecors and eigenvalues of Covariance matrix
    mean_vec = np.mean(x, axis=0)
    cov_mat = np.cov(x.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

    print("Individual value for the selected point", var_exp[datalen-1])
    print("Cumulative value for the selected point", cum_var_exp[datalen-1])

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(var_exp[:datalen])), var_exp[:datalen], alpha=0.3333, align='center', label='individual explained variance', color = 'g')
    plt.step(range(len(cum_var_exp[:datalen])), cum_var_exp[:datalen], where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()