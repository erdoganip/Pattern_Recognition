#Ipek Erdogan
#Pattern Recognition Homework 2
#2019700174

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def data_loader(path):
    data = np.load(path)
    return data

def random_sigma(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())

def initialize_random_params():
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    c = np.random.uniform(0, 1)
    sum_ = a + b + c
    params = {'phi0': a/sum_,
              'phi1': b/sum_,
              'phi2': c/sum_,
              'mu0': np.random.normal(0, 1, size=(2,)),
              'mu1': np.random.normal(0, 1, size=(2,)),
              'mu2': np.random.normal(0, 1, size=(2,)),
              'sigma0': random_sigma(2),
              'sigma1': random_sigma(2),
              'sigma2': random_sigma(2)}
    return params


def e_step(x, params):
    likelihood_0= stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x)
    likelihood_1= stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)
    likelihood_2= stats.multivariate_normal(params["mu2"], params["sigma2"]).pdf(x)
    phi_0= params["phi0"]
    phi_1= params["phi1"]
    phi_2= params["phi2"]
    posterior_0= phi_0 * likelihood_0
    posterior_1= phi_1 * likelihood_1
    posterior_2= phi_2 * likelihood_2
    post_sum_ = np.add(posterior_0, posterior_1)
    post_sum = np.add(post_sum_, posterior_2)
    probabilities_0 = np.divide(posterior_0,post_sum)
    probabilities_1 = np.divide(posterior_1,post_sum)
    probabilities_2 = np.divide(posterior_2,post_sum)
    avg_likelihood = np.array([np.mean(np.log(likelihood_0)), np.mean(np.log(likelihood_1)), np.mean(np.log(likelihood_2))])
    posteriors = np.array([probabilities_0,probabilities_1,probabilities_2])
    return avg_likelihood, probabilities_0, probabilities_1, probabilities_2, posteriors

def m_step(x, params):
    total_count = x.shape[0]
    _ , prob0, prob1, prob2, posteriors = e_step(x, params)
    sum_prob0 = np.sum(prob0)
    sum_prob1 = np.sum(prob1)
    sum_prob2 = np.sum(prob2)
    phi0 = (sum_prob0 / total_count)
    phi1 = (sum_prob1 / total_count)
    phi2 = (sum_prob2 / total_count)
    mu0 = (prob0.T.dot(x)/sum_prob0).flatten()
    mu1 = (prob1.T.dot(x)/sum_prob1).flatten()
    mu2 = (prob2.T.dot(x)/sum_prob2).flatten()
    diff0 = x - mu0
    sigma0 = diff0.T.dot(diff0 * prob0[..., np.newaxis]) / sum_prob0
    diff1 = x - mu1
    sigma1 = diff1.T.dot(diff1 * prob1[..., np.newaxis]) / sum_prob1
    diff2 = x - mu2
    sigma2 = diff2.T.dot(diff2 * prob2[..., np.newaxis]) / sum_prob2
    params = {'phi0': phi0, 'phi1': phi1, 'phi2': phi2, 'mu0': mu0, 'mu1': mu1, 'mu2': mu2, 'sigma0': sigma0, 'sigma1': sigma1, 'sigma2': sigma2}
    return params

def get_avg(x, params):
    log_likelihood,_,_,_,_ = e_step(x, params)
    return np.mean(log_likelihood)

def em_algorithm(x, params):
    avg_lh = []
    while True:
        avg_log = get_avg(x, params)
        avg_lh.append(avg_log)
        if (len(avg_lh) > 2) and (abs(avg_lh[-1]- avg_lh[-2]) < 0.0000001):
            break
        params = m_step(x, params)
    print("\tphi0: %s\n\tphi1: %s\n\tphi2: %s\n\tmu0: %s\n\tmu1: %s\n\tmu2: %s\n\tsigma0: %s\n\tsigma1: %s\n\tsigma2: %s"
               % (params['phi0'],params['phi1'],params['phi2'], params['mu0'], params['mu1'],params['mu2'], params['sigma0'], params['sigma1'],params['sigma2']))
    _,_,_,_,posterior = e_step(x, params)
    return posterior, avg_lh, params


def plot_em(data, params):
    x1 = np.linspace(-4, 13, 250)
    x2 = np.linspace(-4, 13, 250)
    X, Y = np.meshgrid(x1, x2)
    mu0=params['mu0']
    sigma0=params['sigma0']
    mu1 = params['mu1']
    sigma1 = params['sigma1']
    mu2 = params['mu2']
    sigma2 = params['sigma2']
    Z1 = stats.multivariate_normal(mu0, sigma0)
    Z2 = stats.multivariate_normal(mu1, sigma1)
    Z3 = stats.multivariate_normal(mu2, sigma2)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    plt.figure(figsize=(10,10))
    plt.scatter(data[:,0], data[:,1], marker='o')
    plt.contour(X, Y, Z1.pdf(pos), colors="g", alpha = 0.5)
    plt.contour(X, Y, Z2.pdf(pos), colors="r" ,alpha = 0.5)
    plt.contour(X, Y, Z3.pdf(pos), colors="b" ,alpha = 0.5)
    plt.axis('equal')
    plt.xlabel('X-Axis', fontsize=16)
    plt.ylabel('Y-Axis', fontsize=16)
    plt.grid()
    plt.savefig("final.png")
    plt.show()

if __name__ == '__main__':
    np.random.seed(230)
    path = 'dataset.npy'
    data = data_loader(path)
    random_params = initialize_random_params()
    posterior,step_like, params = em_algorithm(data,random_params)
    print("total steps: ", len(step_like))
    plot_em(data,params)
