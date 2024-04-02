import pandas as pd
import numpy as np
# import random
import scipy as sp
import time
import seaborn as sns
import matplotlib.pyplot as plt


start_time = time.time()

dimension = 4


# #making_4d_sigmas
sigma_z = np.array([[1, 0],[0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
identity = np.identity(2)
sigmas = [identity, sigma_x, sigma_y, sigma_z]
sigmas_4d = np.kron(sigmas, sigmas)





#measurement operators
H=np.array([[1,0],[0,0]])
V=np.array([[0,0],[0,1]])
D=np.array([[0.5,0.5],[0.5,0.5]])
R=np.array([[0.5,-0.5j],[0.5j,0.5]])

measurements = [H, V, D, R]

measurements_4d = np.kron(measurements,measurements)

B_matrix = np.zeros((16,16), dtype=np.complex_)

for i in range(16):
    for j in range(16):
        B_matrix[i][j] = np.trace(np.matmul(measurements_4d[i], sigmas_4d[j]))

inv_B_matrix = np.linalg.inv(B_matrix)


# y_matrix = np.array([[1,0,0,0],[-1,2,0,0],[-1,0,2,0],[-1,0,0,2]])



# def rho_to_free_parameters(rho):
#     free_parameters = np.array([np.trace(np.matmul(rho, measurement)) for measurement in sigmas_4d])
#     return free_parameters


# def free_parameters_to_rho(free_parameters):
#     rho = np.zeros((4,4))+0j*np.zeros((4,4))
#     for i in range(dimension**2-1):
#         rho += free_parameters[i]*sigmas_4d[i]
#     rho += np.identity(4)
#     return rho/4


def generate_random_rho():
    G = np.random.normal(0,1,size=(dimension,dimension)) + 1j*np.random.normal(0,1,size=(dimension,dimension))
    rho = np.matmul(G.conj().T, G)
    rho = rho / np.trace(rho)

    return rho

# def generate_random_pure_rho():
#     psi = np.random.uniform(0, 1, size=(dimension)) + 1j*np.random.uniform(0, 1, size=(dimension))

#     rho = np.outer(psi, psi.conj())

#     rho = rho / np.trace(rho)

#     return rho

epsilon=0.0000001
oepsilon=1-epsilon
unit=np.ones(dimension)
unit=np.diag(unit)

def generate_random_pure_rho_haar_pure():
    x = sp.stats.unitary_group.rvs(dimension)
    psi=x[0]
    rho=psi*psi.conj().reshape((-1,1))
    return oepsilon*rho+epsilon*unit



# def measurement_parameters(rho):
#     return np.real([ np.trace(matrix) for matrix in np.matmul(rho, measurement_operators) ])

# def constructed_rho(rho):
#     removed_measurement = np.random.randint(0,15)
#     free_params = rho_to_free_parameters(rho) + np.random.normal(0, 0.01, size=(dimension**2-1)) 
#     free_params[removed_measurement] = 0
#     return free_params


    
def rho_to_free_parameters (rho):
    free_parameters = np.real(np.array([np.trace(np.matmul(rho, measurement)) for measurement in measurements_4d]))
    return free_parameters

def rho_to_free_parameters_with_noise_of_measurement (rho, noise_std):
    free_parameters = np.real(np.array([np.trace(np.matmul(rho, measurement)) for measurement in measurements_4d])) + np.random.normal(0, noise_std, size=(16))
    return free_parameters



def free_parameters_to_rho (free_parameters):
    constructed_rho = np.zeros((4,4))+0j*np.zeros((4,4))
    
 
    r = np.matmul(inv_B_matrix, free_parameters).reshape(4,4)
    # print(r)
    r = r/r[0][0]
    # print(r)
    

    removed_measurement = np.random.randint(0,15)
    removed_measurement_sigma1 = removed_measurement//4
    removed_measurement_sigma2 = removed_measurement % 4

    for i in range(4):
        for j in range(4):
            sigma1 = sigmas[i]
            sigma2 = sigmas[j]
            # if removed_measurement_sigma1 == i and removed_measurement_sigma2 == j:
            #     pass
            sigma12 = np.kron(sigma1, sigma2)
            constructed_rho += r[i][j] * sigma12 

    return constructed_rho/4



def constructed_rho (rho, noise_std):
    free_parameters = np.real(np.array([np.trace(np.matmul(rho, measurement)) for measurement in measurements_4d])) + np.random.normal(0, noise_std, size=(16))

    constructed_rho = np.zeros((4,4))+0j*np.zeros((4,4))
    
 
    r = np.matmul(inv_B_matrix, free_parameters).reshape(4,4)
    # print(r)
    r = r/r[0][0]
    # print(r)
    

    # removed_measurement = np.random.randint(1,15)
    # removed_measurement_sigma1 = removed_measurement//4
    # removed_measurement_sigma2 = removed_measurement % 4

    for i in range(4):
        for j in range(4):
            sigma1 = sigmas[i]
            sigma2 = sigmas[j]
            # if removed_measurement_sigma1 == i and removed_measurement_sigma2 == j:
            #     pass
            sigma12 = np.kron(sigma1, sigma2)
            constructed_rho += r[i][j] * sigma12 

    return constructed_rho/4



cols = []

for i in range(1, 17):
    cols.append('feature'+str(i))

for i in range(1, 17):
    cols.append('label'+str(i))



noise_list = [0.1, 0.01, 0.001]


for noise_std in noise_list:

    rho = generate_random_rho()

    predicted_free_parameters = rho_to_free_parameters_with_noise_of_measurement(rho, noise_std)

    real_free_parameters = rho_to_free_parameters(rho)

    data = np.append(predicted_free_parameters, real_free_parameters).reshape(1, 32)

    # print(data, len(data[0]))
    


    df = pd.DataFrame(data, columns=cols)


    # print(df)
        
    for i in range(10**5 - 10**4 - 1):
        rho = generate_random_rho()

        predicted_free_parameters = rho_to_free_parameters_with_noise_of_measurement(rho, noise_std)

        real_free_parameters = rho_to_free_parameters(rho)

        data = np.append(predicted_free_parameters, real_free_parameters)
        # print(len(data))
        df.loc[len(df)] = data


        if i%1000 == 0:
            print(i, noise_std)


    for i in range(10**4):
        rho = generate_random_pure_rho_haar_pure()

        predicted_free_parameters = rho_to_free_parameters_with_noise_of_measurement(rho, noise_std)

        real_free_parameters = rho_to_free_parameters(rho)

        data = np.append(predicted_free_parameters, real_free_parameters)

        df.loc[len(df)] = data

        
        if i%1000 == 0:
            print(i, noise_std)

    df.to_csv('Tomography - noise_' + str(noise_std) + ' - No of removed measurements_0.csv')

    print(str(noise_std)+'done!')


# print(constructed_rho_new1-rho, sp.linalg.ishermitian(constructed_rho_new1), np.real(np.linalg.eigvals(constructed_rho_new1)))




print( (time.time() - start_time))