
# This code is coppied from pcntoolkit and slightly modeified

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# confidence interval calculation at x_forward
def confidence_interval(s2,x,z, x_forward, n):
    CI=np.zeros((len(x_forward),n))
    for i,xdot in enumerate(x_forward):
        ci_inx=np.isin(x,xdot)
        S2=s2[ci_inx]
        S_hat=np.mean(S2,axis=0)
        n=S2.shape[0]
        CI[i,:]=z*np.power(S_hat/n,.5)
    return CI 



covariate_normsample = pd.read_csv("data/normativeInput/covariateNormSample.txt", 
                                   header=None,
                                   sep=" ")


feature_names=["alpha", "beta", "delta", "exponent", "gamma", "offset", "theta"]
sex_covariates=["1",'2']
c=0
fig, ax=plt.subplots(7,2,figsize=(10, 25))

# Creating plots for Female and male
for i,sex in enumerate(sex_covariates):
    #forward model data
    forward_yhat = pd.read_csv('data/normativeOutput/yhat_estimate.txt', sep = ' ', header=None)
    yhat_forward=forward_yhat.values
    yhat_forward=yhat_forward[7*i:7*(i+1)]
    x_forward=[20, 30, 40, 50, 60, 70, 80]

    # Find the index of the data exclusively for one sex. Female:0, Male: 1
    inx=np.where(covariate_normsample.iloc[:,0]==i+1)[0]
    x=covariate_normsample.values[inx,1]

# actual data
    y = pd.read_csv('/home/zamanzad/trial1/data/normativeInput/responseVarNorm.txt', sep = ' ', header=None)
    y=y.values[inx]
# confidence Interval yhat+ z *(std/n^.5)-->.95 % CI:z=1.96, 99% CI:z=2.58
    s2= pd.read_csv('data/normativeOutput/ys2_estimate.txt', sep = ' ', header=None)
    s2=s2.values[inx]
    
    CI_95=confidence_interval(s2,x,1.96,x_forward, len(feature_names))
    CI_99=confidence_interval(s2,x,2.58, x_forward, len(feature_names))


    # Creat a trejactroy for each point
    
    for j,name in enumerate(feature_names):
        ax[j, c].plot(x_forward,yhat_forward[:,j], linewidth=4, label='Normative trejactory', color="black")


        ax[j, c].plot(x_forward,CI_95[:,j]+yhat_forward[:,j], linewidth=2,linestyle='--',c='g', label='95% confidence interval')
        ax[j, c].plot(x_forward,-CI_95[:,j]+yhat_forward[:,j], linewidth=2,linestyle='--',c='g')

        ax[j, c].plot(x_forward,CI_99[:,j]+yhat_forward[:,j], linewidth=1,linestyle='--',c='k', label='99% confidence interval')
        ax[j, c].plot(x_forward,-CI_99[:,j]+yhat_forward[:,j], linewidth=1,linestyle='--',c='k')

        ax[j, c].scatter(x,y[:,j], label=name, color="orange")

        ax[j,c].set_title(f"{name} - {sex}")

    c+=1
    


plt.legend(loc='upper left')
plt.savefig(f"pictures/normative/res_.jpg", bbox_inches="tight")
plt.close()