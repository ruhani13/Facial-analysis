
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    load_data = np.load(filename)   
    return load_data - (np.mean(load_data,axis=0))


def get_covariance(dataset):
    return np.divide(np.dot(np.transpose(dataset),dataset),len(dataset)-1)


def get_eig(S, m):
    eigh_val,eigh_vec = eigh(S,subset_by_index=[len(S)-m,len(S)-1])
    eigh_val = np.diagflat(np.flip(eigh_val))
    eigh_vec = np.flip(eigh_vec,1)
    return eigh_val,eigh_vec
    
def get_eig_perc(S, perc):
    var = (np.sum(eigh(S,eigvals_only=True)))
    var=var*perc
    eigh_val,eigh_vec = eigh(S,subset_by_value =[var,np.inf])
    eigh_val = np.diagflat(np.flip(eigh_val))
    eigh_vec = np.flip(eigh_vec,1)
    return eigh_val,eigh_vec
    
def project_image(img, U):
    return np.dot(U,np.dot(np.transpose(U),img))


def display_image(orig, proj):
    orig = np.reshape(orig,(32,32),order='F')
    proj = np.reshape(proj,(32,32),order='F')
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    fig1 = ax1.imshow(orig,aspect='equal')
    fig2 = ax2.imshow(proj,aspect='equal')
    fig.colorbar(fig1,ax=ax1,fraction=0.05,pad=0.05)
    fig.colorbar(fig2,ax=ax2,fraction=0.05,pad=0.05)
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
   
