# -*- coding: utf-8 -*-
"""
@author: Román Zúñiga Macías
"""

from numpy import count_nonzero,arange,array,dot,asarray,zeros,apply_along_axis,around,sort,shape,savetxt,array_equal,max,argmin,argmax,fill_diagonal,ones,argsort,std,diag,random,diff
from numpy.linalg import eig
from matplotlib.pyplot import plot,figure,title,legend,xlabel,ylabel,grid,axvline,axhline,savefig,imshow,show,scatter,hist,bar,subplot,subplots,text,stem
from math import sqrt
import pandas as pd
from datetime import datetime,timedelta
from statsmodels.tsa.stattools import adfuller,grangercausalitytests
import statsmodels.api as sm

"""#Functions for cross-correlation analysis and construction of time series weighted graphs"""

#Calculate the correlation between two time series with a lag h (cross-correlation)
def cross_cor_h(x,y,h):
    #This function uses Python's cross-correlation implementation to optimize time
    ro_xy = sm.tsa.stattools.ccf(x, y, adjusted = False, nlags = h)
    return ro_xy

#h_sup indicates the range of days that will be taken into account to apply the lags of the time series
#h_inf is the minimum lag that will be considered
def selec_h(x,y,h_inf,h_sup):
    #Let's evaluate the correlation with values ​​of h from h_inf to h_sup
    #Correlations array from 0 to h_sup
    arr_corr = cross_cor_h(x,y,h_sup)
    #From the previous array we will only take the correlations with an h greater than or equal to h_inf
    corr_cut = arr_corr[h_inf:h_sup+1].tolist()
    corr_max = max(corr_cut)
    h_max = corr_cut.index(max(corr_cut))+h_inf
    return corr_max,h_max

#h_sup indicates the maximum number of days that will be considered as lag in the time series
#Function to detect synchronized time series
def mat_cor_h_sync(data_suav,umbral,h_inf=0,h_sup=15,diff_param=False):
    #If diff=True then each row of the data matrix will be differenced
    if diff_param==True:
        data = apply_along_axis(diff,1,data_suav)
    else:
        data = data_suav.copy()
    #Dimensions of data matrix
    n,m = shape(data)
    #Array of h values ​​with which the maximum cross-correlation was reached
    H_mat = zeros((n,n))
    #Maximum correlation matrix
    cor_mat = zeros((n,n))
    #Get correlation values ​​between each pair of time series
    for i in range(0,n):
        for j in range(i+1,n):
            cor_mat_i_j,H_mat_i_j = selec_h(data[i,:],data[j,:],h_inf,h_sup)
            cor_mat_j_i,H_mat_j_i = selec_h(data[j,:],data[i,:],h_inf,h_sup)
            #Determine the maximum between the components i,j and j,i of the correlation matrix
            if cor_mat_i_j >= cor_mat_j_i:
                #If the lag is no greater than the threshold then it is added to the output
                if H_mat_i_j <= umbral:
                    cor_mat[i,j],H_mat[i,j] = cor_mat_i_j,1
            else:
                #If the lag is no greater than the threshold then it is added to the output
                if H_mat_j_i <= umbral:
                    cor_mat[i,j],H_mat[i,j] = cor_mat_j_i,1
    #Make the matrices symmetric because in this case directions are not distinguished
    cor_mat_sim = cor_mat+cor_mat.T
    H_mat_sim = H_mat+H_mat.T
    return cor_mat_sim,H_mat_sim

#Function to find direction of time series interaction
def constr_cor_max_dir(data_suav,umbral,h_inf=0,h_sup=15,diff_param=False):
    #If diff=True then each row of the data matrix will be differenced
    if diff_param == True:
        data = apply_along_axis(diff,1,data_suav)
    else:
        data = data_suav.copy()
    n,m = shape(data)
    #Matrix of h values 
    H_mat = zeros((n,n))
    #Maximum correlations matrix
    cor_mat = zeros((n,n))
    #Adjacency matrix based on the direction of interaction
    A_mat = zeros((n,n))
    #Get correlation values ​​between each pair of time series
    for i in range(0,n):
        for j in range(i+1,n):
            cor_mat_i_j,H_mat_i_j = selec_h(data[i,:],data[j,:],h_inf,h_sup)
            cor_mat_j_i,H_mat_j_i = selec_h(data[j,:],data[i,:],h_inf,h_sup)
            #Determine the maximum between the components i,j and j,i of the correlation matrix
            if cor_mat_i_j >= cor_mat_j_i:
                #If the lag is no greater than the threshold then it is added to the output
                if H_mat_i_j >= umbral:
                    cor_mat[i,j],H_mat[i,j],A_mat[i,j] = cor_mat_i_j,H_mat_i_j,1
            else:
                #If the lag is no greater than the threshold then it is added to the output
                if H_mat_j_i >= umbral:
                    cor_mat[i,j],H_mat[j,i],A_mat[j,i] = cor_mat_j_i,H_mat_j_i,1
    #Convert the correlation matrix to symmetric
    cor_mat_sim = cor_mat+cor_mat.T
    return cor_mat_sim,H_mat,A_mat


###################################################################################################
###################################################################################################
"""#Functions to manage and transform data"""

#Convert rows of matrix into lists
def conv_to_list(matrix):
    n,m = shape(matrix)
    lists = []
    for i in range(0,n):
        row_act = matrix[i,:]
        lists.append(row_act.tolist())
    return lists

#n is total of municipalities or nodes
def fill_mat(lab,count,n):
    out = zeros((n))
    m = len(count)
    for i in range(0,m):
        out[lab[i]-1] = count[i]
    return out

#Increasing in days a value of date kind 
def increase_date(fecha_ini,dias):
    date = datetime(int(fecha_ini[0:4]),int(fecha_ini[5:7]),int(fecha_ini[8:10]))
    new_date = date+timedelta(days=dias)
    new_str = str(new_date)
    fecha_fin = new_str[0:10]
    return fecha_fin

#Centered moving averages of k time units
def moving_avg(x_fil,k=7):
    x = x_fil.tolist()
    n = len(x)-(k-1)
    suav = zeros((n))
    step = int((k-1)/2)
    for i in range(step,n+step):
        suav[i-step] = sum(x[i-step:i+step+1])/k
    out = asarray(suav)
    return out

#################################################################################################
#################################################################################################
"""#Functions for the creation and visual representation of graphs"""


#Transform symmetric adjacency matrix to edge list
def trans_adj_sim(A):
    n = len(A)
    list_edges = []
    #Obtain non-repeated edges according to lexicographic order
    for i in range(0,n):
        for j in range(i+1,n):
            if A[i,j]!=0:
                list_edges.append((i,j))
    return list_edges

#Transform non-symmetric adjacency matrix to edge list
def trans_adj(A):
    n = len(A)
    list_edges = []
    #Obtain non-repeated edges according to lexicographic order
    for i in range(0,n):
        for j in range(0,n):
            if A[i,j]!=0 and i!=j:
                list_edges.append((i,j))
    return list_edges

#Generate weight matrix from the correlation matrix and the adjacency matrix
def W_cor(mat_cor,A):
    n = len(mat_cor)
    W = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            if A[i,j]==1.0:
                W[i,j] = mat_cor[i,j]
    return W

#Generate list of weights corresponding to the list of edges
def W_list(list_edges,W,num_dig=3):
    #Lista de los pesos
    W_list = []
    #Número de aristas
    m = len(list_edges)
    for i in range(0,m):
        edge_act = list_edges[i]
        W_act = round(W[edge_act[0],edge_act[1]],num_dig)
        W_list.append(W_act)
    return W_list



"""#Implementation of GeNA"""

def delta(i,j):
    if i==j:
        out=1
    else:
        out=0
    return out

def create_com(com_prev,u,threshold=0):
    #The community list corresponds to the vector u
    #All the positive values ​​of u will be assigned to one community and all the negative ones to another
    n = len(com_prev)
    s = zeros((n,1))
    com1,com2 = [],[]
    u11,u22 = [],[]
    #The default threshold value for bipartitioning is zero
    for i in range(0,n):
        if u[i]>threshold:
            com1.append(com_prev[i])
            s[i,0] = 1
            u11.append(u[i])
        #The nodes corresponding to elements of u that are equal to zero are nodes with incident weights equal to zero, so they are discarded
        elif u[i]<threshold:
            com2.append(com_prev[i])
            s[i,0] = -1
            u22.append(u[i])
    return u11,u22,s,com1,com2

def bipartition(B,com):
    n_g = len(com)
    #Build the matrix corresponding to the community that will be split
    B_g = zeros((n_g,n_g))
    cf = 0
    for i in com:
        cc = 0
        for j in com:
            if i==j:
                sum_row = 0
                for k in com:
                    sum_row += B[i,k]
            B_g[cf,cc] = B[i,j]-delta(i,j)*sum_row
            cc += 1
        cf += 1
    #Obtain the eigenvalue and the leading eigenvector
    eigenval,eigenvec = eig(B_g)
    eigenvec = eigenvec[:,argsort(eigenval)]
    eigenval = eigenval[argsort(eigenval)]
    beta1 = eigenval[-1]
    u1 = eigenvec[:,-1]
    u11,u22,s,com1,com2 = create_com(com,u1)
    #It remains to divide the increment by 4m
    inc_mod = dot(dot(s.T,B_g),s)
    return beta1,u11,u22,inc_mod,com1,com2

#A is the adjacency or weight matrix
def GeNA(A):
    n = len(A)
    #Sum the weights of edges that affect each of the nodes
    weight_inc = apply_along_axis(sum,1,A)
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    m = apply_along_axis(sum,1,weight_row)/2
    mat_mod = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            mat_mod[i,j] = A[i,j]-(weight_inc[i]*weight_inc[j])/(2*m)
    #When applying the eig function, the result is a list in which the first element is the array with the eigenvalues
    #The second element is the matrix whose columns are the normalized eigenvectors
    #Get the eigenvalue and the leading eigenvector
    eigenval,eigenvec = eig(mat_mod)
    eigenvec = eigenvec[:,argsort(eigenval)]
    eigenval = eigenval[argsort(eigenval)]
    beta1 = eigenval[-1]
    u1 = eigenvec[:,-1]
    #Lists to store the list of the final partition and the final leading eigenvectors
    com_fin,u_fin = [],[]
    #Create list of labels of the nodes that make up the graph
    com = list(range(0,n))
    #Determine if the obtained partition is trivial
    if beta1<=0.000001:
        com_fin.append(com)
        u_fin.append(u1)
    else:
        #Create communities according to the sign
        u11,u22,s,com1,com2 = create_com(com,u1)
        com_res,com_res_act = [com1,com2],[com1,com2]
        #List of eigenvectors u that contains the belonging level of each node to its respective community
        u_res,u_act = [u11,u22],[u11,u22]
        n_com = len(com_res)
        while(n_com!=0):
            for j in range(0,n_com):
                beta1,u11,u22,inc_mod,com11,com22 = bipartition(mat_mod,com_res[j])
                #Stop dividing the community when the leading eigenvalue is zero or very close to zero
                #A leading eigenvalue close to zero can produce trivial bipartitions
                if beta1<=0.000001:
                    com_fin.append(com_res[j])
                    u_fin.append(u_res[j])
                elif len(com11)<=1 or len(com22)<=1:
                    #In case a trivial partition has been leaked, the empty community will not be included
                    if len(com11)>0:
                        com_fin.append(com11)
                        u_fin.append(u11)
                    if len(com22)>0:
                        com_fin.append(com22)
                        u_fin.append(u22)
                else:
                    com_res_act.append(com11)
                    com_res_act.append(com22)
                    u_act.append(u11)
                    u_act.append(u22)
                com_res_act.remove(com_res[j])
                u_act.remove(u_res[j])
            com_res = com_res_act.copy()
            u_res = u_act.copy()
            n_com = len(com_res)
    return u_fin,com_fin

#Obtain level of membership of each node to its community in the original partition
def level_membership_GeNA(u):
    n = len(u)
    belong = []
    for i in range(0,n):
        belong.append([])
        sum_act=sum(u[i])
        for x in u[i]:
            belong_act = belong[i]
            belong_act.append(x/sum_act)
    return belong

def comp(x,com):
    out = 0
    n = len(com)
    for i in range(0,n):
        if x==com[i]:
            out = 1
    return out

#Function to calculate the modularity of a partition
def modularity(W,partition):
    n = len(W)
    #Sum the weights of the edges that affect each of the nodes
    weight_inc = apply_along_axis(sum,1,W)
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    L = apply_along_axis(sum,1,weight_row)/2
    #Modularity matrix
    B = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            B[i,j] = W[i,j]-(weight_inc[i]*weight_inc[j])/(2*L[0])
    #Calculate the modularity perform the sum on the edges between nodes that belong to the same community
    m = len(partition)
    #Perform the sum in each community
    sum_com = zeros((1,m))
    #Generate arrays from which the necessary rows and columns will be removed
    mat_com = []
    for l in range(0,m):
        mat_com.append(B.copy())
    #Remove the values ​​of nodes that do not belong to the respective community
    for k in range(0,m):
        com_act = partition[k]
        mat_act = mat_com[k]
        for p in range(0,n):
            if comp(p,com_act)==0:
                mat_act[p,:] = zeros((n))
                mat_act[:,p] = zeros((n))
        row = zeros((1,n))
        row[0,:] = apply_along_axis(sum,1,mat_act)
        sum_com[0,k] = apply_along_axis(sum,1,row)
    suma = apply_along_axis(sum,1,sum_com)
    #Modularity value
    mod = suma[0]/(2*L[0])
    return mod


"""Color nodes of a graph according to their group"""

list_colors=['darkorange','royalblue','gold','magenta','green','hotpink','red','purple','yellow','cyan','brown','olive','navy','lime','silver',
             'gray','crimson','salmon','lawngreen','darkred','crimson','lightgreen','tan','indigo','violet','senna','black','orange']

def vertex_com(part,num_nodes):
    #Number of communities
    n = len(part)
    #Array with the community to which each node belongs
    out = zeros((num_nodes))
    for i in range(0,n):
        com_act = part[i]
        len_act = len(com_act)
        for j in range(0,len_act):
            out[com_act[j]] = i
    return out

def list_vertex_color(part,num_nodes):
    num_com = vertex_com(part,num_nodes)
    list_v_color = []
    print(num_com)
    for i in range(0,num_nodes):
        list_v_color.append(list_colors[int(num_com[i])])
    return list_v_color


"""#Granger causality test"""

def grangers_causation_matrix(data_matrix, variables_name, maxlag, test='ssr_chi2test'):
    n_data = len(data_matrix[0,:])
    num_var = len(variables_name)
    df = pd.DataFrame(zeros((num_var, num_var)), columns=variables_name, index=variables_name)
    for i in range(0,num_var):
        for j in range(num_var):
            sub_mat = zeros((n_data,2))
            sub_mat[:,0],sub_mat[:,1] = data_matrix[i,:],data_matrix[j,:]
            test_result = grangercausalitytests(sub_mat, maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = min(p_values)
            df.loc[df.columns[i], df.index[j]] = min_p_value
    df.columns = [var + '_x' for var in variables_name]
    df.index = [var + '_y' for var in variables_name]
    return df



"""Check if the time series is stationary"""

#Series are differenced by default
def verify_stationary(time_series,diff_param=True):
    #Determine if the time series is stationary through the ADF test
    if diff_param==True:
        #In this case the series is differenced
        dftest = adfuller(diff(time_series), autolag='AIC')
    else:
        dftest = adfuller(time_series, autolag='AIC')
    result = pd.Series(dftest[0:4], index=['Test Statistic','P-value','Lags Used','No of Observations'])
    for key,value in dftest[4].items():
        result['Critical Value (%s)'%key] = value
    return result