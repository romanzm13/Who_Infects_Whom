# -*- coding: utf-8 -*-
"""
@author: Román Zúñiga Macías
"""

#pip install xlrd==1.2.0

from numpy import count_nonzero,arange,array,dot,asarray,zeros,apply_along_axis,around,sort,shape,savetxt,array_equal,max,argmin,argmax,fill_diagonal,ones,argsort,std,diag,random,diff
from numpy.linalg import eig
from matplotlib.pyplot import plot,figure,title,legend,xlabel,ylabel,grid,axvline,axhline,savefig,imshow,show,scatter,hist,bar,subplot,subplots,text,stem,rcParams
from math import sqrt
import pandas as pd
from datetime import datetime
import igraph as ig
import matplotlib.patches as mpatches
import statsmodels.api as sm
from Analysis_Series_GeNA import moving_avg,increase_date,mat_cor_h_sync,GeNA,level_membership_GeNA,verify_stationary,modularity,trans_adj_sim,W_list,constr_cor_max_dir,trans_adj,list_vertex_color,grangers_causation_matrix

#################################################################################################
#################################################################################################

"""Function to group data by age"""

def group_by_age(age_group,days_range=184,date_0='2021-07-01'):
    date = date_0
    #We want the last age range to be greater than or equal to the penultimate age_group value
    age_group[-1] = 150
    #Number of groups
    gr = len(age_group)-1
    #Group number list
    num_group = []
    for k in range(0,gr):
        num_group.append(str(k))
    #Time series matrix
    mat_rang_per = zeros((gr,days_range))
    #Generate time series by range of days, counting cases per day within each range
    for i in range(0,days_range):
        df_day = df_per[df_per['FEC_INI_SIN']==date]
        df_day['EDAD'] = pd.cut(df_day['EDAD'],age_group,labels=num_group)
        df_day_count = df_day.groupby(by=['EDAD']).count()
        df_day_count1 = asarray(df_day_count).T
        mat_rang_per[:,i] = df_day_count1
        date_prev = date
        date = increase_date(date_prev,1)
    return mat_rang_per

"""#Group age data"""

def accum_data(data):
    #Data size
    n = len(data)
    #Accumulated data
    data_accum = zeros((n))
    data_accum[0] = data[0]
    for i in range(1,n):
        data_accum[i] = data_accum[i-1]+data[i]
    return data_accum

def n_perc(data_accum,n,equal,fact):
    perc = 1.0/n
    if equal:
        perc_v = arange(perc,1+perc,perc)
    else:
        perc_v = zeros((n))
        perc_act = 0
        #Odd case
        if n%(2)==1:
            for i in range(0,n-1):
                if i<(n-1)/2:
                    perc_v[i] = perc_act+perc+(-1)**i*perc/fact
                elif i==(n-1)/2:
                    perc_v[i] = perc_act+perc
                else:
                    perc_v[i] = perc_act+perc-(-1)**i*perc/fact
                perc_act = perc_v[i]
            perc_v[n-1] = 1.0
        #Even case
        else:
            for i in range(0,n-1):
                perc_v[i] = perc_act+perc+(-1)**i*perc/fact
                perc_act = perc_v[i]
            perc_v[n-1] = 1.0
    #Total population
    pop_tot = data_accum[-1]
    pop_perc_v = pop_tot*perc_v
    return pop_perc_v

def groups_perc(data,n,equal=True,fact=1):
    #Generate array of accumulated data
    data_accum = accum_data(data)
    #Counter of data that has already been evaluated
    count = 0
    pop_v = n_perc(data_accum,n,equal,fact)
    ind_g,sum_g=zeros((n)),zeros((n))
    for i in range(0,n):
          while data_accum[count]<pop_v[i]:
              sum_g[i] += data[count]
              count += 1
          ind_g[i] = count
    return ind_g,sum_g

#Function to create bins from a group partition
def bins_groups(part,groups_age):
    #Number of groups in the partition
    n = len(part)
    #Extract the first element of each group
    bins0 = []
    for i in range(0,n):
        com_act = part[i]
        print(com_act)
        com_act_ini = com_act[0]
        group_age_act = groups_age[com_act_ini]
        print(group_age_act)
        bins0.append(group_age_act)
        print(bins0)
    bins1 = bins0+[150]
    bins1.sort()
    print('Bins obtained:',bins1)
    return bins1


#List of colors to plot
list_colors=['darkorange','royalblue','gold','magenta','green','hotpink','red','purple','yellow','cyan','brown','olive','navy','lime','silver',
             'gray','crimson','salmon','lawngreen','darkred','crimson','lightgreen','tan','indigo','violet','senna','black','orange']
#################################################################################################

#Size of xticks and yticks in all the graphics
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

#################################################################################################
#################################################################################################


"""#Import data
"""

#Case records by municipality
df1 = pd.read_csv('DA_Radar_Confirmed_Cases.csv')
#Dataframe of cases registered in a certain period
#Start and end date of the period to be analyzed
date_ini,date_end = '2021-06-01','2022-02-28'
df_per_accum = df1[df1['FEC_INI_SIN']<=date_end]
df_per = df_per_accum[df_per_accum['FEC_INI_SIN']>=date_ini]
date_ini = datetime(int(date_ini[0:4]),int(date_ini[5:7]),int(date_ini[8:10]))
date_end = datetime(int(date_end[0:4]),int(date_end[5:7]),int(date_end[8:10]))
range_date = date_end-date_ini
range_days = int(range_date.days)

"""Data on inhabitants of Jalisco in 2020 by each age (from 0 years to 100 and more)"""

df_age = pd.read_excel("datos_edades_Jalisco_2020.xlsx",sheet_name="Hoja1")

df_age.info()

#Delete the record corresponding to "No especificado"
df_age.drop(101, axis=0, inplace=True)
#Save the Population by age and the label "Grupo de edad"
pop_per_age = df_age['Poblacion_total']
label_age = df_age['Grupo_edad']
print(label_age)

accum_age = accum_data(pop_per_age)

"""#Determine in which periods the time series will be analyzed"""

#Age ranges
group_unique = [0,130]
gr_unique = len(group_unique)-1
mat_rang_per = group_by_age(group_unique,days_range=range_days,date_0='2021-06-01')
#Smoothed time series matrix
range_days_smoot = range_days-6
mat_rang_smoot_unique = zeros((gr_unique,range_days_smoot))
for i in range(0,gr_unique):
    mat_rang_smoot_unique[i,:] = moving_avg(mat_rang_per[i,:])

print(mat_rang_smoot_unique[0,:])

#Range of days represented
days_time_serie = arange(0,range_days_smoot,1)
figure(figsize = (10,6))
grid()
title("Smoothized time series from June 1, 2021 to February 28, 2022")
xlabel("Day number")
ylabel("Number of cases")
#Plot the smoothed time series
plot(days_time_serie,mat_rang_smoot_unique[0,:],color='red',linewidth=0.5,linestyle="--",marker="o")


"""Set start and end dates for data pools"""

#Start and end date of the period to be analyzed
#The period consists of 9 months
date_ini,date_end = '2021-06-01','2022-02-28'
df_day_accum = df1[df1['FEC_INI_SIN']<=date_end]
df_day = df_day_accum[df_day_accum['FEC_INI_SIN']>=date_ini]
date_ini = datetime(int(date_ini[0:4]),int(date_ini[5:7]),int(date_ini[8:10]))
date_end = datetime(int(date_end[0:4]),int(date_end[5:7]),int(date_end[8:10]))
range_date = date_end-date_ini
range_days = int(range_date.days)
range_days_smoot = range_days-6

"""#Finding the initial age-equitable grouping that maximizes modularity, partitioning by synchronicity"""

#k values ​​to explore
k_min,k_max = 2,15
#Modularity of the partition obtained with each k equal age groups
mod_k = []
#Initial groups found for each k
groups_k = []
#Population of each group that makes up the initial grouping
pop_k = []
#Partition found for each k
part_k = []
#H_max and cor_max matrices
H_max_k,cor_max_k = [],[]
#Smoothed data matrix
M_smoot = []
#Leading eigenvector for the level of community membership
u_k = []
for k in range(k_min,k_max):
    groups_now,pop_now = groups_perc(pop_per_age,k)
    pop_k.append(pop_now)
    #Add 0 to the beginning of the list
    group_age_now = groups_now.tolist()
    group_age_now.insert(0,0)
    groups_k.append(group_age_now)
    gr_now = k
    mat_rang_per = group_by_age(group_age_now,days_range=range_days,date_0='2021-06-01')
    #Test time series smoothed using moving averages
    #Smoothed time series matrix
    mat_rang_smoot = zeros((gr_now,range_days_smoot))
    for i in range(0,gr_now):
        mat_rang_smoot[i,:] = moving_avg(mat_rang_per[i,:])
    M_smoot.append(mat_rang_smoot)
    W_cor_max_now,H_max_now = mat_cor_h_sync(mat_rang_smoot,umbral=1,h_inf=0,h_sup=7,diff_param=True)
    H_max_k.append(H_max_now)
    cor_max_k.append(W_cor_max_now)
    #Generate partition with GeNA algorithm
    u_nk_now,com_nk_now = GeNA(W_cor_max_now)
    part_k.append(com_nk_now)
    u_k.append(u_nk_now)
    mod_now = modularity(W_cor_max_now,com_nk_now)
    mod_k.append(mod_now)

"""Plot the modularity of the partition obtained with k initial age groups"""

figure(figsize=(8,5))
k_lab=arange(k_min,k_max,1)
plot(k_lab,mod_k,color='blue',linewidth=0.5,linestyle="--",marker="o")
title("Modularity of the partition obtained with k initial age groups")
xlabel("k")
ylabel("Modularity")
grid()
savefig('mod_groups_age_jal.png', dpi=350)

"""Modularity was maximized with k=8"""

group_age8 = groups_k[6]
gr8 = len(group_age8)-1
print("Age groups considered")
for i in range(0,gr8-1):
      print("Group "+str(i)+": "+str(group_age8[i])+"-"+str(group_age8[i+1]-1)+" years")
print("Group "+str(gr8-1)+": "+"Over "+str(group_age8[i+1])+" years")
print()
print("Partition found:")
part8 = part_k[6]
print(part8)

"""Plot the time series and check if they are stationary"""

#The number of rows of the matrix corresponds to the number of groups and the number of columns to the number of days in the period
mat_smoot_now = M_smoot[6]
n1,n2 = shape(mat_smoot_now)
#Range of days represented
days_time_serie = arange(0,n2,1)
figure(figsize=(10,6))
grid()
title("Smoothed Time Series by age group")
xlabel("Day number")
ylabel("Number of cases")
#Number of communities
n_com = len(part8)
patches = []
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      com_now = part8[k]
      len_now = len(com_now)
      for l in range(0,len_now):
            plot(days_time_serie,mat_smoot_now[com_now[l],:],linewidth=0.5,color=list_colors[k],linestyle="--",marker="o")
            #Determine if the current time series is stationary through the ADF test
            result_now = verify_stationary(mat_smoot_now[com_now[l],:])
            print("Statistical test for age group time series "+str(com_now[l]))
            print(result_now)
            print()
      patch_now = mpatches.Patch(color=list_colors[k], label='Community '+str(k))
      patches.append(patch_now)
legend(handles=patches,prop={'size':14})
savefig('time_series_group_jal.png', dpi=350)

"""Level of membership of the nodes to their community"""

lev_member = level_membership_GeNA(u_k[6])
#Histogram of level of membership for each community
fig = figure(figsize = (10,5))
subplot(1,2,1)
grid()
xlabel("Age group", fontsize=15)
#xlabel("Grupo")
bar(list(map(str,part8[0])),lev_member[0], color = list_colors[0])
subplot(1,2,2)
grid()
xlabel("Age group", fontsize=15)
bar(list(map(str,part8[1])),lev_member[1], color = list_colors[1])

st = fig.suptitle("(b) Level of membership of each group to its community", fontsize = 18)
st.set_y(0.95)
fig.subplots_adjust(top = 0.89)
savefig('level_membership_jal.png', dpi=350)

"""Population histogram by age group"""

figure(figsize=(10,6))
pop8 = pop_k[6]
n8 = len(pop8)
bar(arange(0,n8,1),pop8,color='blue')
#Add value labels to each bar
for i in range(0,n8):
    text(i-0.4,pop8[i],str(pop8[i]),color="k")
title("Population of Jalisco in 2020 by each age group")
xlabel("Group")
ylabel("Number of inhabitants")
grid()

"""Plot series normalizing them by the total number of infected"""

figure(figsize=(10,6))
grid()
title("(c) Normalized time series of COVID-19 cases by age group", fontsize=18)
xlabel("Day number", fontsize=15)
ylabel("Cases percentage", fontsize=15)

#Number of communities
n_com = len(part8)
patches = []
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      com_now = part8[k]
      len_now = len(com_now)
      for l in range(0,len_now):
            plot(days_time_serie,mat_smoot_now[com_now[l],:]/sum(mat_smoot_now[com_now[l],:]),linewidth=0.5,color=list_colors[k],linestyle="--",marker="o")
      patch_now = mpatches.Patch(color=list_colors[k], label='Community '+str(k))
      patches.append(patch_now)
legend(handles=patches,prop={'size':14})
savefig('series_age_norm_jal.png', dpi=350)

"""Representation of interactions through a graph

"""

edgelist8 = trans_adj_sim(H_max_k[6])
g8 = ig.Graph(edges=edgelist8, directed=False)
g8.es['width'] = 0.5
g8.es["label_size"] = 11
#Weight list
W_list8 = W_list(edgelist8,cor_max_k[6])
g8.es['weight'] = W_list8
fig,ax = subplots(figsize = (9,6))
list_v_colors = list_vertex_color(part8,8)

layout = g8.layout("star")
ig.plot(g8, target=ax, layout=layout, vertex_size = 30 , edge_width = g8.es['width'], edge_label = g8.es["weight"], vertex_label = range(gr8), vertex_color = list_v_colors)


"""Find the maximum cross-correlation between the communities found from synchronicity"""

#Sum the rows of the smoothed data matrix that correspond to groups of the same community
mat_smoot_com = zeros((n_com,range_days_smoot))
pop_com = zeros((n_com))
for i in range(0,n_com):
      com_now = part8[i]
      for j in range(0,len(com_now)):
          mat_smoot_com[i,:] = mat_smoot_com[i,:]+mat_smoot_now[com_now[j],:]
          pop_com[i] += pop8[com_now[j]]

"""Plot the time series corresponding to the two age communities found"""

figure(figsize = (10,6))
grid()

title("Smoothed time series by community")
xlabel("Day number")
ylabel("Number of cases")

#Number of communities
n_com = len(part8)
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      plot(days_time_serie,mat_smoot_com[k,:],linewidth=0.5,color=list_colors[k],linestyle="--",marker="o",label='Comunidad '+str(k))
      #Determine if the current time series is stationary through the ADF test
      result_act = verify_stationary(mat_smoot_com[k,:])
      print("Statistical test for community time series "+str(k))
      print(result_act)
      print()
legend(prop={'size':12})

"""
Series modified to be comparable by normalizing them by the total number of infected people
"""

sum_com0 = sum(mat_smoot_com[0,:])
sum_com1 = sum(mat_smoot_com[1,:])
figure(figsize = (10,6))
grid()
title("Normalized time series of COVID-19 by community", fontsize=18)
xlabel("Day number", fontsize=15)
ylabel("Cases percentage", fontsize=15)

#Plot the time series that corresponds to each group

plot(days_time_serie, mat_smoot_com[0,:]/sum_com0, linewidth = 0.5, color = list_colors[0], linestyle = "--", marker = "o", label = 'Community 0')
plot(days_time_serie, mat_smoot_com[1,:]/sum_com1, linewidth = 0.5, color = list_colors[1], linestyle = "--", marker = "o", label = 'Community 1')
legend(prop={'size':12})
savefig('series_com_norm_jal.png', dpi=350)

"""Representation of interactions through a graph"""

h_min,h_max = 4,8
W_cor_max_8_part,H_new_8_part,A_8_part = constr_cor_max_dir(mat_smoot_com,umbral=h_min,h_inf=h_min,h_sup=h_max,diff_param=True)
print(A_8_part)
print(H_new_8_part)
print(W_cor_max_8_part)

fig,ax = subplots(figsize = (8,5))
edgelist8_part = trans_adj(A_8_part.T)
g8_part = ig.Graph(edges=edgelist8_part, directed=True)
g8_part.es['width'] = 0.7
#Weight list
W_list8_part = W_list(edgelist8_part,W_cor_max_8_part.T)
g8_part.es['weight'] = W_list8_part
ig.plot(g8_part, target=ax, vertex_size=40, edge_width=g8_part.es['width'], edge_label=g8_part.es["weight"], vertex_label=range(n_com), vertex_color = list_colors[0:n_com])


"""**Check if this correlation is significant**

Analyze the autocorrelation of the series
"""

#Cross-correlation between time series with lags over a large interval
number_lags = 90
cross_corr_lag_neg = sm.tsa.stattools.ccf(diff(mat_smoot_com[0,:]), diff(mat_smoot_com[1,:]), adjusted = False, nlags = number_lags+1)
cross_corr_lag_pos = sm.tsa.stattools.ccf(diff(mat_smoot_com[1,:]), diff(mat_smoot_com[0,:]), adjusted = False, nlags = number_lags+1)
cross_corr = zeros((2*number_lags+1))
cross_corr[0:number_lags+1] = cross_corr_lag_neg[::-1]
cross_corr[number_lags+1:2*number_lags+1] = cross_corr_lag_pos[1:]
figure(figsize=(10,6))
xlabel("Lag")
ylabel("Correlation")
title("Sample cross correlation function with lags in [-"+str(number_lags)+","+str(number_lags)+"]")
axhline(y = max(cross_corr), linestyle="--", color = 'red')
grid()
stem(range(-number_lags,number_lags+1),cross_corr)

"""Compare its value with $\frac{1.96}{\sqrt{n}}$ to reject the hypothesis that it is white noise and therefore does not follow a distribution $N(0,σ)$"""

p = 1.96/sqrt(len(diff(mat_smoot_com[0,:])))
print("Since "+str(p)+" < "+str(max(cross_corr))+", then the correlation represented in the graph is significant")

"""Apply Granger causality test"""

labels_com = []
for i in range(0,n_com):
    labels_com.append("Community_"+str(i))
print(grangers_causation_matrix(mat_smoot_com, labels_com, maxlag=list(arange(h_min,h_max+1,1))))