# -*- coding: utf-8 -*-
"""
@author: Román Zúñiga Macías
"""

#pip install xlrd==1.2.0

from numpy import count_nonzero,arange,array,dot,asarray,zeros,apply_along_axis,around,sort,shape,savetxt,array_equal,max,argmin,argmax,fill_diagonal,ones,argsort,std,diag,random,diff
from numpy.linalg import eig
from matplotlib.pyplot import plot,figure,title,legend,xlabel,ylabel,grid,axvline,savefig,imshow,show,scatter,hist,bar,subplot,subplots,text,axhline,stem,rcParams
from math import sqrt
import pandas as pd
from datetime import datetime
import matplotlib.patches as mpatches
import igraph as ig
import statsmodels.api as sm
from Analysis_Series_GeNA import moving_avg,increase_date,mat_cor_h_sync,GeNA,level_membership_GeNA,verify_stationary,modularity,trans_adj_sim,W_list,constr_cor_max_dir,trans_adj,list_vertex_color,grangers_causation_matrix

#################################################################################################
#################################################################################################

#List of colors to plot
list_colors=['darkorange','royalblue','gold','magenta','green','hotpink','red','purple','yellow','cyan','brown','olive','navy','lime','silver',
             'gray','crimson','salmon','lawngreen','darkred','crimson','lightgreen','tan','indigo','violet','senna','black','orange']
#################################################################################################

#Size of xticks and yticks in all the graphics
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

"""#Import data
"""
"""Incidence data"""

#Records of influenza cases per week in the United States from 2021 to 2024
df0 = pd.read_csv('AgeViewByWeek_Feb4.csv')
df0.info()
#We will only keep the columns 'Year', 'Week', 'Age Group' and 'A (H1N1)pdm09'
df1 = df0.drop(['A (H1)','A (Unable to Subtype)','A (H3)','A (Subtyping not Performed)','B (Victoria Lineage)','B (Yamagata Lineage)','B (Lineage Unspecified)','H3N2v'],axis = 1)
#Age groups recorded in the data set
groups_age = df1[' Age Group'].unique()

#Sort records according to date (year and week number)
df2 = df1.sort_values(by = ["Year"," Week"], ascending = True, axis = 0)
#We will keep only the records of 2022, 2023 and 2024
df3 = df2[((df2["Year"] == 2021) & (df2[" Week"] >= 48)) | (df2["Year"] == 2022) | (df2["Year"] == 2023) | ((df2["Year"] == 2024) & (df2[" Week"] <= 8))]
#Generate dataframes according to ages
df_0_4 = df3[df3[' Age Group'] =='0-4 yr']
df_5_24 = df3[df3[' Age Group'] =='5-24 yr']
df_25_64 = df3[df3[' Age Group'] =='25-64 yr']
df_65 = df3[df3[' Age Group'] =='65+ yr']
#Convert case count time series to arrays
arr_0_4 = asarray(df_0_4['A (H1N1)pdm09'])
arr_5_24 = asarray(df_5_24['A (H1N1)pdm09'])
arr_25_64 = asarray(df_25_64['A (H1N1)pdm09'])
arr_65 = asarray(df_65['A (H1N1)pdm09'])
#Build matrix with time series per row
#Number of weeks
n_weeks = len(arr_0_4)
mat_age = zeros((4,n_weeks))
mat_age[0,:] = arr_0_4
mat_age[1,:] = arr_5_24
mat_age[2,:] = arr_25_64
mat_age[3,:] = arr_65

n_weeks_suav = n_weeks-4
#Smoothed time series matrix
mat_smoot = zeros((4,n_weeks_suav))
for i in range(0,4):
    mat_smoot[i,:] = moving_avg(mat_age[i,:],k=5)

#Plot smoothed time series with 5-week moving averages
#Two of the series are not stationary, so the four had to be differenced
figure(figsize = (10,8))
for i in range(0,4):
    plot(mat_smoot[i,:], label = groups_age[i], linestyle = "--", marker = "o")
    #Determine if the current time series is stationary through the ADF test
    result_now = verify_stationary(mat_smoot[i,:])
    print("Statistical test for age group time series "+str(groups_age[i]))
    print(result_now)
    print()
grid()
legend()
xlabel("Week",fontsize=15)
ylabel("Number of cases",fontsize=15)
title("Cases of influenza A (H1N1) per week from September 2022 to January 2024 in public laboratories of USA",fontsize=16)

"""**By differencing the smoothed series, they all become stationary**

#Finding the partition of time series by synchronicity
"""

#Test time series smoothed with moving averages
W_cor_max_now,H_max_now = mat_cor_h_sync(mat_smoot,umbral=1,h_inf=0,h_sup=4,diff_param=True)
#Generate partition with GeNA algorithm
u_nk_now,com_nk_now = GeNA(W_cor_max_now)
mod_now = modularity(W_cor_max_now,com_nk_now)
print("The partition found is:")
print(com_nk_now)
print("The modularity of this partition is:")
print(mod_now)

"""Plot time series grouped by community"""

#The number of rows of the matrix corresponds to the number of groups and the number of columns to the number of weeks in the period
n1,n2 = shape(mat_smoot)
#Range of weeks represented
weeks_time_serie = arange(0,n2,1)
figure(figsize=(10,6))
grid()
title("Smoothed time series by age group")
xlabel("Week number")
ylabel("Number of cases")

#Number of communities
n_com = len(com_nk_now)
patches = []
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      com_now = com_nk_now[k]
      len_now = len(com_now)
      for l in range(0,len_now):
            plot(weeks_time_serie,mat_smoot[com_now[l],:],linewidth=0.5,color=list_colors[k],linestyle="--",marker="o")
      patch_now = mpatches.Patch(color=list_colors[k], label='Community '+str(k))
      patches.append(patch_now)
legend(handles=patches,prop={'size':14})
savefig('time_series_group_USA.png', dpi=300)

"""Level of membership of the nodes to the community"""

lev_member = level_membership_GeNA(u_nk_now)
print(lev_member)
#Histogram of level of membership for each community
fig = figure(figsize = (10,5))
subplot(1,2,1)
grid()
xlabel("Age group",fontsize=15)
bar(list(map(str,com_nk_now[0])),lev_member[0], color = list_colors[0])
subplot(1,2,2)
grid()
xlabel("Age group",fontsize=15)
bar(list(map(str,com_nk_now[1])),lev_member[1], color = list_colors[1])

st = fig.suptitle("(b) Level of membership of each group to its community", fontsize=18 )#"x-large")
savefig('level_membership_USA.png', dpi=300)

"""Plot normalized series by total infected"""

figure(figsize=(10,6))
grid()
title("(c) Normalized time series of influenza A (H1N1) cases by age group", fontsize=18)
xlabel("Week number", fontsize=15)
ylabel("Percentage of infected", fontsize=15)

#Number of communities
patches = []
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      com_now = com_nk_now[k]
      len_now = len(com_now)
      for l in range(0,len_now):
            plot(weeks_time_serie,mat_smoot[com_now[l],:]/sum(mat_smoot[com_now[l],:]),linewidth=0.5,color=list_colors[k],linestyle="--",marker="o")
      patch_now = mpatches.Patch(color=list_colors[k], label='Community '+str(k))
      patches.append(patch_now)
legend(handles=patches,prop={'size':14})
savefig('series_age_norm_USA.png', dpi=300)

"""Representation of interactions between nodes through a graph"""

edgelist = trans_adj_sim(H_max_now)
g = ig.Graph(edges = edgelist, directed = False)
g.es['width'] = 0.5
#Weight list
W_list1 = W_list(edgelist, W_cor_max_now)
g.es['weight'] = W_list1
fig,ax = subplots(figsize = (5,4))
list_v_colors = list_vertex_color(com_nk_now, 4)
ig.plot(g, vertex_size = 30, target = ax, edge_width = g.es['width'], edge_label = g.es["weight"], vertex_label = range(4), vertex_color = list_v_colors)

"""Find the maximum cross-correlation between the communities found from synchronicity"""

#Sum the rows of the smoothed data matrix that correspond to groups from the same community
mat_smoot_com = zeros((n_com,n_weeks_suav))
for i in range(0,n_com):
      com_now = com_nk_now[i]
      for j in range(0,len(com_now)):
          mat_smoot_com[i,:] = mat_smoot_com[i,:]+mat_smoot[com_now[j],:]

"""Plot the time series corresponding to the two age communities found"""

figure(figsize=(10,6))
grid()
title("Smoothed time series by community")
xlabel("Week number")
ylabel("Number of cases")
#Number of communities
n_com = len(com_nk_now)
#Plot the time series that corresponds to each group
for k in range(0,n_com):
      plot(weeks_time_serie,mat_smoot_com[k,:],linewidth=0.5,color=list_colors[k],linestyle="--",marker="o",label='Community '+str(k))
      #Determine if the current time series is stationary through the ADF test
      result_now = verify_stationary(mat_smoot_com[k,:])
      print("Statistical test for community time series "+str(k))
      print(result_now)
      print()
legend(prop={'size':14})

"""**The series are also differenced by communities so that they are stationary**

Plot series normalizing them by the total number of infected
"""

sum_com0 = sum(mat_smoot_com[0,:])
sum_com1 = sum(mat_smoot_com[1,:])
figure(figsize=(10,6))
grid()
title("Normalized time series of influenza A (H1N1) cases by community", fontsize=18)
xlabel("Week number", fontsize=15)
ylabel("Cases percentage", fontsize=15)

#Plot the time series that corresponds to each group
plot(weeks_time_serie, mat_smoot_com[0,:]/sum_com0, linewidth = 0.5, color = list_colors[0], linestyle = "--", marker = "o", label = 'Community 0')
plot(weeks_time_serie, mat_smoot_com[1,:]/sum_com1, linewidth = 0.5, color = list_colors[1], linestyle = "--", marker = "o", label = 'Community 1')
legend(prop={'size':14})
savefig('series_com_norm_USA.png', dpi=300)

"""Representation of interactions between communities through a graph"""

h_min,h_max = 2,4
W_cor_max_part,H_new_part,A_part = constr_cor_max_dir(mat_smoot_com,umbral=h_min,h_inf=h_min,h_sup=h_max,diff_param=True)
print(A_part)
print(H_new_part)
print(W_cor_max_part)

fig,ax = subplots(figsize = (6,4))
edgelist_part=trans_adj(A_part.T)
g_part = ig.Graph(edges=edgelist_part, directed=True)
g_part.es['width'] = 0.7
#Weight list
W_list_part=W_list(edgelist_part,W_cor_max_part)
g_part.es['weight'] = W_list_part
ig.plot(g_part, target=ax, vertex_size=40, edge_width=g_part.es['width'], edge_label=g_part.es["weight"], vertex_label=range(n_com), vertex_color = list_colors[0:n_com])

"""
**Check if this correlation is significant**

Analyze the autocorrelation of the series
"""

#Cross-correlation between time series with lags over a large interval
number_lags = 20
cross_corr_lag_neg = sm.tsa.stattools.ccf(diff(mat_smoot_com[0,:]),diff(mat_smoot_com[1,:]), adjusted = False, nlags = number_lags+1)
cross_corr_lag_pos = sm.tsa.stattools.ccf(diff(mat_smoot_com[1,:]),diff(mat_smoot_com[0,:]), adjusted = False, nlags = number_lags+1)
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
print("Since "+str(p)+" < 0.89761231, then the correlation represented in the graph is significant")

"""Apply Granger causality test"""

labels_com = []
for i in range(0,n_com):
    labels_com.append("Community_"+str(i))
print(grangers_causation_matrix(mat_smoot_com, labels_com, maxlag=list(arange(h_min,h_max+1,1))))