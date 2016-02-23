# -*- coding: utf-8 -*-

"""

	Amazon Collider Project - May 2015
	Authors: 
		Zied Alaoui
		Nathan Berrebi
		Valentin Marek
		Kevin Olivier

"""

from scipy.special import erf, erfc
import scipy.optimize as optimize
from math import sqrt
from math import pi
from math import exp

import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



################################################################################
# COST FUNCTIONS
################################################################################

Cg = 190
Co = 77
Co_prime = 184
Cs = 2352
truck_cap = 425

# Inital model
def g1(xg, xo, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 1 for a daily demand """
	d = np.random.normal(10000, 2000)
	if d <= truck_cap*xg:
		return cg*xg + co*xo
	elif d <= truck_cap*(xg + xo):
		return cg*xg + co*xo + co_p*(d/truck_cap - xg)
	else:
		return cg*xg + co*xo + co_p*xo + cs*(d/truck_cap - xg - xo)

# Model with alpha - use UPS to serve up to .95% of daily demand
def g2(xg, xo, alpha=.95, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 2 for a daily demand """
	d = np.random.normal(10000, 2000)
	if d <= truck_cap*xg:
		return cg*xg + co*xo
	elif d <= truck_cap*(xg + xo):
		return cg*xg + co*xo + co_p*(d/truck_cap - xg)
	else:
		if alpha*d <= truck_cap*(xg + xo):
			return cg*xg + co*xo + co_p*xo
		else:
			return cg*xg + co*xo + co_p*xo + cs*(alpha*d/truck_cap - xg - xo)

################################################################################
# DEPRECATED
# Model with alpha - use UPS AND xo to serve up to .95% of daily demand
# def g3(xg, xo, alpha=.95, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
# 	d = np.random.normal(10000, 2000)
# 	if d <= truck_cap*xg:
# 		return cg*xg + co*xo
# 	elif d <= truck_cap*(xg + xo):
# 		return cg*xg + co*xo + co_p*(alpha*d/truck_cap - xg)
# 	else:
# 		return cg*xg + co*xo + co_p*xo + cs*(alpha*d/truck_cap - xg - xo)
################################################################################


# Model with alpha - serve 100% unless d >= F-1(alpha) - (min: xg = 15, xo = 15)
def g4(xg, xo, alpha=.95, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 3 for a daily demand """
	z_alpha = scipy.stats.norm.ppf(alpha, 10000, 2000)
	d = np.random.normal(10000, 2000)
	if d <= z_alpha:
		if d <= truck_cap*xg:
			return cg*xg + co*xo
		elif d <= truck_cap*(xg + xo):
			return cg*xg + co*xo + co_p*(d/truck_cap - xg)
		else:
			return cg*xg + co*xo + co_p*xo + cs*(d/truck_cap - xg - xo)
	else:
		return cg*xg + co*xo + co_p*xo + cs*(z_alpha/truck_cap - xg -xo)




################################################################################
# COST FUNCTIONS - WEEKLY VERSION
################################################################################

Cg = 190
Co = 77
Co_prime = 184
Cs = 2352
truck_cap = 425

# Inital model (min: xg = 15, xo = 13)
def g1_horizon(xg, xo, horizon=7, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 1 on a weekly horizon """
	day_number = 1
	cost = 0
	while day_number <= horizon:
		if day_number % 6 == 0 or day_number % 7 == 0:
			d = np.random.normal(.8*10000, .8*2000)
		else:
			d = np.random.normal(1.2*10000, 1.2*2000)

		if d <= truck_cap*xg:
			cost += cg*xg + co*xo
		elif d <= truck_cap*(xg + xo):
			cost += cg*xg + co*xo + co_p*(d/truck_cap - xg)
		else:
			cost += cg*xg + co*xo + co_p*xo + cs*(d/truck_cap - xg - xo)

		day_number += 1
	return cost

# Model with alpha - use UPS to serve up to .95% of daily demand
def g2_horizon(xg, xo, alpha=.95, horizon=7, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 2 on a weekly horizon """
	day_number = 1
	cost = 0
	while day_number <= horizon:
		if day_number % 6 == 0 or day_number % 7 == 0:
			d = np.random.normal(.8*10000, .8*2000)
		else:
			d = np.random.normal(1.2*10000, 1.2*2000)

		if d <= truck_cap*xg:
			cost += cg*xg + co*xo
		elif d <= truck_cap*(xg + xo):
			cost += cg*xg + co*xo + co_p*(d/truck_cap - xg)
		else:
			if alpha*d <= truck_cap*(xg + xo):
				cost += cg*xg + co*xo + co_p*xo
			else:
				cost += cg*xg + co*xo + co_p*xo + cs*(alpha*d/truck_cap - xg - xo)

		day_number += 1
	return cost


# Model with alpha - serve 100% unless d >= F-1(alpha) - (min: xg = 15, xo = 15)
def g4_horizon(xg, xo, alpha=.95, horizon=7, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap): 
	""" Compute total cost for Model 3 on a weekly horizon """
	if alpha != .95:
		z_alpha_week = scipy.stats.norm.ppf(alpha, 1.2*10000, 1.2*2000)
		z_alpha_weekend = scipy.stats.norm.ppf(alpha, .8*10000, .8*2000)
	else:
		z_alpha_week = 15948.507
		z_alpha_weekend = 10632.338
	cost = 0
	day_number = 1
	while day_number <= horizon:
		if day_number % 6 == 0 or day_number % 7 == 0:
			d = np.random.normal(.8*10000, .8*2000)
			if d <= z_alpha_weekend:
				if d <= truck_cap*xg:
					cost += cg*xg + co*xo
				elif d <= truck_cap*(xg + xo):
					cost += cg*xg + co*xo + co_p*(d/truck_cap - xg)
				else:
					cost += cg*xg + co*xo + co_p*xo + cs*(d/truck_cap - xg - xo)
			else:
				cost += cg*xg + co*xo + co_p*xo + cs*(z_alpha_weekend/truck_cap - xg -xo)
		else:
			d = np.random.normal(1.2*10000, 1.2*2000)
			if d <= z_alpha_week:
				if d <= truck_cap*xg:
					cost += cg*xg + co*xo
				elif d <= truck_cap*(xg + xo):
					cost += cg*xg + co*xo + co_p*(d/truck_cap - xg)
				else:
					cost += cg*xg + co*xo + co_p*xo + cs*(d/truck_cap - xg - xo)
			else:
				cost += cg*xg + co*xo + co_p*xo + cs*(z_alpha_week/truck_cap - xg - xo)

		day_number += 1
	return cost


# Model with beta - (min: xg = 15, xo = 12)
def g5(xg, xo, threshold=500., horizon=7, cg=Cg, co=Co, co_p=Co_prime, cs=Cs, truck_cap=truck_cap):
	""" Compute total cost for Model 4 on a weekly horizon """
	cost = 0
	overflow = 0
	day_number = 1
	for day in range(horizon):
		if day_number % 6 == 0 or day_number % 7 == 0:
			d = np.random.normal(.8*10000, .8*2000)
		else:
			d = np.random.normal(1.2*10000, 1.2*2000)

		if d + overflow <= truck_cap*xg:
			cost += cg*xg + co*xo
			overflow = 0
		elif d + overflow <= truck_cap*(xg + xo):
			cost += cg*xg + co*xo + co_p*((d + overflow)/truck_cap - xg)
			overflow = 0
		elif d + overflow <= truck_cap*(xg + xo) + threshold:
			cost += cg*xg + co*xo + co_p*((d + overflow)/truck_cap - xg)
			overflow = overflow + d - truck_cap*(xg + xo)
		else:
			cost += cg*xg + co*xo + co_p*xo + cs*((d + overflow)/truck_cap - xg - xo - threshold/truck_cap)
			overflow = threshold
		
		day_number += 1
	return cost


################################################################################
# SIMULATION
################################################################################

# This simulation will return the tables of the total cost for different values 
# of Xg and Xo, and the minimum will be inside.
# Of course, it took us a larger number of trials to pinpoint the optimal values
set_xg = range(20, 26)
set_xo = range(10, 16)
set_g1 = []
set_g2 = []
set_g4 = []
set_g5 = []
for xg in set_xg:
	set_g1 += [ [] ]
	set_g2 += [ [] ]
	set_g4 += [ [] ]
	set_g5 += [ [] ]
	for xo in set_xo:
		print 'xg = {0}, xo = {1}'.format(xg,xo)
		set_g1[-1] += [ np.mean([ g1_horizon(xg,xo) for i in range(100000) ]) ]
		set_g2[-1] += [ np.mean([ g2_horizon(xg,xo) for i in range(100000) ]) ]
		if truck_cap*(xg + xo) >= 15948.507: 	# F-1(.95)
			set_g4[-1] += [ np.mean([ g4_horizon(xg,xo) for i in range(100000) ]) ]
		set_g5[-1] += [ np.mean([ g5(xg,xo) for i in range(100000) ]) ]
		print 'COST WITH INITIAL MODEL: {0}'.format(set_g1[-1][-1])
		print 'COST WITH ALPHA MODEL: {0}'.format(set_g2[-1][-1])
		print 'COST WITH ALPHA MODEL #2: {0}'.format(set_g4[-1][-1])
		print 'COST WITH BETA MODEL: {0}'.format(set_g5[-1][-1])
		print ''
set_g1 = np.array(set_g1)
set_g2 = np.array(set_g2)
set_g4 = np.array(set_g4)
set_g5 = np.array(set_g5)

xg_min = [ np.argmin(s)/len(set_xg) for s in [set_g1, set_g2, set_g4, set_g5] ]
xo_min = [ np.argmin(s[xg_min[i],:]) for i, s in enumerate([set_g1, set_g2, set_g4, set_g5]) ]
models = ['model1', 'model2', 'model4', 'model5']

sim_results = pd.DataFrame({'model': models, 'xg': xg_min, 'xo': xo_min})



################################################################################
# SENSITIVITY ANALYSIS
################################################################################

# Sensitivity Analysis - alpha for fixed xg, xo
set_alpha = [ float(i)/100 for i in range(70, 100) ]
sensitivity_alpha_fixedh = pd.DataFrame({'alpha': set_alpha})
g_min = []
xg = 24
xo = 13
for a in set_alpha:
	print 'APLHA = {0}'.format(a)
	g_min += [ np.mean([ g4_horizon(xg, xo, alpha=a) for i in range(100000) ]) ]
sensitivity_alpha_fixedh['cost'] = g_min

plt.figure()
plt.plot(sensitivity_alpha_fixedh['alpha'], sensitivity_alpha_fixedh['cost'], label='Model 3')
plt.plot(np.arange(.65, 1, .01), [45114]*len(np.arange(.65, 1, .01)), color="black", label='Model 1 Baseline')
plt.xlabel('alpha')
plt.ylabel('cost')
plt.title('Model 3 - Cost vs. Alpha')
plt.legend(loc='bottom right')
plt.ylim([0, 50000])
plt.show()


# Sensitivity Analysis - threshold
set_thr = [50] + range(100, 2001, 100)
sensitivity_threshold = pd.DataFrame({'threshold': set_thr})
g_min = []
xg = 23
xo = 14
for thr in set_thr:
	print 'THRESHOLD = {0}'.format(thr)
	g_min += [ np.mean([ g5(xg, xo, threshold=float(thr)) for i in range(100000) ]) ]
sensitivity_threshold['cost'] = g_min

plt.figure()
plt.plot(sensitivity_threshold['threshold'], sensitivity_threshold['cost'], label='Model 4')
plt.plot(np.arange(2050), [45114]*2050, color="black", label='Model 1 Baseline')
plt.xlabel('Threshold')
plt.ylabel('Cost')
plt.title('Model 4 - Cost vs. Threshold')
plt.legend(loc='upper right')
plt.ylim([43500, 45500])
plt.xlim([0, 2050])
plt.show()



################################################################################
# BONUS - APPENDIX B
################################################################################

# Analysis of the robustness of the simulation method
cost_sim1 = []
cost_sim2 = []
cost_sim4 = []
cost_sim5 = []
for i in range(1000):
	print 'SIMULATION # {0}'.format(i+1)
	cost_sim1 += [ np.mean([ g1_horizon(23,14) for i in range(100000) ]) ]
	cost_sim2 += [ np.mean([ g2_horizon(23,12) for i in range(100000) ]) ]
	cost_sim4 += [ np.mean([ g4_horizon(24,13) for i in range(100000) ]) ]
	cost_sim5 += [ np.mean([ g5(23,14) for i in range(100000) ]) ]
cost_sim = pd.DataFrame({'model1': cost_sim1, 'model2': cost_sim2, 'model4': cost_sim4, 'model5': cost_sim5})



def n_spot(n_sim=100000, horizon=7, truck_cap=truck_cap):
	""" Returns the number of spot vehicles used for all models over 
	n_sim * horizon simulations """
	spot_used = [0]*4
	for i in range(n_sim):
		day_number = 1
		while day_number <= horizon:
			if day_number % 6 == 0 or day_number % 7 == 0:
				d = np.random.normal(.8*10000, .8*2000)
			else:
				d = np.random.normal(1.2*10000, 1.2*2000)

			if d >= truck_cap*(23 + 14):
				spot_used[0] += 1
			if .95*d >= truck_cap*(23 + 12):
				spot_used[1] += 1
			if d >= truck_cap*(24 + 13):
				spot_used[2] += 1
			if  d + 500 >= truck_cap*(23 + 14):
				spot_used[3] += 1

			day_number += 1

	return spot_used

sim_spot = n_spot()







