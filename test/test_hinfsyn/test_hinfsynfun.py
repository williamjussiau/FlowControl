# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:15:34 2023

@author: wjussiau
"""

import control as ct

# Warning: compared to example on python-control.readthedocs, we need tf2io instead of tf


# Unstable first order SISO system
G = ct.tf2io([1], [1,-1], inputs=['u'], outputs=['y'])
max(G.pole()) < 0

# Create partitioned system with trivial unity systems
P11 = ct.tf2io([0], [1], inputs=['w'], outputs=['z'])
P12 = ct.tf2io([1], [1], inputs=['u'], outputs=['z'])
P21 = ct.tf2io([1], [1], inputs=['w'], outputs=['y'])
P22 = G
P = ct.interconnect([P11, P12, P21, P22], inplist=['w', 'u'], outlist=['z', 'y'])

# Synthesize Hinf optimal stabilizing controller
K, CL, gam, rcond = ct.hinfsyn(P, nmeas=1, ncon=1)
T = ct.feedback(G, K, sign=1)
max(T.pole()) < 0



# Mixed-sensitivity Hinf synthesis
# Weights on S, KS, T
#[z] = [p11 p12] [w]
#[y]   [p21   g] [u]
# cl = w>z with u=-ky
Kmx, CLmx, infomx = ct.mixsyn(G, ct.tf([0],[1]), ct.tf([1],[1]), ct.tf([1],[1]))

