#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

# Modified for TUDelft course CS4010 by Canmanie T. Ponnambalam, September 2018

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
from random import randint

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def create_qtable(num_states, num_actions):
    #qtable = np.zeros((num_states, num_actions), dtype =int)
    #NOT SURE HOW EXACTLY WE NEED TO INITIALIZE THIS
    qtable= 10 * np.random.random_sample((num_states, num_actions))
    return qtable

def calc_density(num_halt):
    if num_halt < 2: #low density
        density = 0
    elif num_halt < 4: #medium density
        density = 1
    else:
        density = 2

    return density

def get_state():
    state = 0
    #density = 0
    num_detectors = 2
    num_phases =2

    halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0") #number of halting cars on wA0
    halt_nA0 = traci.lanearea.getLastStepHaltingNumber("nA0")
    halt_eA0 = traci.lanearea.getLastStepHaltingNumber("eA0") #number of halting cars on wA0
    halt_sA0 = traci.lanearea.getLastStepHaltingNumber("sA0")

    #density_wA0 = calc_density(halt_wA0)
    #density_nA0 = calc_density(halt_nA0)
    
    ### Below is two different ways of computing the density
    ### 1
    #density_horiz = calc_density(halt_wA0) + calc_density(halt_eA0)
    #density_vert = calc_density(halt_nA0) + calc_density(halt_sA0)
    
    ### 2 
    density_horiz = calc_density(halt_wA0 + halt_eA0)
    density_vert = calc_density(halt_nA0 + halt_sA0)

    phase_A = traci.trafficlight.getPhase("A") #phase of the traffic light: 0 or 2

    #THIS IS A VERY STUPID WAY OF DEFINING STATES
    #FIND SOMETHING BETTER 
    states = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], [0, 0, 2], [0, 1, 2], [0, 2, 2], [1, 0, 2], [1, 1, 2], [1, 2, 2], [2, 0, 2], [2, 1, 2], [2, 2, 2]])
    state_values = np.array([density_horiz, density_vert, phase_A])
    state = np.where(np.all(states==state_values, axis=1))[0][0]
    return state
    
def choose_action(state, qtable, epsilon): 
    chance = np.random.random()
    
    if epsilon <= chance:
        #print("IF")
        action = np.argmax(qtable[state,:]) #returns the action with the max value at current state
    else:
        #print("ELSE")
        action = random.randint(0, np.size(qtable, 1)-1)

    #action = (randint(0, 1))
    return action
    
    
def calc_reward():
    halt_horiz = traci.lanearea.getLastStepHaltingNumber("wA0") + traci.lanearea.getLastStepHaltingNumber("eA0")
    halt_vert = traci.lanearea.getLastStepHaltingNumber("nA0") + traci.lanearea.getLastStepHaltingNumber("sA0")
    #halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0") #number of halting cars on wA0
    #reward = -1*halt_wA0 #this is gonna be total normally
    halt_total = halt_horiz + halt_vert
    reward = -1*halt_total
    return reward

def update_table(qtable, reward, state, action, alpha, gamma, next_state): #NOT SURE ABOUT THE Q-FUNCTION
    next_action = np.argmax(qtable[next_state,:])
    q = (1-alpha)*qtable[state,action] + alpha*(reward+gamma*(qtable[next_state][next_action]))
    qtable[state][action] = q
    print(q)
    print(qtable)
    return qtable


def check_goal(): #NEED TO IMPLEMENT THIS TO END TRAINING 
    return true

def generate_routefile(N):
    random.seed()  # make tests reproducible by random.seed(some_number)
    
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeCar" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

        <route id="NS" edges="n2A A2s" />
        <route id="WE" edges="w2A A2e" />
        <route id="SN" edges="s2A A2n" />
        <route id="EW" edges="e2A A2w" />""", file=routes)
        
        vehNr = 0
        
        # demand in each direction per time step
        p_NS = 1. / 10
        p_SN = 1. / 20
    	p_WE = 3. / 20
    	p_EW = 3. / 20
        
        for i in range(N):
        	if random.uniform(0, 1) < p_NS:
        		print('    <vehicle id="car_%i" type="typeCar" route="NS" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1
        		
        	if random.uniform(0, 1) < p_SN:
        		print('    <vehicle id="car_%i" type="typeCar" route="SN" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1
        		
        	if random.uniform(0,1) < p_WE:
        		print('    <vehicle id="car_%i" type="typeCar" route="WE" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1
        		
        	if random.uniform(0,1) < p_EW:
        		print('    <vehicle id="car_%i" type="typeCar" route="EW" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1
        		
        print("</routes>", file=routes)

def run(algorithm):
    """execute the TraCI control loop"""
    step = 0
     
    if algorithm == 0: #hardcoded
        print("hardcoded")
     
        ##
        vehiclesPast = 0 #need to count like that cause otherwise it only checks per 10 secs
        #traci.trafficlight.setPhase("A", 2) #trial 1
        traci.trafficlight.setPhase("A", 0) #trial 2
        ##
        while traci.simulation.getMinExpectedNumber() > 0:
            #trial 1 (induction loops)
            # if traci.trafficlight.getPhase("A") == 2:
            #     # if there are more than 2 cars have passed the induction loop (thus waiting), make it green
            #     if traci.inductionloop.getLastStepVehicleNumber("nA1") > 0 or traci.inductionloop.getLastStepVehicleNumber("nA0") > 0:
            #         vehiclesPast += 1
            #     if vehiclesPast > 2:
            #         traci.trafficlight.setPhase("A", 0)
            #         vehiclesPast = 0
            #     else:
            #         traci.trafficlight.setPhase("A", 2)
            ##

            #trial 2 lane area detectors 
            if traci.trafficlight.getPhase("A") == 0:
                if traci.lanearea.getLastStepHaltingNumber("wA0") > 2:
                     traci.trafficlight.setPhase("A", 2)
                else:
                     traci.trafficlight.setPhase("A", 0)

            traci.simulationStep()

            step += 1

    elif algorithm == 1: #q-learning
        print("q-learning")
        #create "q-table"
        qtable = create_qtable(18,2) #6 states, 2 actions
        total_reward = 0
        state = get_state()
        epsilon = 0.9
        alpha = 0.01 #1
        gamma = 0.01 #0


        while traci.simulation.getMinExpectedNumber() > 0:
            traci.trafficlight.setPhase("A", 2)
            
            action = choose_action(state, qtable, epsilon)
            if action == 0:
                traci.trafficlight.setPhase("A", 2)
            else:
                traci.trafficlight.setPhase("A", 0)

            for i in range(10): #changing this makes difference

                traci.simulationStep()

            step += 10

            next_state = get_state()
            reward = calc_reward()
            total_reward += reward
            qtable = update_table(qtable, reward, state, action, alpha, gamma, next_state)
            #print(qtable)
            #print(reward)
            #qtable[state,action] = reward
            state = next_state
            epsilon -= 0.01 #this might be something else
        
        print("total reward")
        print(total_reward)

    else: #original  
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
        

        
    traci.close()
    sys.stdout.flush()

# this is the main entry point of this script
def simulate_n_steps(N,gui_opt):
    # this will start sumo as a server, then connect and run
    if gui_opt=='nogui':
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile(N)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg","--tripinfo-output", "tripinfo.xml"]) # add ,"--emission-output","emissions.xml" if you want emissions report to be printed
    
    run(1) #enter the number for the algorithm to run
