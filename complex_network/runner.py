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
import itertools
from itertools import product
import matplotlib.pyplot as plt


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
    num_detectors = 8
    num_phases = 4

    halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0") #number of halting cars on wA0
    halt_n1A0 = traci.lanearea.getLastStepHaltingNumber("n1A0")
    halt_BA0 = traci.lanearea.getLastStepHaltingNumber("BA0") 
    halt_s1A0 = traci.lanearea.getLastStepHaltingNumber("s1A0")
    halt_eB0 = traci.lanearea.getLastStepHaltingNumber("eB0") 
    halt_n2B0 = traci.lanearea.getLastStepHaltingNumber("n2B0")
    halt_AB0 = traci.lanearea.getLastStepHaltingNumber("AB0") 
    halt_s2B0 = traci.lanearea.getLastStepHaltingNumber("s2B0")

    #density_wA0 = calc_density(halt_wA0)
    #density_nA0 = calc_density(halt_nA0)
    
    ### Below is two different ways of computing the density
    ### 1
    #density_horiz = calc_density(halt_wA0) + calc_density(halt_eA0)
    #density_vert = calc_density(halt_nA0) + calc_density(halt_sA0)
    
    ### 2 
    #calculate densities at different directions
    density_horiz1 = calc_density(halt_wA0 + halt_BA0)
    density_vert1 = calc_density(halt_n1A0 + halt_s1A0)
    density_horiz2 = calc_density(halt_AB0 + halt_eB0)
    density_vert2 = calc_density(halt_n2B0 + halt_s2B0)

    phase_A = traci.trafficlight.getPhase("A") #phase of the traffic light: 0 or 2
    phase_B = traci.trafficlight.getPhase("B") #phase of the traffic light: 0 or 2

    #create state matrix

    #define combinations of densities
    density_combs = np.array(list(itertools.product([0,1,2], repeat=4)))
    densities = np.tile(density_combs, (4,1))
    #define combinations of phases
    phase_combs = np.array([[0,0], [0,2], [2,0], [2,2]])
    phases = np.zeros((len(densities), 2), dtype=int)

    t = 0
    for i in phase_combs:
        a = np.tile(i,(81,1))
        phases[81*t:81*(t+1)] = a
        t +=1

    #genereate the state matrix
    states = np.concatenate((densities, phases), axis = 1)

    #states = np.array([[0, 0, 0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], [0, 0, 2], [0, 1, 2], [0, 2, 2], [1, 0, 2], [1, 1, 2], [1, 2, 2], [2, 0, 2], [2, 1, 2], [2, 2, 2]])
    state_values = np.array([density_horiz1, density_vert1, density_horiz2, density_vert2, phase_A, phase_B])
    #print(state_values)
    state = np.where(np.all(states==state_values, axis=1))[0][0]
    #print("state %i" %state)
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
    #compute total number of halting cars 
    halt_total = traci.lanearea.getLastStepHaltingNumber("wA0") + \
        traci.lanearea.getLastStepHaltingNumber("n1A0") + \
        traci.lanearea.getLastStepHaltingNumber("BA0") + \
        traci.lanearea.getLastStepHaltingNumber("s1A0") + \
        traci.lanearea.getLastStepHaltingNumber("eB0") + \
        traci.lanearea.getLastStepHaltingNumber("n2B0") + \
        traci.lanearea.getLastStepHaltingNumber("AB0") + \
        traci.lanearea.getLastStepHaltingNumber("s2B0")

    #halt_horiz = traci.lanearea.getLastStepHaltingNumber("wA0") + traci.lanearea.getLastStepHaltingNumber("eA0")
    #halt_vert = traci.lanearea.getLastStepHaltingNumber("nA0") + traci.lanearea.getLastStepHaltingNumber("sA0")
    #halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0") #number of halting cars on wA0
    #reward = -1*halt_wA0 #this is gonna be total normally
    #halt_total = halt_horiz + halt_vert
    reward = -1*halt_total
    return reward

def update_table(qtable, reward, state, action, alpha, gamma, next_state): #NOT SURE ABOUT THE Q-FUNCTION
    next_action = np.argmax(qtable[next_state,:])
    q = (1-alpha)*qtable[state,action] + alpha*(reward+gamma*(qtable[next_state][next_action]))
    qtable[state][action] = q
    #print(q)
    #print(qtable)
    return qtable


def check_goal(): #NEED TO IMPLEMENT THIS TO END TRAINING 
    return true

def plot_rewards(rewards):
    plt.plot(rewards)   
    plt.ylabel('Reward')
    plt.show()

def generate_routefile(N):
    random.seed()  # make tests reproducible by random.seed(some_number)
    
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeCar" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

        <route id="NS1" edges="n12A A2s1" />
        <route id="SN1" edges="s12A A2n1" />
        <route id="NS2" edges="n22B B2s2" />
        <route id="SN2" edges="s22B B2n2" />
        <route id="WE" edges="w2A A2B B2e" />
        <route id="EW" edges="e2B B2A A2w" />""", file=routes)
        
        vehNr = 0
        
        # demand in each direction per time step
        # CHANGE THIS!!!!!
        p_NS1 = 3. / 20
        p_SN1 = 3. / 20
        p_NS2 = 3. / 20
        p_SN2 = 3. / 20
    	p_WE = 3. / 20
    	p_EW = 3. / 20
        
        for i in range(N):
            if random.uniform(0, 1) < p_NS2:
                print('    <vehicle id="car_%i" type="typeCar" route="NS2" depart="%i"/>' % (vehNr,i), file=routes)
                vehNr += 1

        	if random.uniform(0, 1) < p_NS1:
        		print('    <vehicle id="car_%i" type="typeCar" route="NS1" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1
        		
        	if random.uniform(0, 1) < p_SN1:
        		print('    <vehicle id="car_%i" type="typeCar" route="SN1" depart="%i"/>' % (vehNr,i), file=routes)
        		vehNr += 1

            if random.uniform(0, 1) < p_SN2:
                print('    <vehicle id="car_%i" type="typeCar" route="SN2" depart="%i"/>' % (vehNr,i), file=routes)
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
     
    # if algorithm == 0: #hardcoded
    #     print("hardcoded")
     
    #     ##
    #     total_reward = 0 
    #     vehiclesPast = 0 #need to count like that cause otherwise it only checks per 10 secs
    #     #traci.trafficlight.setPhase("A", 2) #trial 1
    #     traci.trafficlight.setPhase("A", 0) #trial 2
    #     ##
    #     while traci.simulation.getMinExpectedNumber() > 0:
    #         #lane area detectors 
    #         if traci.trafficlight.getPhase("A") == 0:
    #             if traci.lanearea.getLastStepHaltingNumber("wA0") or traci.lanearea.getLastStepHaltingNumber("BA0") > 2:
    #                 traci.trafficlight.setPhase("A", 2)
    #             else:
    #                  traci.trafficlight.setPhase("A", 0)
    #             if traci.lanearea.getLastStepHaltingNumber("eB0") or traci.lanearea.getLastStepHaltingNumber("AB0") > 2:
    #                 traci.trafficlight.setPhase("B", 2)
    #             else:
    #                  traci.trafficlight.setPhase("B", 0)

    #         #traci.simulationStep()
    #         #step += 1
    #         for i in range(10): #changing this makes difference

    #             traci.simulationStep()

    #         step += 10

    #         reward = calc_reward()
    #         total_reward += reward

    #     print("total reward %i" % total_reward)


    if algorithm == 1: #q-learning
        print("q-learning")
        #create "q-table"
        qtable = create_qtable(324,4) #324 states, 4 actions
        total_reward = 0
        state = get_state()
        epsilon = 0.9
        alpha = 0.01 #1
        gamma = 0.01 #0
        rewards = []


        while traci.simulation.getMinExpectedNumber() > 0:
            traci.trafficlight.setPhase("A", 2)
            traci.trafficlight.setPhase("B", 2)
            
            action = choose_action(state, qtable, epsilon)
            if action == 0:
                traci.trafficlight.setPhase("A", 2)
                traci.trafficlight.setPhase("B", 2)
            elif action == 1:
                traci.trafficlight.setPhase("A", 0)
                traci.trafficlight.setPhase("B", 2)
            elif action == 2:
                traci.trafficlight.setPhase("A", 2)
                traci.trafficlight.setPhase("B", 0)
            else:
                traci.trafficlight.setPhase("A", 0)
                traci.trafficlight.setPhase("B", 0)

            for i in range(10): #changing this makes difference

                traci.simulationStep()

            step += 10

            next_state = get_state()
            reward = calc_reward()
            total_reward += reward
            rewards.append(reward)
            qtable = update_table(qtable, reward, state, action, alpha, gamma, next_state)
            #print(qtable)
            print("reward %i" % reward)
            #qtable[state,action] = reward
            state = next_state
            epsilon -= 0.01 #this might be something else
        
        print("total reward %i" % total_reward)
        rewards = np.hstack(rewards)
        plot_rewards(rewards)

    else: #fixed time light phases  
        total_reward = 0
        rewards = []
        while traci.simulation.getMinExpectedNumber() > 0:
            #traci.simulationStep()
            #step += 1
            for i in range(10): #changing this makes difference

                traci.simulationStep()

            step += 10
            reward = calc_reward()
            total_reward += reward
            rewards.append(reward)

        print("total_reward %i" % total_reward)      
        rewards = np.hstack(rewards)
        plot_rewards(rewards)  

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
    
    run(2) #enter the number for the algorithm to run
