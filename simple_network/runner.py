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
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def create_qtable(num_lights):
    qtable = 10 * np.random.random_sample((np.power(2, num_lights)*np.power(3, num_lights), np.power(2, len(num_lights))))
    return qtable


def calc_density(num_halt):
    if num_halt < 2:  # low density
        density = 0
    elif num_halt < 4:  # medium density
        density = 1
    else:
        density = 2

    return density


def get_state():
    halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0")
    halt_nA0 = traci.lanearea.getLastStepHaltingNumber("nA0")
    halt_eA0 = traci.lanearea.getLastStepHaltingNumber("eA0")
    halt_sA0 = traci.lanearea.getLastStepHaltingNumber("sA0")

    ### 2 
    density_horiz = calc_density(halt_wA0 + halt_eA0)
    density_vert = calc_density(halt_nA0 + halt_sA0)

    phase_A = traci.trafficlight.getPhase("A")  # phase of the traffic light: 0 or 2

    # THIS IS A VERY STUPID WAY OF DEFINING STATES
    # FIND SOMETHING BETTER
    states = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], [0, 0, 2],
         [0, 1, 2], [0, 2, 2], [1, 0, 2], [1, 1, 2], [1, 2, 2], [2, 0, 2], [2, 1, 2], [2, 2, 2]])
    state_values = np.array([density_horiz, density_vert, phase_A])
    state = np.where(np.all(states == state_values, axis=1))[0][0]
    # print("-------------- CURRENT STATE -----------------------")
    # print(np.where(np.all(states == state_values, axis=1)))
    return state


def choose_action(state, qtable, epsilon):
    chance = np.random.random()

    if epsilon <= chance:
        action = np.argmax(qtable[state, :])
    else:
        action = random.randint(0, np.size(qtable, 1) - 1)

    return action


def calc_reward():
    halt_horiz = traci.lanearea.getLastStepHaltingNumber("wA0") + traci.lanearea.getLastStepHaltingNumber("eA0")
    halt_vert = traci.lanearea.getLastStepHaltingNumber("nA0") + traci.lanearea.getLastStepHaltingNumber("sA0")
    # halt_wA0 = traci.lanearea.getLastStepHaltingNumber("wA0") #number of halting cars on wA0
    # reward = -1*halt_wA0 #this is gonna be total normally
    halt_total = halt_horiz + halt_vert
    reward = -1 * halt_total
    return reward


def update_table(qtable, reward, state, action, alpha, gamma, next_state):  # NOT SURE ABOUT THE Q-FUNCTION
    next_action = np.argmax(qtable[next_state, :])
    q = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * (qtable[next_state][next_action]))
    qtable[state][action] = q
    # print(q)
    # print(qtable)
    return qtable


def check_goal():  # NEED TO IMPLEMENT THIS TO END TRAINING
    return true


def plot(waiting_cars):
    plt.plot(waiting_cars)
    plt.ylabel('Number of waiting cars')
    plt.show()


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
                print('    <vehicle id="car_%i" type="typeCar" route="NS" depart="%i"/>' % (vehNr, i), file=routes)
                vehNr += 1

            if random.uniform(0, 1) < p_SN:
                print('    <vehicle id="car_%i" type="typeCar" route="SN" depart="%i"/>' % (vehNr, i), file=routes)
                vehNr += 1

            if random.uniform(0, 1) < p_WE:
                print('    <vehicle id="car_%i" type="typeCar" route="WE" depart="%i"/>' % (vehNr, i), file=routes)
                vehNr += 1

            if random.uniform(0, 1) < p_EW:
                print('    <vehicle id="car_%i" type="typeCar" route="EW" depart="%i"/>' % (vehNr, i), file=routes)
                vehNr += 1

        print("</routes>", file=routes)


def start_q_learning(epsilon, alpha, gamma, wait_time):
    print("start q-learning")

    waiting_cars_array = []
    waiting_cars = 0
    total_reward = 0
    time_step = 0
    step = 0

    # somehow get these from environment
    traffic_lights = ["A"]
    # num_of_actions = np.power(2, len(traffic_lights))

    qtable = create_qtable(len(traffic_lights))
    state = get_state()

    while traci.simulation.getMinExpectedNumber() > 0:
        action = choose_action(state, qtable, epsilon)
        bin_action = [int(x) for x in list('{0:0b}'.format(action))]

        for i in range(len(traffic_lights)):
            if bin_action[i] == 1:
                set_phase = 2
            else:
                set_phase = 0
            traci.trafficlight.setPhase(traffic_lights[i], set_phase)

        for i in range(wait_time):  # changing this makes difference
            traci.simulationStep()

        step += wait_time
        time_step += 1

        next_state = get_state()
        reward = calc_reward()
        total_reward += reward

        # to plot the total number of cars waiting for every 100 time steps
        if time_step < wait_time:
            waiting_cars += -1 * reward

        else:
            waiting_cars += -1 * reward
            waiting_cars_array.append(waiting_cars)
            waiting_cars = 0
            time_step = 0

        qtable = update_table(qtable, reward, state, action, alpha, gamma, next_state)
        # print(qtable)
        # print(reward)
        # qtable[state,action] = reward
        state = next_state
        # print("*********** the state ****************")
        # print(state)
        epsilon -= 0.01  # this might be something else

    print("total reward: %i" % total_reward)
    waiting_cars_array = np.hstack(waiting_cars_array)
    plot(waiting_cars_array)
    # print(rewards)


def start_original():
    waiting_cars_array = []
    total_reward = 0
    waiting_cars = 0
    time_step = 0
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        # traci.simulationStep()
        # step += 1
        for i in range(10):  # changing this makes difference

            traci.simulationStep()

        step += 10
        time_step += 1
        reward = calc_reward()
        # print("reward: %i" % reward)
        total_reward += reward

        # to plot the total number of cars waiting for every 100 time steps
        if time_step < 10:
            waiting_cars += -1 * reward

        else:
            waiting_cars += -1 * reward
            # print("waiting_cars %i" % waiting_cars)
            waiting_cars_array.append(waiting_cars)
            waiting_cars = 0
            time_step = 0

    print("total reward: %i" % total_reward)
    waiting_cars_array = np.hstack(waiting_cars_array)
    plot(waiting_cars_array)



def run(algorithm):
    """execute the TraCI control loop"""

    if algorithm == 1:  # q-learning
        start_q_learning(0.9, 0.01, 0.01, 10)
    else:  # original
        start_original()

    traci.close()
    sys.stdout.flush()


# this is the main entry point of this script
def simulate_n_steps(N, gui_opt):
    # this will start sumo as a server, then connect and run
    if gui_opt == 'nogui':
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile(N)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
                 "tripinfo.xml"])  # add ,"--emission-output","emissions.xml" if you want emissions report to be printed

    run(1)  # enter the number for the algorithm to run
