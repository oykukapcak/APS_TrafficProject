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
import itertools

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

# def create_qtable(num_lights):
#    qtable = 10 * np.random.random_sample((np.power(2, num_lights)*np.power(3, num_lights), np.power(2, len(num_lights))))
#    return qtable


def create_qtable(num_states, num_actions):
    # qtable = np.zeros((num_states, num_actions), dtype =int)
    # NOT SURE HOW EXACTLY WE NEED TO INITIALIZE THIS
    print("creating qtable")
    qtable= 10 * np.random.random_sample((num_states, num_actions))
    print("created qtable")
    print(qtable)
    return qtable


def calc_density(num_halt):
    if num_halt < 2:  # low density
        density = 0
    elif num_halt < 4:  # medium density
        density = 1
    else:
        density = 2

    return density


def calc_density2(num_halt, lane_length):
    car_size = 5.0
    density = num_halt*car_size/lane_length
    return density


def create_state_matrix(halt_areas, traffic_lights):
    # halt = []
    # phases = []
    print("creating state matrix")

    # define combinations of phases
    phase_combs = np.array(list(itertools.product([0, 2], repeat=len(traffic_lights))))

    # define combinations of densities
    print(len(halt_areas))
    density_combs = np.array(list(itertools.product([0, 1, 2], repeat=len(halt_areas))))

    densities = np.tile(density_combs, (len(phase_combs), 1))
    phases = np.zeros((len(densities), len(traffic_lights)), dtype=int)

    t = 0
    for i in phase_combs:
        a = np.tile(i, (len(density_combs), 1))
        phases[len(density_combs)*t:len(density_combs)*(t + 1)] = a
        t += 1

    # generate the state matrix
    states = np.concatenate((densities, phases), axis=1)
    print("created state matrix:")
    print(states)
    return states


def get_state(state_matrix, halt_areas, traffic_lights):
    # TODO: Make scalable, not sure how this function works
    halt = []
    phases = []
    densities = []
    # lane_lengths = []

    for i in range(len(halt_areas)):
        halt.append(traci.lanearea.getLastStepHaltingNumber(halt_areas[i]))
        densities.append(calc_density(halt[i]))

    # for i in range(len(halt_areas)):
    #    halt.append(traci.lane.getLastStepHaltingNumber(halt_areas[i]))
    #    lane_lengths.append(traci.lane.getLength(halt_areas[i]))
    #    densities.append(calc_density2(halt[i]), lane_lengths[i])

    for i in range(len(traffic_lights)):
        phase = traci.trafficlight.getPhase(traffic_lights[i])
        phases.append(phase)

    state_values = np.concatenate((densities, phases), axis=None)
    state = np.where(np.all(state_matrix == state_values, axis=1))[0][0]
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


def calc_reward(halt_areas):
    total_halt = 0

    for i in range(len(halt_areas)):
        total_halt += traci.lanearea.getLastStepHaltingNumber(halt_areas[i])

    return -1 * total_halt


def update_table(qtable, reward, state, action, alpha, gamma, next_state):  # NOT SURE ABOUT THE Q-FUNCTION
    next_action = np.argmax(qtable[next_state, :])
    q = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * (qtable[next_state][next_action]))
    qtable[state][action] = q

    return qtable


# def check_goal():  # NEED TO IMPLEMENT THIS TO END TRAINING
#     return true


def plot(waiting_cars):
    plt.plot(waiting_cars)
    plt.ylabel('Number of waiting cars')
    plt.show()


def generate_routefile(N):
    # TODO: Make scalable, not sure how
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

    # Traffic lights and lane area detectors are found from xml files. Directories might change.
    traffic_lights = []

    with open("data/CustomNet.net.xml") as nodes:
        lines = nodes.readlines()

    for line in lines:
        if 'traffic_light' in line:
            # print(line)
            light = line.split('id="')[1].split('"')[0]
            traffic_lights.append(light)
            # lanes = traci.trafficlight.getControlledLanes(light) # to get the lanes leading to that traffic light

            # for i in range(len(lanes)):
            #    halt_areas.append(lanes[i])
    print("Traffic lights are found: ")
    print(traffic_lights)

    # traffic_lights = ["A"]                  # simple network
    # traffic_lights = ["A", "B"]           # complex network

    halt_areas = []
    with open("data/CustomNetAdditionals2.xml") as areas:
        lines = areas.readlines()

    for line in lines:
        if 'laneAreaDetector' in line:
            # print(line)
            area = line.split('id="')[1].split('"')[0]
            halt_areas.append(area)

    print("Lane area detectors are found: ")
    print(halt_areas)

    # halt_areas = ["wA0", "nA0", "eA0", "sA0"]                                       # simple network
    # halt_areas = ["wA0", "n1A0", "BA0", "s1A0", "eB0", "n2B0", "AB0", "s2B0"]     # complex network
    # num_of_actions = np.power(2, len(traffic_lights))

    state_matrix = create_state_matrix(halt_areas, traffic_lights)
    qtable = create_qtable(len(state_matrix), 2**len(traffic_lights))
    state = get_state(state_matrix, halt_areas, traffic_lights)
    # print("Current state: %i:" % state)

    while traci.simulation.getMinExpectedNumber() > 0:
        action = choose_action(state, qtable, epsilon)
        # print("Chosen action: %i" % action)
        bin_action = [int(x) for x in list('{0:0b}'.format(action))]

        for i in range(len(bin_action)):
            if bin_action[i] == 1:
                bin_action[i] = 2

        length = len(traffic_lights)
        if len(bin_action) < length:
            a = np.zeros(length, int)
            j = length - 1
            for i in reversed(bin_action):
                a[j] = i
                j -= 1
            bin_action = a

        for i in range(len(traffic_lights)):
            traci.trafficlight.setPhase(traffic_lights[i], bin_action[i])

        for i in range(wait_time):  # changing this makes difference
            traci.simulationStep()

        step += wait_time
        time_step += 1

        next_state = get_state(state_matrix, halt_areas, traffic_lights)
        reward = calc_reward(halt_areas)
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
        #print("*********** the state ****************")
        #print(state)
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
    # generate_routefile(N) # commented out cause we have the premade route file now

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/CustomNet.sumocfg", "--tripinfo-output",
                 "tripinfo.xml"])  # add ,"--emission-output","emissions.xml" if you want emissions report to be printed

    run(1)  # enter the number for the algorithm to run
