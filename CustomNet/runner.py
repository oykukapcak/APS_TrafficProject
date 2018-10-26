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


def parse_xmls():
    traffic_lights = []
    detectors = []
    actions = []
    with open("data/CustomNet.net.xml") as nodes:
        lines = nodes.readlines()

    with open("data/CustomNetAdditionals2.xml") as areas:
        lines2 = areas.readlines()

    for line in lines:
        if 'traffic_light' in line:
            # print(line)
            light = line.split('id="')[1].split('"')[0]
            # traffic_lights.append(light)
            lanes = traci.trafficlight.getControlledLanes(light) # to get the lanes leading to that traffic light

            halt_areas = []
            for i in range(len(lanes)):
                lane = lanes[i]
                for line2 in lines2:
                    if 'e2Detector' in line2:
                        if lane in line2:
                            # print(line)
                            area = line2.split('id="')[1].split('"')[0]
                            if area not in halt_areas:
                                halt_areas.append(area)

            program_definition = traci.trafficlight.getCompleteRedYellowGreenDefinition(light)
            phases = list(range(0, len(program_definition[0]._phases)))
            actions.append(list(phases))

            traffic_lights.append(light)
            detectors.append(list(halt_areas))
            # for i in range(len(lanes)):
            #    halt_areas.append(lanes[i])

    print("Traffic lights are found: ")
    print(traffic_lights)

    print("Lane area detectors are found: ")
    print(detectors)

    print("Action set for each traffic light: ")
    print(actions)

    return traffic_lights, detectors, actions


def create_qtable(state_matrix, actions):
    # qtable = np.zeros((num_states, num_actions), dtype =int)
    # NOT SURE HOW EXACTLY WE NEED TO INITIALIZE THIS
    print("creating qtable")
    # creates a single qtable for each intersection
    qtable = []

    for i in range(len(state_matrix)):
        states = state_matrix[i]
        num_states = len(states)
        num_actions = len(actions[i])
        table = 10 * np.random.random_sample((num_states, num_actions))
        qtable.append(np.array(table))

    print("created qtable")
    # print(qtable)
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


def create_state_matrix(detectors, actions):
    state_matrix = []
    print("creating state matrix")

    # creates a single state matrix for each intersection, stores them in a big array
    for i in range(len(detectors)):
        # define combinations of phases
        phase_combs = np.array(list(itertools.product(actions[i])))

        # define combinations of densities
        # print(len(halt_areas))
        # density_combs = np.array(list(itertools.product([0, 1, 2], repeat=len(halt_areas))))
        density_combs = np.array(list(itertools.product([0, 1, 2], repeat=len(detectors[i]))))

        densities = np.tile(density_combs, (len(phase_combs), 1))
        phases = np.zeros((len(densities), 1), dtype=int)

        t = 0
        for i in phase_combs:
            a = np.tile(i, (len(density_combs), 1))
            phases[len(density_combs)*t:len(density_combs)*(t + 1)] = a
            t += 1

        # generate the state matrix
        states = np.concatenate((densities, phases), axis=1)
        state_matrix.append(np.array(states))

    print("created state matrix:")
    # print(state_matrix)
    return state_matrix


def get_state(state_matrix, detectors, traffic_lights):

    # First compute state arrays per light
    all_lights_state = []

    for i in range(len(detectors)):
        state_per_light = []
        for detector in detectors[i]:
            density = calc_density(traci.lanearea.getLastStepHaltingNumber(detector))
            state_per_light.append(density)

        phase = traci.trafficlight.getPhase(traffic_lights[i])
        state_per_light.append(phase)

        all_lights_state.append(state_per_light)

    # Then find the corresponding row index from state matrix to return real stata number
    real_states = []
    for s in range(len(all_lights_state)):
        # print("Light no: %i" %s)
        # print("State to be found: ")
        # print(all_lights_state[s])
        # print("State matrix to be searched: ")
        # print(state_matrix[s])
        real_state = np.where(np.all(state_matrix[s] == all_lights_state[s], axis=1))[0][0]
        real_states.append(real_state)

    # print("-------------- CURRENT STATE -----------------------")
    # print(np.where(np.all(states == state_values, axis=1)))
    return real_states


def choose_action(state, qtable, epsilon):
    # Choose an action for each traffic light
    # Thus, it needs to go over each sub qtable
    actions = []

    for i in range(len(state)):
        subqtable = qtable[i]

        chance = np.random.random()

        if epsilon <= chance:
            action = np.argmax(subqtable[state[i], :])
        else:
            action = random.randint(0, np.size(subqtable, 1) - 1)

        actions.append(action)

    return actions


def calc_reward(halt_areas):
    total_halt = 0

    for item in halt_areas:
        for i in item:
            total_halt += traci.lanearea.getLastStepHaltingNumber(i)

    return -1 * total_halt


def update_table(qtable, reward, states, actions, alpha, gamma, next_states):  # NOT SURE ABOUT THE Q-FUNCTION
    # Decide on next actions for each traffic light
    next_actions = []

    for i in range(len(states)):
        subqtable = qtable[i]
        action = actions[i]
        state = states[i]
        next_state = next_states[i]
        next_action = np.argmax(subqtable[next_state, :])

        q = (1 - alpha) * subqtable[state, action] + alpha * (reward + gamma * (subqtable[next_state, next_action]))
        subqtable[state][action] = q

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
    traffic_lights, detectors, actions = parse_xmls()

    state_matrix = create_state_matrix(detectors, actions)
    qtable = create_qtable(state_matrix, actions)
    state = get_state(state_matrix, detectors, traffic_lights)
    # print("Current state: %i:" % state)

    while traci.simulation.getMinExpectedNumber() > 0:
        chosen_actions = choose_action(state, qtable, epsilon)
        # print("Chosen action: %i" % action)
        # bin_action = [int(x) for x in list('{0:0b}'.format(action))]

        for i in range(len(traffic_lights)):
            traci.trafficlight.setPhase(traffic_lights[i], chosen_actions[i])

        for i in range(wait_time):  # changing this makes difference
            traci.simulationStep()

        step += wait_time
        time_step += 1

        next_state = get_state(state_matrix, detectors, traffic_lights)
        reward = calc_reward(detectors)
        total_reward += reward

        # to plot the total number of cars waiting for every 100 time steps
        if time_step < wait_time:
            waiting_cars += -1 * reward

        else:
            waiting_cars += -1 * reward
            waiting_cars_array.append(waiting_cars)
            waiting_cars = 0
            time_step = 0

        qtable = update_table(qtable, reward, state, actions, alpha, gamma, next_state)
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
        start_q_learning(0.9, 0.01, 0.01, 30)
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
