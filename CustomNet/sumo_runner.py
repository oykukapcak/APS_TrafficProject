#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0



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
from solver import genetic 
from solver import belief as blf
import copy 

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
    #qtable = np.zeros((num_states, num_actions), dtype =int)
    #NOT SURE HOW EXACTLY WE NEED TO INITIALIZE THIS
    qtable= 10 * np.random.random_sample((num_states, num_actions))
    return qtable

def calc_density(num_halt):
    if num_halt < 2:  # low density
        density = 0
    elif num_halt < 4:  # medium density
        density = 1
    else:
        density = 2

    return density

def create_state_matrix(halt_areas, traffic_lights):
    # halt = []
    # phases = []

    # define combinations of phases
    phase_combs = np.array(list(itertools.product([0, 2], repeat=len(traffic_lights))))

    # define combinations of densities
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
    return states

def get_state(state_matrix, halt_areas, traffic_lights):
    # TODO: Make scalable, not sure how this function works
    halt = []
    phases = []
    densities = []

    for i in range(len(halt_areas)):
        halt.append(traci.lanearea.getLastStepHaltingNumber(halt_areas[i]))
        densities.append(calc_density(halt[i]))

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
    #TODO: Make scalable, not sure how
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

    # traffic_lights = ["A"]                  # simple network
    # traffic_lights = ["A", "B"]           # complex network

    halt_areas = []
    with open("data/CustomNet.net.xml") as areas:
        lines = areas.readlines()

    for line in lines:
        if 'laneAreaDetector' in line:
            # print(line)
            area = line.split('id="')[1].split('"')[0]
            halt_areas.append(area)

    # halt_areas = ["wA0", "nA0", "eA0", "sA0"]                                       # simple network
    # halt_areas = ["wA0", "n1A0", "BA0", "s1A0", "eB0", "n2B0", "AB0", "s2B0"]     # complex network
    # num_of_actions = np.power(2, len(traffic_lights))

    state_matrix = create_state_matrix(halt_areas, traffic_lights)
    qtable = create_qtable(len(state_matrix), len(traffic_lights)*2)
    print(qtable)
    state = get_state(state_matrix, halt_areas, traffic_lights)
    print(state)

    while traci.simulation.getMinExpectedNumber() > 0:
        action = choose_action(state, qtable, epsilon)
        print("action %i" % action)
        bin_action = [int(x) for x in list('{0:0b}'.format(action))]

        for i in range(len(bin_action)):
            if bin_action[i] == 1:
                bin_action[i] = 2

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
    traffic_lights = ["gneJ4", "gneJ5", "gneJ6", "gneJ9", "gneJ10", "gneJ11"]
    belief = blf.Belief(traffic_lights)
    skip = 0 
    time_cycle = 120
    
    total_waiting = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        belief.current_tick = step
        belief.addCars(traci.simulation.getDepartedIDList())
        belief.removeCars(traci.simulation.getArrivedIDList())
        
        if step % time_cycle == 0 and skip < step and np.sum(list(belief.get_halting_cars().values())) > 0:
            if belief.hasCars():
                total_waiting.append(np.sum(list(belief.get_halting_cars().values())))

    print()
    print('amount of cars waited ', np.sum(total_waiting))
    print()
    traci.close()
    sys.stdout.flush()
    return total_waiting

def start_ga():
    """execute the TraCI control loop"""
    traffic_lights = ["gneJ4", "gneJ5", "gneJ6", "gneJ9", "gneJ10", "gneJ11"]
    belief = blf.Belief(traffic_lights)

    step = 0
    halting_cars = []
    time_step = 0
    waiting_cars = 0
    belief.current_tick = 0
    time_cycle = 120
    skip = 0

    belief.time_cycle = time_cycle
    
    ga = genetic.Genetic()
    # ga.initial_population(population_size=10, chromosome_size=8, segment_size=5, belief=belief)
    # print('pop 0', ga.population[0].chromosome)
    
    # # print('pop 0 decode', ga.population[0].decode())
    # print('fitness', ga.population[0].fitness())


    total_waiting = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        
        belief.current_tick = step
        belief.addCars(traci.simulation.getDepartedIDList())
        belief.removeCars(traci.simulation.getArrivedIDList())
        
        if step % time_cycle == 0 and skip < step and np.sum(list(belief.get_halting_cars().values())) > 0:
            if belief.hasCars():
                traci_prints(belief)

                # print(belief.get_halting_cars())
                print()
                print('waiting: ', np.sum(list(belief.get_halting_cars().values())))
                total_waiting.append(np.sum(list(belief.get_halting_cars().values())))
                # print('bs ', total_waiting)
                # print('AMUNA ', belief._get_phase_mapping())
                # new_logics = belief.starting_logics
                ga.initial_population(population_size=20, chromosome_size=8, segment_size=5, belief=belief)
                best, evo =  ga.approximate(epochs=32, verbose=1)
                new_logics = best.decode()
                set_new_logics(new_logics)

                # print('fitness starting ', sum(belief.calculate_logic_fitness(belief.starting_logics).values()))
                # print('fitness chromosome ', sum(belief.calculate_logic_fitness(new_logics).values()))
                # if False:
                    # print('wadawda', traci.trafficlight.getCompleteRedYellowGreenDefinition('gneJ5'))

                    # ga.initial_population(population_size=20, chromosome_size=8, segment_size=5, belief=belief)
                    # best, evo =  ga.approximate(epochs=4, verbose=1)
                    

                    # # print('total load: {}, {}'.format(np.sum(list(load.values())), load))
                    # print('new fitness: ', best.fitness())
                    # fits = [[i.calculated_fitness for i in li] for li in evo]
                    # for i in fits:
                    #     print(i)
                    
                    # print('DECODED ', best.decode())
                    # load = belief.calculate_load(step, time_cycle, passes=True, phases=belief.tls_phases)
                    # print('total load1: {}, {}'.format(np.sum(list(load.values())), load))
                    # load2 = belief.calculate_load(step, time_cycle, phases=best.decode(), passes=True)
                    # print('total load2: {}, {}'.format(np.sum(list(load2.values())), load2))
                    # new_load = belief.calculate_load(step, time_cycle, phases=best.decode(), passes=True)
                    # print('new total load: {}, {}'.format(best_fitness), new_load))

    print()
    print('amount of cars waited ', np.sum(total_waiting))
    print()
    traci.close()
    sys.stdout.flush()
    return total_waiting
    # plt.plot(halting_cars)
    # plt.xlabel("time")
    # plt.ylabel("#waiting cars")
    # plt.show()

def traci_prints(belief):
    car = belief.cars[0]
    lane_pos = traci.vehicle.getLanePosition(car)
    lane = traci.vehicle.getLaneID(car) # gneEX_X
    laneidx = traci.vehicle.getLaneIndex(car) # 0
    route = traci.vehicle.getRoute(car)
    speed = traci.vehicle.getSpeedWithoutTraCI(car)
    next_tls = traci.vehicle.getNextTLS(car)
    acspeed = traci.vehicle.getSpeed(car)
    maxspeed = traci.lane.getMaxSpeed(lane)
    # print('{}, tls {}, speed {}, max {}, '.format(car, next_tls, speed, maxspeed))
    lane = 'gneE17_0'
    wt = traci.lane.getWaitingTime(lane)
    tt = traci.lane.getTraveltime(lane)
    halting = traci.lane.getLastStepHaltingNumber(lane)
    # print('wt {}, tt {}'.format(wt, tt))
    # print('total load: {}, {}'.format(np.sum(list(belief.load.values())), belief.load))
    tls = 'gneJ6'
    complete = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
    # print()
    # # print('amount of logics ', len(complete))
    # print('current dur', complete[0].getPhases()[2]._duration)
    # print('first dur', belief.starting_logics[tls].getPhases()[2]._duration)
    # logic = copy.deepcopy(belief.starting_logics)[tls]
    # p = logic.getPhases()[2]
    # p._duration = 13
    # complete[0] = logic

    # complete[0].getPhases()[2]._duration = 13
    # print(len(complete[0].getPhases()))
    # print('type {}, subParameter {}, curPhaseIdx {}, subID {}'.format(logic._type, logic._subParameter, logic._currentPhaseIndex, logic._subID))
    # traci.trafficlight.setCompleteRedYellowGreenDefinition(tls, complete[0])
    
    curprogram = traci.trafficlight.getRedYellowGreenState(tls)
    program = traci.trafficlight.getProgram(tls)

    phase = traci.trafficlight.getPhase(tls)
    dur = traci.trafficlight.getPhaseDuration(tls)
    controlledlanes = traci.trafficlight.getControlledLanes(tls)
    switch = traci.trafficlight.getNextSwitch(tls)
    # traci.trafficlight.setCompleteRedYellowGreenDefinition(tls, complete)
    # t = belief._get_edge_condition('gneJ6', 'gneE5_0', step, step + 1)
    controlledlinks = traci.trafficlight.getControlledLinks(tls)
    # print('links {}'.format(controlledlanes))
    # print('halting {}'.format(halting))

    # print('{}, tls {}, speed {}, max {}, '.format(car, next_tls, speed, maxspeed))
    # print('prog {}, phase {}, dur {}'.format(curprogram, phase, dur))
    # print(type(complete[0].getPhases()), type(complete[0].getPhases()[0]), complete[0].getPhases()[0]._phaseDef)
    # print(complete[0].getPhases())
    
    # print(route)
    # phase = solve(halt_nA0, halt_eA0, halt_sA0, halt_wA0)
    
    # traci.trafficlight.setPhase("gneJ6", 0)
def set_new_logics(logics):
    for k, v in logics.items():
        traci.trafficlight.setCompleteRedYellowGreenDefinition(k, v)

# def ga_config():
#     genetic.POPULATION_SIZE = 100
#     genetic.CHROMOSOME_SIZE = 1

# def solve(n, e, s, w):
#     ga = genetic.Genetic()
#     ga.initial_population(chromosome_params=[n, e, s, w])
#     best, evo = ga.approximate()
#     return 2*best.chromosome[0]


def run(algorithm):
    """execute the TraCI control loop"""

    results = None
    if algorithm == 1:  # q-learning
        results = start_original()
    elif algorithm == 2:
        results = start_ga()
    # else:  # original
    #     start_q_learning(0.9, 0.01, 0.01, 10)

    traci.close()
    sys.stdout.flush()
    return results


# this is the main entry point of this script
def simulate_n_steps(algorithm, gui_opt):
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

    return run(algorithm)  # enter the number for the algorithm to run
