import numpy as np
import traci

class Belief:

    def __init__(self):
        self.cars = []
        self.network = {}
        self.load = {}

    def addCars(self, car_ids):
        for i in car_ids:
            self.cars.append(i)

    def removeCars(self, car_ids):
        for i in car_ids:
            self.cars.remove(i)

    def hasCars(self):
        return len(self.cars) > 0

    def calculate_load(self, cur_tick, t):
        general_speed = 12 # Needs to be done dynamically
        # calculate the load per tls. Calculate how many cars are going to be waiting
        load = {}
        for c in self.cars:
            next_tls = traci.vehicle.getNextTLS(c)
            # For every tls in the list
            for tls in next_tls:
                # check if its within range
                name = tls[0]
                if name not in load:
                    load[name] = 0
                rnge = general_speed * t

                ## Placeholder
                # Get the state it is going to be after #ticks
                # + Get all of the states from that light
                # + Get the lane which the car is going to take while approaching the tls
                # + Get the light phase on time t. 
                # + Get the position in the phase of the light depending on the car's lane
                
                edge = self._get_approaching_lane(c, name)
                state = self._get_edge_condition(name, edge, cur_tick, cur_tick + t)
                condition = (state == 'r' or state == 'y') # Change condition to compare to next state
                if rnge > tls[2] and condition:
                    load[name] += 1
                ##

        self.load = load
        return load
    
    def _get_approaching_lane(self, car, tls):
        controlled = [i.split('_')[0] for i in traci.trafficlight.getControlledLanes(tls)] #TODO: REMOVE 
        route = list(traci.vehicle.getRoute(car))


        # print('controlled ', controlled)
        # print('car ', car)
        # print('route ', route)
        res = ''
        for edge in route:
            # print('edge', type(edge), edge)
            if edge in controlled:
                res = edge
                break

        return res if res else ValueError('No edges in route that connects to that tls.')

    def _get_all_tls_phases(self,tls):
        """
            Return list of Phases
            [ ..., 
            Phase:
            duration: 10.0
            minDuration: 10.0
            maxDuration: 10.0
            phaseDef: rrrrrrrrrryyrrrrrrrrrryy
            ]
        """
        return list(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].getPhases())

    # Get edge condition at time t
    def _get_edge_condition(self, tls, edge, cur_tick, t):
        phases = self._get_all_tls_phases(tls)
        # total_duration = 120 # change to dynamic?
        total_duration = np.sum([p._duration for p in phases])
        res_tick = cur_tick % total_duration
        # print('td ', total_duration)
        # print('phases', type(phases), len(phases))
        # print()
        # print('res1', res_tick)
        for i in range(len(phases)):
            # print('pdur', phases[i]._duration)
            res_tick -= phases[i]._duration
            # print('res'+str(i), res_tick)
            if res_tick < 0:
                phase_at_t = phases[i]
                break            
        
        # Turn phase into condition
        controlled = [i.split('_')[0] for i in traci.trafficlight.getControlledLanes(tls)]
        light = 'r'
        for i in range(len(controlled)):
            if controlled[i] == edge and  phase_at_t._phaseDef[i] == 'G':
                light = 'G'
        return light

