import numpy as np
import traci

class Belief:

    def __init__(self, traffic_light_list):
        self.cars = []
        self.network = {}
        self.load = {}
        self.tls_phases = self._init_tls_phases()
        self.current_tick = 0
        self.traffic_lights = traffic_light_list
        self.controlled_lanes = self._get_controlled_lanes(traffic_light_list)
        self.starting_logics = self._get_logic_per_tls()
        self.halted_cars = {}
        self.phase_mapping = self._get_phase_mapping()

    def _init_tls_phases(self):
        traffic_lights = ["gneJ4", "gneJ5", "gneJ6", "gneJ9", "gneJ10", "gneJ11"]
        d = {}
        phase_count = 0
        for tls in traffic_lights:
            phases = self._get_all_tls_phases(tls)
            # print(len(phases), phases)
            phase_count += len(phases)
            # print('count', phase_count)
            d[tls] = phases
        self.phase_count = phase_count
        return d

    def addCars(self, car_ids):
        for i in car_ids:
            self.cars.append(i)

    def removeCars(self, car_ids):
        for i in car_ids:
            self.cars.remove(i)

    def hasCars(self):
        return len(self.cars) > 0

    def _get_controlled_lanes(self, traffic_lights): 
        # Get controlled lanes
        controlled_lanes = set()
        for tls in self.traffic_lights:
            for lane in traci.trafficlight.getControlledLanes(tls):
                controlled_lanes.add(lane)
        controlled_lanes = list(controlled_lanes)
        return controlled_lanes
    
    def _get_logic_per_tls(self):
        logics = {}
        for tls in self.traffic_lights:
            logics[tls] = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
        return logics

    # Create mapping tls->phase->lanes
    def _get_phase_mapping(self):
        tls_d = {}
        for tls, logic in self.starting_logics.items():
            tls_lanes = traci.trafficlight.getControlledLanes(tls)
            phases = logic.getPhases()
            phase_book = {}
            for p in phases:
                phase_lanes = []
                for i in range(len(p._phaseDef)):
                    if p._phaseDef[i] == 'G' or p._phaseDef[i] == 'g':
                        phase_lanes.append(tls_lanes[i])
                phase_book[p._phaseDef] = list(set(phase_lanes))
            tls_d[tls] = phase_book
        # print(tls_d)
        return tls_d
    ## Method 1
    # Get the state it is going to be after #ticks
    # + Get all of the states from that light
    # + Get the lane which the car is going to take while approaching the tls
    # + Get the light phase on time t. 
    # + Get the position in the phase of the light depending on the car's lane
    
    def calculate_load(self, cur_tick, t, phases=[], method=0, passes=False ):
        general_speed = 10 # Needs to be done dynamically
        # calculate the load per tls. Calculate how many cars are going to be waiting
        load = {}
        # Method 1, check per car
        for c in self.cars:
            next_tls = traci.vehicle.getNextTLS(c)
            # For every tls in the list
            for tls in next_tls:
                # Method 1 check if its within range
                name = tls[0]
                edge = self._get_approaching_lane(c, name)
                if edge not in load:
                    load[edge] = 0
                rnge = general_speed * t

               
                state = self._get_edge_condition(name, edge, cur_tick, cur_tick + t, new_phases=phases)
                if passes:
                    condition = (state == 'G')
                else: 
                    condition = (state == 'r' or state == 'y') # Change condition to compare to next state
                if rnge > tls[2] and condition:
                    load[edge] += 1
                ##

        # self.load = load
        return load
    
    ## Method 2
    # + Get halted cars right now (SHOULD BE DONE ONCE)
    # - Get mapping of phases to lanes
    def get_halting_cars(self, phases=[]):
        # print('Cur tick {}, has {} keys.'.format(self.current_tick, len(self.halted_cars.keys())))
        assert len(self.traffic_lights) > 0

        if self.current_tick not in self.halted_cars.keys():
            halting_cars = {}
            # (SHOULD BE DONE ONCE)
            for lane in self.controlled_lanes:
                halting_cars[lane] = traci.lane.getLastStepHaltingNumber(lane)
                # halting_cars[lane] = traci.lane.getLastStepVehicleNumber(lane)

            self.halted_cars[self.current_tick] = halting_cars
        
        return self.halted_cars[self.current_tick]

    def calculate_logic_fitness(self, tls_logics):
        # Get tls -> phase -> lanes
        # For every tls logic
        all_fitnesses = {}
        halted = self.get_halting_cars()
        actual_total = 0
        for tls, logic in tls_logics.items():
            # traci.trafficlight.getControlledLanes(tls)
            phases = logic.getPhases()
            for p in phases:
                lanes = self.phase_mapping[tls][p._phaseDef]
                # maybe floor
                # print('tls {}, phase {}'.format(tls,p._phaseDef))
                
                for l in lanes:
                    fitness = min(p._duration / 3.0, halted[l])
                    all_fitnesses[l] = fitness
                    actual_total += fitness
                    # print('fitness {} for lane {}'.format(fitness, l))
        # print('dict tot ', sum(all_fitnesses.values()))
        # print('actual tot ', actual_total)
        return all_fitnesses

    # def get_cars_that_will_pass()

    def _get_approaching_lane(self, car, tls):
        controlled = [i.split('_')[0] for i in traci.trafficlight.getControlledLanes(tls)] #TODO: REMOVE 
        route = list(traci.vehicle.getRoute(car))
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
    def _get_edge_condition(self, tls, edge, cur_tick, t, new_phases=[]):
        # if new_phases:
        # print ('new garbage ', new_phases)
        assert any(new_phases), 'shit empty yo'
        phases = new_phases[tls]
        # else:
        #     print('SOME BULL SHIT ')    
        #     phases = self._get_all_tls_phases(tls)
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
            if controlled[i] == edge and phase_at_t._phaseDef[i] == 'G':
                light = 'G'
        return light
