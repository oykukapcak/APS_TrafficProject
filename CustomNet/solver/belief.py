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

    def calculate_load(self, time):
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
                rnge = general_speed * time

                ## Placeholder
                state = tls[3]
                condition = (state == 'r') # Change condition to compare to next state
                if rnge > tls[2] and condition:
                    load[name] += 1
                ##

        self.load = load
        return load

    def phase_mapping(self):
        mapping = {}
        
        return 0