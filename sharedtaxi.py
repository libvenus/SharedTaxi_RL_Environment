import sys
import copy
from io import StringIO

import gym
from gym import utils
from gym.envs.toy_text import discrete

import random
import numpy as np
from numpy import random

from taxi import Taxi
from passenger import Passenger

class SharedTaxi(discrete.DiscreteEnv): #Playground
    def __init__(self, no_of_rows, no_of_cols, no_of_actions, max_passengers):
        self.P = {}
        self.no_of_rows = no_of_rows
        self.no_of_cols = no_of_cols
        self.position_to_coordinates = {}
        self.coordinates_to_position = {}
        self.no_of_actions = no_of_actions
        self.taxi_max_capacity = max_passengers
        self.no_of_passengers = max_passengers
        self.position_in_taxi = no_of_rows * no_of_cols
        self.MAP = self.create_visual_grid(no_of_rows, no_of_cols)
        self.desc = np.asarray(self.MAP, dtype='c')

        position = 0
        for row in range(no_of_rows):
            for col in range(no_of_cols):
                self.position_to_coordinates[position] = (row, col)
                self.coordinates_to_position[(row, col)] = position
                position = position + 1

        self.position_to_coordinates[self.position_in_taxi] = (no_of_rows, no_of_cols)
        self.coordinates_to_position[(no_of_rows, no_of_cols)] = self.position_in_taxi

        self.states, self.state_to_no, self.no_to_state = self._get_states()
        self.init_state_dist = np.zeros(len(self.states))

        state_keys = list(self.states.keys())
        random.shuffle(state_keys)

        self._explore_environment(state_keys)
        
        self.init_state_dist /= self.init_state_dist.sum()
        discrete.DiscreteEnv.__init__(
            self, len(self.states), self.no_of_actions, self.P, self.init_state_dist)

    def _explore_environment(self, state_keys):

        for state in state_keys:
            self.P[self.state_to_no[state]] = {action : [] for action in range(self.no_of_actions)}

            if self._is_state_valid(state): self.init_state_dist[self.state_to_no[state]] += 1
            
            for action in range(self.no_of_actions):
                taxi_loc, pass1_loc, pass1_dest, pass2_loc, pass2_dest = self._get_pos_from_state(state)
                done = False
                taxi = self.states[state][0]
                taxi_row, taxi_col = self.position_to_coordinates[taxi_loc]
                passengers = self.states[state][1:]

                #Increase the capacity penalty to see behavioral changes in the agent.
                #The agent will start giving more preference to maintaining optimum capacity
                #than to drop the passenger
                reward = - 1 - (taxi.get_max_capacity() - taxi.get_no_of_passengers()) * 5
                
                if action == 0: #move south
                    taxi_row = min(taxi_row + 1, self.no_of_rows - 1)
                elif action == 1: #move north
                    taxi_row = max(taxi_row - 1, 0)
                elif action == 2 and self.desc[1 + taxi_row, 2 * taxi_col + 2] == b":": #move east
                    taxi_col = min(taxi_col + 1, self.no_of_cols - 1)
                elif action == 3 and self.desc[1 + taxi_row, 2 * taxi_col] == b":": #move_west
                    taxi_col = max(taxi_col - 1, 0)
                elif action == 4: #Pickup
                    reward, passengers = self._pickup_passengers(state, reward, taxi_loc)
                elif action == 5: #Drop
                    reward, passengers, done = self._drop_passengers(state, reward, taxi_loc)

                taxi_loc   = self.coordinates_to_position[(taxi_row, taxi_col)]
                pass1_loc  = passengers[0].get_location()
                pass1_dest = passengers[0].get_destination()
                pass2_loc  = passengers[1].get_location()
                pass2_dest = passengers[1].get_destination()

                next_state = self.encode(taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest)                      
                next_state_no = self.state_to_no[next_state]
                self.P[self.state_to_no[state]][action].append((1.0, next_state_no, reward, done))
                if done == True:
                    print('state action next_state reward', state, action, next_state, reward)

    def _is_state_valid(self, state):
        taxi_loc, pass1_loc, pass1_dest, pass2_loc, pass2_dest = self._get_pos_from_state(state)
        taxi, pass1, pass2 = self.states[state]
        state_valid = True
        dropped = [p.get_passenger_id() for p in taxi.get_passengers_dropped()]

        if pass1_loc == pass1_dest == pass2_loc == pass2_dest:
            state_valid = False
        if (pass1_loc == pass1_dest) or (pass2_loc == pass2_dest):
            state_valid = False

        return state_valid

    def _get_pos_from_state(self, state):

        taxi, pass1, pass2 = self.states[state]

        taxi_loc = taxi.get_location()
        pass1_loc = pass1.get_location()
        pass1_dest = pass1.get_destination()
        pass2_loc = pass2.get_location()
        pass2_dest = pass2.get_destination()

        return taxi_loc, pass1_loc, pass1_dest, pass2_loc, pass2_dest

    def create_visual_grid(self, no_of_rows, no_of_cols):
        """
        Visualization based on the two dimensional grid size passed
            - number of rows 
            - number of columns 
        Adds random pipes - '|' signifying hurdles that the agent cannot cross directly

        """
        visual_grid = []
        locations = [' ' for _ in range(no_of_cols)]

        first_and_last_row = '+' + '-' * ((no_of_cols * 2) - 1) + '+'

        visual_grid.append(first_and_last_row)

        for row_num in range(no_of_rows):
            row = '|' + ':'.join(locations) + '|'
            row_l = list(row)
            indices = [idx for idx, elm in enumerate(row_l) if elm == ':']

            if row_num % 2 == 0: row_l[random.choice(indices)] = '|'

            row = ''.join(row_l)

            visual_grid.append(row)

        visual_grid.append(first_and_last_row)
        
        return visual_grid
    
    def _drop_passengers(self, state, reward, taxi_loc):
        done = False
        taxi, passengers = copy.deepcopy(self.states[state][0]), copy.deepcopy(self.states[state][1:])
        passengers_in_taxi = [p.get_passenger_id() for p in taxi.get_passengers_in_taxi()]
        
        if taxi.get_no_of_passengers() > 0:
            for passenger in passengers:
                passenger_drop_loc = passenger.get_destination()
                passenger_id = passenger.get_passenger_id()
                if taxi_loc == passenger_drop_loc and passenger_id in passengers_in_taxi:
                    reward += 10
                    taxi.drop_passenger(passenger)
                    passenger.set_location(taxi_loc)
                    if taxi.get_no_of_passengers_dropped() == self.no_of_passengers:
                        reward += 10
                        done = True
                #elif taxi_loc != passenger_drop_loc and passenger_id in passengers_in_taxi: 
                #    passenger.set_location(taxi_loc)

        if len(passengers_in_taxi) == len(taxi.get_passengers_in_taxi()):
            reward -= 10

        return reward, passengers, done

    def _pickup_passengers(self, state, reward, taxi_loc):
        taxi, passengers = copy.deepcopy(self.states[state][0]), copy.deepcopy(self.states[state][1:])
        board_passenger  = taxi.get_no_of_passengers() < taxi.get_max_capacity()
        passengers_in_taxi = [p.get_passenger_id() for p in taxi.get_passengers_in_taxi()]
        passengers_dropped = [p.get_passenger_id() for p in taxi.get_passengers_dropped()]

        if board_passenger:
            for passenger in passengers:
                passenger_id = passenger.get_passenger_id()
                passenger_pickup_loc = passenger.get_location()

                board_passenger = taxi.get_no_of_passengers() < taxi.get_max_capacity()
                board_passenger = board_passenger and passenger_id not in passengers_in_taxi
                board_passenger = board_passenger and passenger_id not in passengers_dropped

                if taxi_loc ==  passenger_pickup_loc and board_passenger: 
                    taxi.pickup_passenger(passenger)
                    reward += 10 #pick up rewarded
                    passenger.set_location(self.position_in_taxi)

        if len(passengers_in_taxi) == len(taxi.get_passengers_in_taxi()):
            reward -= 10

        return reward, passengers
    
    def _get_states(self):
        states = {}
        state_to_no = {}
        no_to_state = {}
        state_count = 0
        no_of_dests = self.no_of_rows * self.no_of_cols
        no_of_locs  = no_of_dests + 1 #+ 1 for being in taxi

        #What a mess - think of a way to eliminate the nestedness
        for taxi_loc in range(no_of_dests):
            for pass1_loc in range(no_of_locs): 
                for pass2_loc in range(no_of_locs): 
                    for pass1_dest in range(no_of_dests):
                        for pass2_dest in range(no_of_dests):
                            #if pass1_loc == pass1_dest == pass2_loc == pass2_dest: continue
                            state = self.encode(taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest)

                            taxi  = Taxi(taxi_loc, 0, self.taxi_max_capacity)
                            pass1 = Passenger(1, pass1_loc, pass1_dest)
                            pass2 = Passenger(2, pass2_loc, pass2_dest)

                            if pass1_loc == self.position_in_taxi:
                                taxi.pickup_passenger(pass1)
                            if pass2_loc == self.position_in_taxi:
                                taxi.pickup_passenger(pass2)
                            if pass1_loc == pass1_dest:
                                taxi.pickup_passenger(pass1)
                                taxi.drop_passenger(pass1)
                            if pass2_loc == pass2_dest:
                                taxi.pickup_passenger(pass2)
                                taxi.drop_passenger(pass2)

                            states[state] = [taxi, pass1, pass2]

                            state_to_no[state] = state_count
                            no_to_state[state_count] = state

                            state_count = state_count + 1

        return states, state_to_no, no_to_state

    def encode(self, taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest):
        """
        Returns a string that reprsents a unique state based on the following variables
            - taxi location
            - first passenger's starting location
            - second passenger's starting location
            - first passenger's destination location
            - second passenger's destination location

        Location is an integer that points to a specific (row,col) position in the grid.
        For example, in a 2 by 2 grid with 4 different (row, col) combinations, 0 location
        translates to a (0, 0) cell in the grid
        """
        # (5) 5, 5, 4
        taxi_row, taxi_col = self.position_to_coordinates[taxi_loc]
        pass1_loc_row, pass1_loc_col = self.position_to_coordinates[pass1_loc]
        pass2_loc_row, pass2_loc_col = self.position_to_coordinates[pass2_loc]
        pass1_dest_row, pass1_dest_col = self.position_to_coordinates[pass1_dest]
        pass2_dest_row, pass2_dest_col = self.position_to_coordinates[pass2_dest]
        
        encoded_state  = str(taxi_row) + str(taxi_col)  
        encoded_state += str(pass1_loc_row)+ str(pass1_loc_col) 
        encoded_state += str(pass2_loc_row) + str(pass2_loc_col) 
        encoded_state += str(pass1_dest_row) + str(pass1_dest_col)
        encoded_state += str(pass2_dest_row) + str(pass2_dest_col)

        return encoded_state

    def decode(self, state):
        """
        Returns a list with the following based on the state passed in as a parameter
            - taxi location
            - first passenger's starting location
            - second passenger's starting location
            - first passenger's destination location
            - second passenger's destination location

        Location is an integer that points to a specific (row,col) position in the grid.
        For example, in a 2 by 2 grid with 4 different (row, col) combinations, 0 location
        translates to a (0, 0) cell in the grid
        """

        positions = [int(position) for position in state]
        
        taxi_loc  = self.coordinates_to_position[(positions[0], positions[1])]
        pass1_loc = self.coordinates_to_position[(positions[2], positions[3])]
        pass2_loc = self.coordinates_to_position[(positions[4], positions[5])]
        pass1_dest = self.coordinates_to_position[(positions[6], positions[7])]
        pass2_dest = self.coordinates_to_position[(positions[8], positions[9])]
        
        return taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest

    def render(self, mode='human'):
        """
        Visualization in conjunction with create_visual_grid
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        snapshot = ""

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]

        for position in self.position_to_coordinates:
            if position != self.position_in_taxi:
                row, col = self.position_to_coordinates[position] 
                out[row + 1][2 * col + 1] = utils.colorize(str(position), 'yellow', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        state_objs = self.states[self.no_to_state[self.s]]
        taxi, passengers = state_objs[0], state_objs[1:]
        taxi_row, taxi_col = self.position_to_coordinates.get(taxi.get_location())

        for passenger in passengers:
            passenger_id = passenger.get_passenger_id()
            if passenger.get_location() == self.position_in_taxi: #in Taxi
                snapshot += "Passenger " + str(passenger_id) + " in taxi with " 
                snapshot += str(passenger_id) + " as destination\n"
            else:
                snapshot += "Passenger " + str(passenger) + " at " + str(passenger.get_location())
                snapshot += " with " + str(passenger.get_destination()) + " as destination\n"

        if taxi.get_no_of_passengers() == taxi.get_max_capacity():
            snapshot += "Taxi fully occupied at " + str(taxi.get_location())
        else:
            snapshot += "Taxi not fully occupied at " + str(taxi.get_location())

        outfile.write(snapshot)
