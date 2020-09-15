#!/usr/bin/env python
# coding: utf-8

import copy
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from numpy import random
import gym

import random

class Passenger():

    def __init__(self, passenger_id, location, destination):
        self.location = location
        self.destination = destination
        self.passenger_id = passenger_id

    def get_passenger_id(self):

        return self.passenger_id

    def get_location(self):

        return self.location

    def set_location(self, location):

        self.location = location

        return True

    def get_destination(self):

        return self.destination

    def set_destination(self, destination):

        self.destination = destination

        return True

class Taxi():

    def __init__(self, location, no_of_passengers, max_capacity):
        self.location = location
        self.passengers_in_taxi = []
        self.passengers_dropped = []
        self.max_capacity = max_capacity
        self.no_of_passengers = no_of_passengers

    def get_passengers_in_taxi(self):

        return self.passengers_in_taxi

    def pickup_passenger(self, passenger):

        self.passengers_in_taxi.append(passenger)
        self.no_of_passengers = len(self.passengers_in_taxi)

        return True

    def drop_passenger(self, passenger):

        self._drop_passenger(passenger)
        self.passengers_dropped.append(passenger)
        self.no_of_passengers = len(self.passengers_in_taxi)

        return True

    def _drop_passenger(self, passenger):
        passenger_index = 0

        passenger_to_drop = passenger.get_passenger_id()

        for passenger in self.passengers_in_taxi:
            if passenger.get_passenger_id() != passenger_to_drop:
                passenger_index = passenger_index + 1
            else:
                break

        del self.passengers_in_taxi[passenger_index]

        return True

    def get_no_of_passengers_dropped(self):

        return len(self.passengers_dropped)

    def get_location(self):

        return self.location

    def set_location(self, location):

        self.location = location

        return True

    def get_max_capacity(self):

        return self.max_capacity

    def set_max_capacity(self, max_capacity):

        self.max_capacity = max_capacity

        return True

    def get_no_of_passengers(self):

        return self.no_of_passengers

    def set_no_of_passengers(self, no_of_passengers):

        self.no_of_passengers = no_of_passengers

        return True

    def get_passengers_dropped(self):

        return self.passengers_dropped


class TaxiCapacityEnv(discrete.DiscreteEnv): #Playground
    def __init__(self, no_of_rows, no_of_cols, no_of_actions, max_passengers):
        P = {}
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
        self.initial_state_distribution = np.zeros(len(self.states))

        state_keys = list(self.states.keys())
        random.shuffle(state_keys)
        
        for state in state_keys:

            P[self.state_to_no[state]] = {action : [] for action in range(self.no_of_actions)}

            if self._is_state_valid(state):
                self.initial_state_distribution[self.state_to_no[state]] += 1
            
            for action in range(self.no_of_actions):
                taxi_loc, pass1_loc, pass1_dest, pass2_loc, pass2_dest = self._get_pos_from_state(state)

                done = False
                taxi = self.states[state][0]
                taxi_row, taxi_col = self.position_to_coordinates[taxi_loc]
                passengers = self.states[state][1:]
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

                taxi_loc = self.coordinates_to_position[(taxi_row, taxi_col)]
                pass1_loc = passengers[0].get_location()
                pass1_dest = passengers[0].get_destination()
                pass2_loc = passengers[1].get_location()
                pass2_dest = passengers[1].get_destination()

                next_state = self.encode(taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest)                      
                next_state_no = self.state_to_no[next_state]
                P[self.state_to_no[state]][action].append((1.0, next_state_no, reward, done))
                if done == True:
                    print('state action next_state reward', state, action, next_state, reward)

        self.initial_state_distribution /= self.initial_state_distribution.sum()
        discrete.DiscreteEnv.__init__(
            self, len(self.states), self.no_of_actions, P, self.initial_state_distribution)

    def _is_state_valid(self, state):
        taxi_loc, pass1_loc, pass1_dest, pass2_loc, pass2_dest = self._get_pos_from_state(state)
        taxi, pass1, pass2 = self.states[state]
        state_valid = True
        dropped = [p.get_passenger_id() for p in taxi.get_passengers_dropped()]

        if pass1_loc == pass1_dest == pass2_loc == pass2_dest:
            state_valid = False
        if (pass1_loc == pass1_dest) or (pass2_loc == pass2_dest):
            state_valid = False
        #if (pass1.get_passenger_id() not in dropped) and (pass1_loc == pass1_dest):
        #    state_valid = False
        #if (pass2.get_passenger_id() not in dropped) and (pass2_loc == pass2_dest):
        #    state_valid = False

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

        positions = [int(position) for position in state]
        
        taxi_loc  = self.coordinates_to_position[(positions[0], positions[1])]
        pass1_loc = self.coordinates_to_position[(positions[2], positions[3])]
        pass2_loc = self.coordinates_to_position[(positions[4], positions[5])]
        pass1_dest = self.coordinates_to_position[(positions[6], positions[7])]
        pass2_dest = self.coordinates_to_position[(positions[8], positions[9])]
        
        return taxi_loc, pass1_loc, pass2_loc, pass1_dest, pass2_dest

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        snapshot = ""

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]

        for position in self.position_to_coordinates:
            if position != self.position_in_taxi:
                row, col = self.position_to_coordinates[position] 
                out[row + 1][2 * col + 1] = utils.colorize(str(position), 'yellow', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        #positions = self.decode(self.no_to_state[self.s]) #What is the purpose of decode
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

if __name__ == '__main__':
    env = TaxiCapacityEnv(2, 2, 6, 2)

    """
    initial_state = env.s = 701
    done = False

    while not done:
        action = env.action_space.sample()
        last_state = env.s
        state, reward, done, info = env.step(action)
        print('state, action, next_state, reward - ', last_state, action, state, reward)

    if done:
        print('Done for state - ', initial_state)
    for s in range(1600):
        initial_state = env.reset()
        done = False
        epochs = 0

        while not done:
            action = env.action_space.sample()
            last_state = env.s
            state, reward, done, info = env.step(action)

            epochs = epochs + 1
            if epochs % 100000 == 0:
                print('States inital->last->current: ', initial_state, last_state, state)
                break

        if done:
            print('Done for state - ', initial_state)

    epochs = 0
    penalties, reward = 0, 0

    done = False

    while not done:
        action = env.action_space.sample()
        old_state = env.s
        state, reward, done, info = env.step(action)

        if reward < 0:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        epochs += 1
        
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

    """
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 1000000):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward < 0:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100000 == 0:
            print(f"Episode: {i}")

    print("Training finished.\n")

    episodes = 1
    total_epochs, total_penalties = 0, 0

    for episode in range(episodes):
        initial_state = state = env.reset()
        env.render()
        #initial_state = state = 946
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            last_state = env.s
            state, reward, done, info = env.step(action)
            env.render()

            if reward < 0:
                penalties += 1

            epochs += 1
            if epochs % 100000 == 0:
                print(f"Epochs : {epochs}", "states initial->last->current", initial_state, last_state, state)

        if episode % 1000 == 0:
            print(f"Episode: {episode}")

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
