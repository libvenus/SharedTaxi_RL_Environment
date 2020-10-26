# SharedTaxi - A Reinforcement Learning Environment
An Environment that simulates surroundings and conditions for a SharedTaxi to live-in, operate and learn.  The environment closely resembles a shared taxi with two passengers and helps in training autonomous agents. It is "generic" and adapts to the dimensions specified while initializing it. It balances taxi capacity and passenger(s) pickup/drop, however, preference is always given to passengers’ pickup and drop.  For simplicity, a grid is chosen as the physical representation of the environment with hurdles and diversions.

The environment simulates an episodic Markov decision process and ends once both the passengers are dropped at their respective destinations. It starts at an initial state and yields a subsequent state and reward every time it receives an action input. The state’s (a.k.a. observations) dimensions are controlled by the following entities’ location:

	1.Taxi’s location – (no of rows X no of cols). For example, in a 5 X 5 grid, the no of locations a taxi can be is 25  
	2.First passenger’s pickup location – (no of rows X no of cols) + 1 . 1 position extra as the passenger could also be in the taxi in the initial state 
	3.Second passenger’s pickup location – (no of rows X no of cols) + 1
	4.First passenger’s drop location – (no of rows X no of cols)
	5.Second passenger’s drop location - (no of rows X no of cols)

We can see below that the state space increases very rapidly and this reflects in the time the agent takes while exploring the environment. It is important to keep in mind  that more the number of  states, greater will the need for training as the agent will need more iterations to learn the environment dynamics.

	1.2 X 2 grid - 4 (possible taxi locations) * 5 (possible first passenger pickup locations) * 5 (possible first passenger pickup locations) * 4 (possible first passenger drop locations) * 4 (possible second passenger drop locations)  = 1600 states
	2.3 X 3 grid – 9  * 10  * 10 * 9 * 9  = 72900 states
	3.4 X 4 grid – 16 * 17 * 17 * 16 * 16  ~ 1 million states
	4.5 X 5 grid – 25 * 26 * 26 * 25 * 25  ~ 10 million states

## Code details
The entire functionality is abstracted into the following Python modules:

	1. sharedtaxi.py - Contains the main SharedTaxi class that acts as the playground for the SharedTaxi
	2. taxi.py - Encapsulates the taxi and related operations
	3. passenger.py - Encapsulates the passenger and related operations
	4. sharedtaxi_test.py - The driver/wrapper that initializes the SharedTaxi envrionment, helps the agent interact with the environment and learn
	
## Todo

1. Visualization - represent and render agent's interaction with environment and vice-versa pictorially.
2. Performance - replace table based Q-learning with Deep Neural network based as that will help perform and scale better
3. Dynamics - improve environment dynamics and make the environment a closer approximation to its real world counterpart
