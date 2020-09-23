# SharedTaxi - A Reinforcement Learning Environment
The environment closely resembles a shared taxi with two passengers and helps in training autonomous agents. The environment is "generic" and adapts to the dimensions specified while initializing it. It balances taxi capacity and passenger(s) pickup/drop, however, preference is always given to a passengers’ pickup and drop.  

The environment simulates an episodic Markov decision process and ends once both the passengers are dropped at their respective destinations. It starts at an initial state and yields a subsequent state and reward every time it receives an action input. The state’s (a.k.a. observations) dimensions are controlled by the following entities’ location:

	1.	Taxi’s location – (no of rows X no of cols). For example, in a 5 X 5 grid, the no of locations a taxi can be is 25  
	2.	First passenger’s starting location – (no of rows X no of cols) + 1 . 1 position extra as the passenger could also be in the taxi in the initial state 
	3.	Second passenger’s starting location – (no of rows X no of cols) + 1
	4.	First passenger’s end location – (no of rows X no of cols)
	5.	Second passenger’s end location - (no of rows X no of cols)

We can see below that the state space increases very rapidly and this reflects in the time the agent takes while exploring the environment. It is important to keep in mind is that more the number of  states, greater will the need for training as the agent will need more iterations to learn the environment dynamics.

	1.	2 X 2 grid - 4 (possible taxi locations) * 5 (possible first passenger starting locations) * 5 (possible first passenger starting locations) * 4 (possible first passenger end locations) * 4 (possible second passenger locations)  = 1600 states
	2.	3 X 3 grid – 9  * 10  * 10 * 9 * 9  = 72900 states
	3.	4 X 4 grid – 16 * 17 * 17 * 16 * 16  ~ 1 million states
	4.	5 X 5 grid – 25 * 26 * 26 * 25 * 25  ~ 10 million states
