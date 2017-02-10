import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from random import randint

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.alpha = 0.4
        self.gamma = 0.7 
        self.Q_table = {}
        for waypoint in ['left', 'right', 'forward']:
            for light in ['red', 'green']:
                for oncoming in self.env.valid_actions:
                    for left in self.env.valid_actions:
                        for right in self.env.valid_actions:
                             for action in self.env.valid_actions:
                                   self.Q_table[((waypoint, light, oncoming, left, right), action)] = 3


    def reset(self, destination=None):
        self.planner.route_to(destination)
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        ## action  = random.choice(self.env.valid_actions) # random choose action
        action = None 
        if randint(0,10) < 1:
            action = self.env.valid_actions[randint(0, 3)]
        else:
            Q_list = [self.Q_table[(self.state, act)] for act in self.env.valid_actions]        
            index = Q_list.index(max(Q_list))
            action = self.env.valid_actions[index]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward    
        inputs_next = self.env.sense(self)   
        next_waypoint_next = self.planner.next_waypoint()
        state_next = (next_waypoint_next, inputs_next['light'], inputs['oncoming'], inputs['left'], inputs['right']) 
        max_Q = max([self.Q_table[(state_next, act)] for act in self.env.valid_actions])
        ## update Q_table
        self.Q_table[(self.state, action)] = (1 - self.alpha) * self.Q_table[(self.state, action)] + self.alpha * (reward + self.gamma * max_Q)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        # print self.Q_table   # [debug]
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
