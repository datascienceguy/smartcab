import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.q = {}
        self.possibleActions = [None, 'left', 'right', 'forward']
        self.gamma = 0.15
        self.epsilon = 0.15
        self.alpha = 0.2
        self.rewardTotalForTrip = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.rewardTotalForTrip = 0

        # print self.q

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.getState(inputs, self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.selectAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewardTotalForTrip += reward

        # TODO: Learn policy based on state, action, reward
        newInputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        nextState = self.getState(newInputs, self.next_waypoint)

        self.learnPolicy(self.state, action, reward, nextState)

        print "Total Reward:{}".format(self.rewardTotalForTrip)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getState(self, inputs, next_waypoint):
        return (inputs['light'], inputs['left'], inputs['oncoming'], next_waypoint)

    def selectAction(self, state):
        # In order to explore, if random value is less than epsilon,
        # choose random action instead of maxQ action
        shouldChooseRandomAction = random.random() < self.epsilon
        if shouldChooseRandomAction:
            action = random.choice(self.possibleActions)
        else:
            q = [self.q.get((state, a)) for a in self.possibleActions]
            i = q.index(max(q))
            action = self.possibleActions[i]

        return action

    def learnPolicy(self, state, action, reward, nextState):
        oldQValue = self.q.get((state, action))
        if oldQValue is None:
            self.q[(state, action)] = reward
        else:
            maxQ = max([self.q.get((nextState, a), 0.0) for a in self.possibleActions])
            learningRate = reward + self.gamma * maxQ
            self.q[(state, action)] = oldQValue + self.alpha * (learningRate - oldQValue)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
