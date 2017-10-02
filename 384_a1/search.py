# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # use stack to implement DFS
    open = util.Stack()
    # implement node as a tuple of succ tuples
    start_state = problem.getStartState()
    node = ((start_state, None, 0),)    
    open.push(node)
    
    while not open.isEmpty():
        current_node = open.pop()
        current_state = current_node[-1][0]
        #get list of states and actions
        states = []
        actions = []
        for state, action, cost in current_node:
            states.append(state)
            if action is not None:
                actions.append(action)
        
        if problem.isGoalState(current_state):
            return actions
        
        for succ in problem.getSuccessors(current_state):
            #path checking
            if not succ[0] in states:
                open.push(current_node + (succ,))
                
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # use Queue to implement DFS
    open = util.Queue()
    # implement node as a tuple of succ tuples
    start_state = problem.getStartState()
    node = ((start_state, None, 0),)
    open.push(node)
    # seen is implemented for cycle checking
    seen = {start_state:0}
    
    while not open.isEmpty():
        current_node = open.pop()
        current_state = current_node[-1][0]
        actions = []
        #get list of actions
        for state, action, cost in current_node:
            if action is not None:
                actions.append(action)   
                
        current_cost = problem.getCostOfActions(actions)
        # cost checking for cycle checking
        if current_cost <= seen[current_state]:
            if problem.isGoalState(current_state):
                return actions
            
            sucessors = problem.getSuccessors(current_state)
            for succ in sucessors:
                #cycle checking
                next_cost = current_cost + succ[2]
                if (not succ[0] in seen) or (next_cost < seen[succ[0]]):
                    open.push(current_node + (succ,))
                    seen[succ[0]] = next_cost
                
    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # use PriorityQueue to implement UCS, PriorityQueue pop out item with lowest priority
    open = util.PriorityQueue()
    # implement node as a tuple of succ tuples
    start_state = problem.getStartState()
    node = ((start_state, None, 0),)
    open.push(node, 0)
    # seen is implemented for cycle checking
    seen = {start_state:0}
    
    while not open.isEmpty():
        current_node = open.pop()
        current_state = current_node[-1][0]
        actions = []
        #get list of actions
        for state, action, cost in current_node:
            if action is not None:
                actions.append(action)   
                
        current_cost = problem.getCostOfActions(actions)
        # cost checking for cycle checking
        if current_cost <= seen[current_state]:
            if problem.isGoalState(current_state):
                return actions
            
            sucessors = problem.getSuccessors(current_state)
            for succ in sucessors:
                #cycle checking
                next_cost = current_cost + succ[2]
                if (not succ[0] in seen) or (next_cost < seen[succ[0]]):
                    open.push(current_node + (succ,), next_cost)
                    seen[succ[0]] = next_cost
                
    return False


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # use PriorityQueue to implement A*, PriorityQueue pop out item with lowest priority
    open = util.PriorityQueue()
    # implement node as a tuple of succ tuples
    start_state = problem.getStartState()
    node = ((start_state, None, 0),)
    open.push(node, heuristic(start_state, problem))
    # seen is implemented for cycle checking
    seen = {start_state:0}
    
    while not open.isEmpty():
        current_node = open.pop()
        current_state = current_node[-1][0]
        actions = []
        #get list of actions
        for state, action, cost in current_node:
            if action is not None:
                actions.append(action)   
                
        current_cost = problem.getCostOfActions(actions)
        # cost checking for cycle checking
        if current_cost <= seen[current_state]:
            if problem.isGoalState(current_state):
                return actions
            
            sucessors = problem.getSuccessors(current_state)
            for succ in sucessors:
                #cycle checking
                next_cost = current_cost + succ[2]
                if (not succ[0] in seen) or (next_cost < seen[succ[0]]):
                    open.push(current_node + (succ,), next_cost + heuristic(succ[0], problem))
                    seen[succ[0]] = next_cost
                
    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
