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
    return [s, s, w, s, w, w, s, w]

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
    # start code
    from util import Stack

    open = Stack()
    start = ((problem.getStartState(), None, 0),)
    open.push(start)
    while open:
        n = open.pop()
        lastStateTuplet = n[-1]
        state = lastStateTuplet[0]
        if problem.isGoalState(state):
            # extract directions from n
            directions = []
            for state, direction, cost in n:
                if direction is not None:
                    directions.append(direction)
            return directions
        else:
            sucessors = problem.getSuccessors(state)
            nStates = [i[0] for i in n]
            for succ in sucessors:
                succState = succ[0]
                if succState not in nStates:
                    open.push(n + (succ,))
    return []
    #  end code

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # start code
    from util import Queue

    open = Queue()
    start = ((problem.getStartState(), None, 0),)
    statesVisited = {problem.getStartState(): 0}
    open.push(start)
    while open:
        n = open.pop()
        nPathCost = sum([i[2] for i in n])
        lastStateTuplet = n[-1]
        state = lastStateTuplet[0]
        if nPathCost <= statesVisited[state]:
            if problem.isGoalState(state):
                # extract directions from n
                directions = []
                for state, direction, cost in n:
                    if direction is not None:
                        directions.append(direction)
                return directions
            else:
                sucessors = problem.getSuccessors(state)
                for succ in sucessors:
                    succState = succ[0]
                    succCost = succ[2]
                    succPathCost = nPathCost + succCost
                    if (succState not in statesVisited) or (succPathCost < statesVisited[succState]):
                        open.push(n + (succ,))
                        statesVisited[succState] = succPathCost
    return []
    #  end code

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # start code
    from util import PriorityQueue

    open = PriorityQueue()
    start = ((problem.getStartState(), None, 0),)
    statesVisited = {problem.getStartState(): 0}
    open.update(start, 0)
    while open:
        n = open.pop()
        nPathCost = sum([i[2] for i in n])
        lastStateTuplet = n[-1]
        state = lastStateTuplet[0]
        if nPathCost <= statesVisited[state]:
            if problem.isGoalState(state):
                # extract directions from n
                directions = []
                for state, direction, cost in n:
                    if direction is not None:
                        directions.append(direction)
                return directions
            else:
                sucessors = problem.getSuccessors(state)
                for succ in sucessors:
                    succState = succ[0]
                    succCost = succ[2]
                    succPathCost = nPathCost + succCost
                    if (succState not in statesVisited) or (succPathCost < statesVisited[succState]):
                        open.update(n + (succ,), succPathCost)
                        statesVisited[succState] = succPathCost
    return []
    #  end code

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # start code
    from util import PriorityQueue

    open = PriorityQueue()
    startState = problem.getStartState()
    start = ((startState, None, 0),)
    statesVisited = {startState: heuristic(startState, problem)}
    open.update(start, 0)
    while open:
        n = open.pop()
        nPathCost = sum([i[2] for i in n])
        lastStateTuplet = n[-1]
        state = lastStateTuplet[0]
        fPathCost = nPathCost + heuristic(state, problem)
        if fPathCost <= statesVisited[state]:
            if problem.isGoalState(state):
                # extract directions from n
                directions = []
                for state, direction, cost in n:
                    if direction is not None:
                        directions.append(direction)
                return directions
            else:
                sucessors = problem.getSuccessors(state)
                for succ in sucessors:
                    succState = succ[0]
                    succCost = succ[2]
                    succPathCost = nPathCost + succCost
                    succFPathCost = succPathCost + heuristic(succState, problem)
                    if (succState not in statesVisited) or (succFPathCost < statesVisited[succState]):
                        open.update(n + (succ,), succFPathCost)
                        statesVisited[succState] = succFPathCost
    return []
    #  end code


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
