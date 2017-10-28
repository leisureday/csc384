# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentPos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        height = newFood.height
        width = newFood.width
        "*** YOUR CODE HERE ***"
        # want to minimize the minFoodDist and not run into ghost
        minFoodDist = float('inf')
        minGhostDist = float('inf')
        
        # if there is food on newPos, set minFoodDist = 0, otherwise find minFoodDist using loop
        if currentFood[newPos[0]][newPos[1]] is True:
            minFoodDist = 0
        else:
            for x in range(width):
                for y in range(height):
                    if newFood[x][y] is True:
                        newFoodDist = manhattanDistance(newPos, (x, y))
                        if newFoodDist < minFoodDist:
                            minFoodDist = newFoodDist
                        
        for ghostPosition in newGhostPositions:
            newGhostDist = manhattanDistance(newPos, ghostPosition)
            if newGhostDist < minGhostDist:
                minGhostDist = newGhostDist
       
        # if run into ghost in next move, set score to -infinity
        if minGhostDist <= 1:
            return -float('inf')
        return -minFoodDist
        
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


# q2 helper function
def DFMiniMax(self, gameState, agentIndex, depth):
    bestAction = None
    # depth bound and terminal check
    if gameState.isWin() or gameState.isLose() or depth >= self.depth:
        return bestAction, self.evaluationFunction(gameState)
    if agentIndex == 0: # pacman: max player
        score = -float('inf')
    else: # ghosts: min player
        score = float('inf')
    legalActions = gameState.getLegalActions(agentIndex)
    for action in legalActions:
        # get successor gameState, agentIndex, depth
        # after last ghost, agentIndex goes back to 0
        # everytime agentIndex hits 0, depth ++
        succesorGameState = gameState.generateSuccessor(agentIndex, action)
        succesorAgentIndex = agentIndex + 1
        if succesorAgentIndex >= gameState.getNumAgents():
            succesorAgentIndex = 0
        succesorDepth = depth
        if succesorAgentIndex == 0:
            succesorDepth += 1
        # compute minimax action and score
        succesorAction, succesorScore = DFMiniMax(self, succesorGameState, succesorAgentIndex, succesorDepth)
        if agentIndex == 0 and score < succesorScore:
            bestAction, score = action, succesorScore
        if agentIndex != 0 and score > succesorScore:
            bestAction, score = action, succesorScore
    return bestAction, score

    
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        action, score = DFMiniMax(self, gameState, 0, 0)
        return action


# q3 helper function
def AlphaBeta(self, gameState, agentIndex, depth, alpha, beta):
    bestAction = None
    # depth bound and terminal check
    if gameState.isWin() or gameState.isLose() or depth >= self.depth:
        return bestAction, self.evaluationFunction(gameState)
    if agentIndex == 0: # pacman: max player
        score = -float('inf')
    else: # ghosts: min player
        score = float('inf')
    legalActions = gameState.getLegalActions(agentIndex)
    for action in legalActions:
        # get successor gameState, agentIndex, depth
        # after last ghost, agentIndex goes back to 0
        # everytime agentIndex hits 0, depth ++
        succesorGameState = gameState.generateSuccessor(agentIndex, action)
        succesorAgentIndex = agentIndex + 1
        if succesorAgentIndex >= gameState.getNumAgents():
            succesorAgentIndex = 0
        succesorDepth = depth
        if succesorAgentIndex == 0:
            succesorDepth += 1
        # compute minimax action and score
        # do alpha beta pruning
        succesorAction, succesorScore = AlphaBeta(self, succesorGameState, succesorAgentIndex, succesorDepth, alpha, beta)
        if agentIndex == 0:
            if score < succesorScore:
                bestAction, score = action, succesorScore
            if score >= beta:
                return bestAction, score
            alpha = max(alpha, score)
        if agentIndex != 0:
            if score > succesorScore:
                bestAction, score = action, succesorScore
            if score <= alpha:
                return bestAction, score
            beta = min(beta, score)
    return bestAction, score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = AlphaBeta(self, gameState, 0, 0, -float('inf'), float('inf'))
        return action
        
        
# q3 helper function
def Expectimax(self, gameState, agentIndex, depth):
    bestAction = None
    # depth bound and terminal check
    if gameState.isWin() or gameState.isLose() or depth >= self.depth:
        return bestAction, self.evaluationFunction(gameState)
    if agentIndex == 0: # pacman: max player
        score = -float('inf')
    else: # ghosts: chance player
        score = 0
    legalActions = gameState.getLegalActions(agentIndex)
    numLegalActions = len(legalActions)
    for action in legalActions:
        # get successor gameState, agentIndex, depth
        # after last ghost, agentIndex goes back to 0
        # everytime agentIndex hits 0, depth ++
        succesorGameState = gameState.generateSuccessor(agentIndex, action)
        succesorAgentIndex = agentIndex + 1
        if succesorAgentIndex >= gameState.getNumAgents():
            succesorAgentIndex = 0
        succesorDepth = depth
        if succesorAgentIndex == 0:
            succesorDepth += 1
        # compute expectimax action and score
        # no bestAction for ghost (chance player)
        succesorAction, succesorScore = Expectimax(self, succesorGameState, succesorAgentIndex, succesorDepth)
        if agentIndex == 0 and score < succesorScore:
            bestAction, score = action, succesorScore
        if agentIndex != 0:
            score = score + (1/float(numLegalActions))*succesorScore
    return bestAction, score

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = Expectimax(self, gameState, 0, 0)
        return action
    
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # try using BFS to find the distances instead of using manhattanDistance
    pacmanPosition = currentGameState.getPacmanPosition()    
    food = currentGameState.getFood()
    capsulesPostions = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()    
    ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    height = food.height
    width = food.width
    score = currentGameState.getScore()
    # want to minimize the minFoodDist and not run into ghost
    minFoodDist = float('inf')
    minGhostDist = float('inf')
    minCapsuleDist = float('inf')
    
    for x in range(width):
        for y in range(height):
            if food[x][y] is True:
                foodDist = manhattanDistance(pacmanPosition, (x, y))
                if foodDist < minFoodDist: minFoodDist = foodDist
                
    for capsulesPosition in capsulesPostions:
        capsuleDist = manhattanDistance(pacmanPosition, capsulesPosition)
        if capsuleDist < minCapsuleDist: minCapsuleDist = capsuleDist
            
    for ghostPosition in ghostPositions:
        currentGhostDist = manhattanDistance(pacmanPosition, ghostPosition)
        if currentGhostDist < minGhostDist: minGhostDist = currentGhostDist
   
    # if run into ghost in next move, set score to be small
    if ScaredTimes[0] == 0 and minGhostDist <= 1:
        score = score - 1000
    score = score + 1/float(minFoodDist)
    
    return score

# Abbreviation
better = betterEvaluationFunction

