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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        walls = successorGameState.getWalls()
        for idx, wall in enumerate(walls):
            continue
        perimeter = idx+len(walls[0]) # Height and Width of the grid

        if newPos == currentGameState.getPacmanPosition():
            stayPenalty = -100
        else:
            stayPenalty = 0

        if foodList:
            foodDist = [manhattanDistance(food, newPos) for food in foodList]
            foodDist.sort()
            foodBonus = 2/foodDist[0]
            if len(foodDist) > 1:
                secFoodDist = foodDist[1]
                foodBonus += 1/secFoodDist
        else:
            foodBonus = 0

        ghostDist = [manhattanDistance(ghost, newPos) for ghost in successorGameState.getGhostPositions()]
        ghostPenalty = 0
        numThreats = 0
        for dis in ghostDist:
            if dis < 2:
                numThreats += 1
                ghostPenalty += -perimeter
        ghostPenalty = ghostPenalty/numThreats if numThreats is not 0 else 0

        eatGhostBonus = 0
        for ghostIdx, scaredTime in enumerate(newScaredTimes):
            currGhostPos = successorGameState.getGhostPosition(ghostIdx+1)
            currGhostDis = manhattanDistance(currGhostPos, currentGameState.getPacmanPosition())
            if scaredTime > currGhostDis:
                eatGhostBonus = 5*perimeter-manhattanDistance(currGhostPos, newPos)

        score = successorGameState.getScore() + stayPenalty + foodBonus + ghostPenalty + eatGhostBonus
        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def minmax(state, agentIdx, currLayer):
            nonlocal numAgents
            if currLayer == (self.depth)*numAgents or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIdx)
            if len(actions) == 0:
                return self.evaluationFunction(state)

            sucStates = [state.generateSuccessor(agentIdx, action) for action in actions]

            if agentIdx == 0:
                return max([minmax(suc, agentIdx+1, currLayer+1) for suc in sucStates])

            if agentIdx == numAgents-1:
                return min([minmax(suc, 0, currLayer+1) for suc in sucStates])

            return min([minmax(suc, agentIdx+1, currLayer+1) for suc in sucStates])

        return max(gameState.getLegalActions(0), key=lambda act: minmax(gameState.generateSuccessor(0,act), 1, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def abminmax(state, agentIdx, currLayer, alpha, beta):
            nonlocal numAgents
            if currLayer == (self.depth)*numAgents or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIdx)
            if len(actions) == 0:
                return self.evaluationFunction(state)

            # Max-value
            if agentIdx == 0:
                val = float("-inf")
                for act in actions:
                    val = max(val, abminmax(state.generateSuccessor(agentIdx, act), agentIdx+1, currLayer+1, alpha, beta))
                    if val > beta:
                        return val
                    alpha = max(alpha, val)
                return val

            if agentIdx == numAgents-1:
                nextAgentIdx = 0
            else:
                nextAgentIdx = agentIdx+1

            # Min-value
            val = float("inf")
            for act in actions:
                val = min(val, abminmax(state.generateSuccessor(agentIdx, act), nextAgentIdx, currLayer+1, alpha, beta))
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val

        alpha = float("-inf")
        beta = float("inf")
        actions = gameState.getLegalActions(0)
        path = 'Stop'
        for act in actions:
            val = abminmax(gameState.generateSuccessor(0, act), 1, 1, alpha, beta)
            if val > alpha:
                alpha, path = val, act
        return path

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
        numAgents = gameState.getNumAgents()
        def expectiminmax(state, agentIdx, currLayer):
            nonlocal numAgents
            if currLayer == (self.depth)*numAgents or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIdx)
            if len(actions) == 0:
                return self.evaluationFunction(state)

            sucStates = [state.generateSuccessor(agentIdx, action) for action in actions]

            if agentIdx == 0:
                return max([expectiminmax(suc, agentIdx+1, currLayer+1) for suc in sucStates])

            if agentIdx == numAgents-1:
                nextAgentIdx = 0
            else:
                nextAgentIdx = agentIdx+1;

            sucValues = [expectiminmax(suc, nextAgentIdx, currLayer+1) for suc in sucStates]
            return sum(sucValues)/len(sucValues)

        return max(gameState.getLegalActions(0), key=lambda act: expectiminmax(gameState.generateSuccessor(0,act), 1, 1))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1. Normalize the scale of bonus and penalty using perimeter of game's grid
    2. Give little bonus to the nearest two food
    3. Escape only when the ghost is just next to the Pacman
    4. Incentivize the Pacman to eat the scared ghost
    """
    "*** YOUR CODE HERE ***"
    # Define a function to evaluate each action
    def evaluate(state, action):
        nonlocal currentGameState
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodList = newFood.asList()
        walls = successorGameState.getWalls()
        for idx, wall in enumerate(walls):
            continue
        # Height and Width of the wall
        # Use perimeter to normalize the scale of bonus and penalty
        perimeter = idx+len(walls[0])

        # EatFood bonus is already included in game.
        # We only need to give a little bonus to the nearest one,
        # and also a little little bonus to the second nearest one
        # to avoid Pacman from stuck
        if foodList:
            foodDist = [manhattanDistance(food, newPos) for food in foodList]
            foodDist.sort()
            foodBonus = 2/foodDist[0]
            if len(foodDist) > 1:
                foodBonus += 1/foodDist[1]
        else:
            foodBonus = 0

        # Act aggressively - only when the ghost is just next to the Pacman 
        # will the Pacman considered the ghost a threat
        ghostDist = [manhattanDistance(ghost, newPos) for ghost in successorGameState.getGhostPositions()]
        ghostPenalty = 0
        numThreats = 0
        for dis in ghostDist:
            if dis < 2:
                numThreats += 1
                ghostPenalty += -2*perimeter
        ghostPenalty = ghostPenalty/numThreats if numThreats else 0

        # Give bonus to the Pacman as an incentive to eat the scared ghost
        eatGhostBonus = 0
        for ghostIdx, scaredTime in enumerate(newScaredTimes):
            currGhostPos = successorGameState.getGhostPosition(ghostIdx+1)
            currGhostDis = manhattanDistance(currGhostPos, currentGameState.getPacmanPosition())
            if scaredTime > currGhostDis:
                eatGhostBonus = 5*perimeter-manhattanDistance(currGhostPos, newPos)

        score = successorGameState.getScore() + foodBonus + ghostPenalty + eatGhostBonus
        return score

    actions = currentGameState.getLegalActions()
    if len(actions) == 0:
        return currentGameState.getScore()

    return max([evaluate(currentGameState, action) for action in actions])

# Abbreviation
better = betterEvaluationFunction
