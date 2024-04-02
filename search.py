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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    searchStack = util.Stack()
    visit_set = set()
    searchStack.push((problem.getStartState(), []))
    stack_len = 1  # Initialize the fringe length

    while stack_len > 0:  # Check the fringe length
        node = searchStack.pop()
        moves = node[1]
        stack_len -= 1  # Decrement the fringe length
        if node[0] in visit_set:
            continue
        else:
            visit_set.add(node[0])
            if problem.isGoalState(node[0]):
                return moves
            else:
                successors = problem.getSuccessors(node[0])
                i = 0
                while i < len(successors):
                    newState, newMove, newCost = successors[i]
                    searchStack.push((newState, (moves + [newMove])))
                    stack_len += 1  # Increment the fringe length
                    i += 1






def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    """
    "*** YOUR CODE HERE ***"
    from util import Queue
    searchQueue = Queue()
    visit_set = set()
    searchQueue.push((problem.getStartState(), []))
    queueLen = 1  # Initialize the fringe length

    while queueLen > 0:
        node = searchQueue.pop()
        moves = node[1]
        queueLen -= 1
        if node[0] in visit_set:
            continue
        else:
            visit_set.add(node[0])
            if problem.isGoalState(node[0]):
                return moves
            else:
                successors = problem.getSuccessors(node[0])
                i = 0
                while i < len(successors):
                    nexState, newmove, newcost = successors[i]
                    searchQueue.push((nexState, (moves + [newmove])))
                    queueLen +=1
                    i += 1



def uniformCostSearch(problem):
    """
    Search the node of the least total cost first.
    """
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    searchPQ = PriorityQueue()
    visit_dict = {}  # Use a dictionary to track visited states and costs
    searchPQ.push(( problem.getStartState(), [], 0), 0)
    pqueue_len = 1  # Initialize the fringe length

    while pqueue_len > 0:
        node = searchPQ.pop()
        current = node[0]
        moves = node[1]
        cost = node[2]
        pqueue_len -= 1
        if current in visit_dict and cost >= visit_dict[current]:
            continue
        else:
            visit_dict[current] = cost  # Update the cost for the visited state
            if problem.isGoalState(current):
                return moves
            else:
                successors = problem.getSuccessors(current)
                i = 0
                while i < len(successors):
                    newState, newMove, newCost = successors[i]
                    searchPQ.update((newState, (moves + [newMove]), (cost + newCost)), (cost + newCost))
                    pqueue_len += 1
                    i += 1





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    searchPQ = PriorityQueue()
    visit_dict = {}

    searchPQ.push(((problem.getStartState(), [], 0)), 0)
    pqueueLen = 1

    while pqueueLen > 0:
        node = searchPQ.pop()
        current = node[0]
        moves = node[1]
        cost = node[2]
        pqueueLen -= 1

        if current not in visit_dict or cost < visit_dict[current]:
            visit_dict[current] = cost
            if problem.isGoalState(current):
                return moves
            else:
                successors = problem.getSuccessors(current)
                i = 0
                while i < len(successors):
                    sucState, sucAction, sucCost = successors[i]

                    addn = (sucState, (moves + [sucAction]), (cost + sucCost))
                    priority = (cost + sucCost) + heuristic(sucState, problem)
                    searchPQ.update(addn, priority)
                    pqueueLen += 1
                    i += 1




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
