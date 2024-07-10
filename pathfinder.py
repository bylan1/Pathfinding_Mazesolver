import sys
import math
import heapq
from itertools import count
from collections import deque

# problem class
class Problem:
    def __init__(self, initial, goal, grid):
        self.initial = initial
        self.goal = goal
        self.grid = grid


    def goal_check(self, state):
        return self.goal == state

    def initial_state(self):
        return self.initial

    def get_cost(self, state):
        x, y = state
        return int(self.grid[x][y])

    def manhattan_heuristic(self, state):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def euclidean_heuristic(self, state):
        return math.sqrt((state[0] - self.goal[0])**2 + (state[1] - self.goal[1])**2)

    def expand(self, state):
        successors = []
        x, y = state

        moveset = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        curr_cost = self.get_cost((x,y))

        for x_move, y_move in moveset:
            new_x, new_y = x + x_move, y + y_move

            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] != 'X':
                cost = (self.get_cost((new_x, new_y)) - curr_cost) if ((self.get_cost((new_x, new_y)) - curr_cost) > 0 or self.get_cost((new_x, new_y)) == 0) else 0
                successors.append(((new_x, new_y), cost))

        return successors

# processing of input file data
def data_creation(input):
    size = (int(input[0].split()[0]), int(input[0].split()[1]))
    initial_state = (int(input[1].split()[0])-1, int(input[1].split()[1])-1)
    end_state = (int(input[2].split()[0])-1, int(input[2].split()[1])-1)

    grid = []
    for row in input[3:3+size[0]]:
        grid.append([str(x) for x in row.split()])
    
    return size, initial_state, end_state, grid

# breadth first search
def bfs(problem):
    closed = set()
    fringe = deque([(problem.initial_state(), 0, [])])  # state, cost, path
    
    while fringe:
        state, total_cost, path = fringe.popleft()
        
        if problem.goal_check(state):
            return path + [state], total_cost
        
        if state not in closed:
            closed.add(state)
            successors = problem.expand(state)
            for successor, cost in successors:
                fringe.append((successor, total_cost + cost, path + [state]))
    
    return "failure", 0

# uniform cost search
def ucs(problem):
    closed = set()
    fringe = []
    heapq.heapify(fringe)
    counter = count()

    heapq.heappush(fringe, (0, next(counter), problem.initial_state(), [])) # cost, counter (for path priority), state, path
    
    while fringe:
        total_cost, _, state, path = heapq.heappop(fringe)
        
        if problem.goal_check(state):
            return path + [state], total_cost
        
        if state not in closed:
            closed.add(state)
            successors = problem.expand(state)
            for successor, cost in successors:
                heapq.heappush(fringe, (total_cost + cost, next(counter), successor, path + [state]))
    
    return "failure", 0

# a* search with different heuristics
def astar(problem, heuristic):
    closed = set()
    fringe = []
    heapq.heapify(fringe)
    counter = count()

    if heuristic == "manhattan":
        heapq.heappush(fringe, (0 + problem.manhattan_heuristic(problem.initial_state()), next(counter), 0, problem.initial_state(), [])) # g + h, counter, cost, state, path
        
        while fringe:
            _, _, total_cost, state, path = heapq.heappop(fringe)
            
            if problem.goal_check(state):
                return path + [state], total_cost
            
            if state not in closed:
                closed.add(state)
                successors = problem.expand(state)
                for successor, cost in successors:
                    heapq.heappush(fringe, (total_cost + cost + problem.manhattan_heuristic(successor), next(counter), total_cost + cost, successor, path + [state]))

    elif heuristic == "euclidean":
        heapq.heappush(fringe, (0 + problem.euclidean_heuristic(problem.initial_state()), next(counter), 0, problem.initial_state(), [])) # g + h, counter, cost, state, path
        
        while fringe:
            _, _, total_cost, state, path = heapq.heappop(fringe)
            
            if problem.goal_check(state):
                return path + [state], total_cost
            
            if state not in closed:
                closed.add(state)
                successors = problem.expand(state)
                for successor, cost in successors:
                    heapq.heappush(fringe, (total_cost + cost + problem.euclidean_heuristic(successor), next(counter), total_cost + cost, successor, path + [state]))

    return "failure", 0

if __name__ == "__main__":
    map_file = sys.argv[1]
    algorithm = sys.argv[2]
    
    f = open(map_file, 'r')

    content = f.readlines()

    size, init_state, goal_state, grid = data_creation(content)
    problem = Problem(init_state, goal_state, grid)
    
    if algorithm == 'bfs':
        results = bfs(problem)
    elif algorithm == 'ucs':
        results = ucs(problem)
    elif algorithm == 'astar':
        heuristic = sys.argv[3]
        results = astar(problem, heuristic)
    else:
        print("unknown algorithm")


    if results != "failure":
        try:
            for x, y in results[0]:
                grid[x][y] = '*'
            for row in grid:
                print(' '.join(row))
        except:
            print("null")
    else:
        print("null")

# function GRAPH-SEARCH (problem, fringe) returns a solution, or failure
#     closed <- an empty set
#     fringe <- INSERT(MAKE-NODE(INITIAL-STATE[problem]), fringe)
#     loop do
#         if fringe is empty then return failure
#         node <- REMOVE-FRONT(fringe)
#         if GOAL-TEST(problem, STATE[node]) then return node
#         if STATE[node] is not in closed then
#             add STATE[node] to closed
#             fringe <- INSERTALL(EXPAND(node, problem), fringe)
#     end