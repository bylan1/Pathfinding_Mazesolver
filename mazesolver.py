import sys
import math
from collections import deque
from itertools import count
import heapq
from inputhandler import *

# To run code, type in command line: python mazesolver.py [path_name] [method (optional)]
# path_name must be a image file stored in paths folder within mazesolver (mazesolver/paths/)
# For simple testing sake, run: python mazesolver.py path-1

### From pathfinder.py
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


def path_creation(path, preproc_list, base_image):
    """Rescales the path in list form to the base image size
    Converts the path list into a fitting application for the base image
    Paints the path on the base image

    Inputs:
        path: list of tuples (x,y) defining path in list image
        preproc_list: preprocessed image in list form
        base_image: original image of the maze

    Returns:
        out: original image of the maze with a drawn path from start to end points
    """

    pre_width = len(preproc_list[0])
    pre_length = len(preproc_list)

    base_width = base_image.shape[0]
    base_length = base_image.shape[1]

    


# Choose a particular path, i.e. path-1 and input
maze_file = sys.argv[1]
test_image = load(f'paths/{maze_file}.png')

preproc_img, coords = preprocess(test_image, 0.5)

display(preproc_img, 'Preprocessed path')

image_list = preproc_img.tolist()

problem = Problem(coords[0], coords[1], image_list)

results = astar(problem, "euclidean")

# Processes completed in path_creation function
# for x, y in results[0]:
#     image_list[x][y] = 1000

# final_image = np.array(image_list)

# display(final_image, 'Completed path using BFS')