import sys
import os
import math
from collections import deque
from itertools import count
import heapq
from inputhandler import *
import matplotlib.pyplot as plt

# To run code, type in command line: python mazesolver.py <filename> <algorithm> <heuristic (opt)>
# filename must be a png image file stored in paths folder within mazesolver (mazesolver/paths/)
# algorithm must be one of "ucs" or "astar". For "astar" algorithm, include the heuristic of choice between "euclidean" or "manhattan" 
# For simple testing sake, run: python mazesolver.py path-1 ucs

# For saving the image as a file in paths folder
def save(img, caption, filename):
    os.makedirs('paths', exist_ok=True)
    image_prep(img, caption)

    file_loc = f'paths/{filename}'
    plt.savefig(file_loc, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

### From pathfinder.py: problem, ucs, astar
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

# For path visibility
def pixel_expand(space_list):
    """Explands list to contain tuples expanding initial tuples

    Inputs:
        space_list: list of distant tuples
    
    Returns:
        out: list of tuples expanded around initial tuples
    """
    complete_path = []

    for x, y in space_list:
        complete_path.append((x,y))
        complete_path.extend([(x + i, y) for i in range(5)])
        complete_path.extend([(x - i, y) for i in range(5)])
        complete_path.extend([(x, y + i) for i in range(5)])
        complete_path.extend([(x, y - i) for i in range(5)])
        complete_path.extend([(x + 1, y + i) for i in range(5)])
        complete_path.extend([(x - 1, y + i) for i in range(5)])
        complete_path.extend([(x - 1, y - i) for i in range(5)])
        complete_path.extend([(x + 1, y - i) for i in range(5)])
        complete_path.extend([(x - i, y + 1) for i in range(5)])
        complete_path.extend([(x + i, y - 1) for i in range(5)])
        complete_path.extend([(x - i, y - 1) for i in range(5)])
        complete_path.extend([(x + i, y + 1) for i in range(5)])

    return complete_path

# For converting the list of tuple coordinates into a visible path on the original maze
def path_creation(path, preproc_list, base_image, coordinates):
    """Rescales the path in list form to the base image size
    Converts the path list into a fitting application for the base image
    Paints the path on the base image

    Inputs:
        path: list of tuples (x,y) defining path in list image
        preproc_list: preprocessed image in list form
        base_image: original image of the maze
        coordinates: (x,y) coordinates of key points

    Returns:
        out: original image of the maze with a drawn path from highlighted start to end points
    """

    pre_length = len(preproc_list)
    pre_width = len(preproc_list[0])

    base_length = base_image.shape[0]
    base_width = base_image.shape[1]

    scaler_width = base_width / pre_width
    scaler_length = base_length / pre_length

    new_path = [(int(a * scaler_width), int(b * scaler_length)) for a, b in path]
    
    expanded_path = pixel_expand(new_path)

    out = base_image.copy()

    for x, y in expanded_path:
        if 0 <= x < out.shape[0] and 0 <= y < out.shape[1]:
            out[x, y] = [1.0, 0, 1.0]

    new_coords = [(int(a * scaler_width), int(b * scaler_length)) for a, b in coordinates]
    expanded_coords = pixel_expand(new_coords)

    for i, (x, y) in enumerate(expanded_coords):
        if i > len(expanded_coords) / 2:
            out[x, y] = [0, 1.0, 0]
        else:
            out[x, y] = [1.0, 0, 0]

    return out

def main():
    # Load the test maze image
    maze_file = sys.argv[1]
    test_image = load(f'paths/{maze_file}')
    display(test_image, 'Loaded maze image')

    # Preprocess the image, handle missing key points error and display key point coordinates
    preproc_img, coords = preprocess(test_image, 0.5)
    
    if len(coords) < 2:
        print('No starting or ending point, define red and green points')
        exit(1)

    print('Starting point coordinates: (' + str(coords[0][1]) + ', ' + str(coords[0][0]) + ')')
    print('Ending point coordinates: (' + str(coords[1][1]) + ', ' + str(coords[1][0]) + ')')

    # Convert image data into processable form
    image_list = preproc_img.tolist()

    problem = Problem(coords[0], coords[1], image_list)

    # Check for inputted algorithm or run euclidean A* algorithm otherwise
    if len(sys.argv) > 2:
        algorithm = sys.argv[2]
        if algorithm == 'astar':
            if len(sys.argv) < 4:
                print("Incorrect usage, please use format: python mazesolver.py <filename> <algorithm> <heuristic> for A* algorithm")
                exit(1)
            heuristic = sys.argv[3]
            results = astar(problem, heuristic)
            algorithm = f'{algorithm}-{heuristic}'
        elif algorithm == 'ucs':
            results = ucs(problem)
        else:
            print("Unknown algorithm, please use algorithm: astar or ucs")
            exit(1)
    else:
        eucl_out = astar(problem, "euclidean")
        manh_out = astar(problem, "manhattan")

        costs = np.array([len(eucl_out[0]), len(manh_out[0])])
        
        lowest_cost = costs.min()

        if lowest_cost == len(eucl_out[0]):
            results = eucl_out
            algorithm = 'astar-euclidean'
        elif lowest_cost == len(manh_out[0]):
            results = manh_out
            algorithm = 'astar-manhattan'
        else:
            print("Internal error")
            exit(1)

    if results[0] == "failure" or results[1] > 1000000:
        print('No possible path from start to end points')
        exit(1)

    final_image = path_creation(results[0], image_list, test_image, coords)

    display_alg = algorithm.replace('-', ' ')

    display(final_image, f'Final image with marked {display_alg} path')

    # To save preprocessed image
    # save(preproc_img, f'Preprocessed maze image', f'{maze_file}-preproc.png')
    save(final_image, f'Final image with marked {display_alg} path', f'{maze_file}-{algorithm}-path.png')

    print(f'Path found successfully using {display_alg} algorithm with a length of {len(results[0])} units')
    exit(0)


if __name__ == "__main__":
    main()