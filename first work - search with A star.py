import csv
import os
import math
import heapq
from itertools import permutations  # to create later all the combinations of start and goal location of each party

locations = {}
state_coordinates = {'AL': '32.7794°N 86.8287°W',
        'AK': '64.0685°N 152.2782°W',
        'AZ': '34.2744°N 111.6602°W',
        'AR': '34.8938°N 92.4426°W',
        'CA': '37.1841°N 119.4696°W',
        'CO': '38.9972°N 105.5478°W',
        'CT': '41.6219°N 72.7273°W',
        'DE': '38.9896°N 75.5050°W',
        'FL': '28.6305°N 82.4497°W',
        'GA': '32.6415°N 83.4426°W',
        'HI': '20.2927°N 156.3737°W',
        'ID': '44.3509°N 114.6130°W',
        'IL': '40.0417°N 89.1965°W',
        'IN': '39.8942°N 86.2816°W',
        'IA': '42.0751°N 93.4960°W',
        'KS': '38.4937°N 98.3804°W',
        'KY': '37.5347°N 85.3021°W',
        'LA': '31.0689°N 91.9968°W',
        'ME': '45.3695°N 69.2428°W',
        'MD': '39.0550°N 76.7909°W',
        'MA': '42.2596°N 71.8083°W',
        'MI': '44.3467°N 85.4102°W',
        'MN': '46.2807°N 94.3053°W',
        'MS': '32.7364°N 89.6678°W',
        'MO': '38.3566°N 92.4580°W',
        'MT': '47.0527°N 109.6333°W',
        'NE': '41.5378°N 99.7951°W',
        'NV': '39.3289°N 116.6312°W',
        'NH': '43.6805°N 71.5811°W',
        'NJ': '40.1907°N 74.6728°W',
        'NM': '34.4071°N 106.1126°W',
        'NY': '42.9538°N 75.5268°W',
        'NC': '35.5557°N 79.3877°W',
        'ND': '47.4501°N 100.4659°W',
        'OH': '40.2862°N 82.7937°W',
        'OK': '35.5889°N 97.4943°W',
        'OR': '43.9336°N 120.5583°W',
        'PA': '40.8781°N 77.7996°W',
        'RI': '41.6762°N 71.5562°W',
        'SC': '33.9169°N 80.8964°W',
        'SD': '44.4443°N 100.2263°W',
        'TN': '35.8580°N 86.3505°W',
        'TX': '31.4757°N 99.3312°W',
        'UT': '39.3055°N 111.6703°W',
        'VT': '44.0687°N 72.6658°W',
        'VA': '37.5215°N 78.8537°W',
        'WA': '47.3826°N 120.4472°W',
        'WV': '38.6409°N 80.6227°W',
        'WI': '44.6243°N 89.9941°W',
        'WY': '42.9957°N 107.5512°W',
        'AS': '-14.2710°S 170.1322°W',
        'DC': '38.9072°N 77.0369°W',
        'GU': '13.4443°N 144.7937°E',
        'MP': '15.0979°N 145.6739°E',
        'VI': '18.3358°N 64.8963°W',
        'PR': '18.2208°N 66.5901°W'}
frontier_heap = []
explored = set()

def dms_to_dd(dms): #convrets DMS to degrees formula
    direction = dms[-1]
    degrees = float(dms[:-2])

    if direction in ['S', 'W']:
        degrees = -degrees

    return degrees

def convert_to_decimal(coord): #converts the coordinates from DMS format to decimal degrees
    lat, lon = coord.split()

    lat_decimal = dms_to_dd(lat)
    lon_decimal = dms_to_dd(lon)

    return (lat_decimal, lon_decimal)

class Location:
    def __init__(self, region, state, state_coordinate):
        self.region = region
        self.state = state
        self.my_state_coordinate = state_coordinate
        self.neighbors = set()
        self.huristic = 0
        self.dist = 0  # how much it really cost me to reach this location
        self.visited = False
        self.path_to_destination = []
        self.is_starting_location = False
        self.is_goal_location = False
        self.father = None

    def __lt__(self, other):
        # priority in heap will be based on the sum of actual cost and the heuristic
        return (self.dist + self.huristic) < (other.dist + other.huristic)

    def get_region(self):
        return self.region

    def get_state(self):
        return self.state

    def get_coordinate(self):
        return self.my_state_coordinate

    def get_neighbors(self):
        return self.neighbors

    def get_dist(self):
        return self.dist

    def get_huristic(self):
        return self.huristic

    def get_father(self):
        return self.father

    def get_is_starting_location(self):
        return self.is_starting_location

    def set_coordinate(self, coor):
        self.my_state_coordinate = coor

    def set_dist(self, dist):
        self.dist = dist

    def set_father(self, father):
        self.father = father

    def set_huristic(self, huristic):
        self.huristic = huristic

    def set_is_starting_location(self, is_starting):
        self.is_starting_location = is_starting

    def set_is_goal_location(self, is_goal):
        self.is_goal_location = is_goal

    def add_neighbor(self, location):
        if location.get_region() != self.region or location.get_state() != self.state:
            self.neighbors.add(location)

def read_from_file():
    file_path = 'adjacency.csv'

    # checks if the file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return locations

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                loc1, loc2 = row
                # splitting the location to region and state
                name1, state1 = loc1.split(", ")
                name2, state2 = loc2.split(", ")
                # converting the coordinates from DMS to degrees
                coord1 = convert_to_decimal(state_coordinates.get(state1, None))
                coord2 = convert_to_decimal(state_coordinates.get(state2, None))

                # if the location wasn't added yet to the dictionary of locations
                if loc1 not in locations:
                    locations[loc1] = Location(name1, state1, coord1)
                # if the location wasn't added yet to the dictionary of locations
                if loc2 not in locations:
                    locations[loc2] = Location(name2, state2, coord2)

                locations[loc1].add_neighbor(locations[loc2])
                locations[loc2].add_neighbor(locations[loc1])

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return locations

def print_locations_in_state(locations, state_code):
    print(f"Locations in state {state_code}:")
    for loc in locations.values():
        if loc.get_state() == state_code:
            print(f"Region: {loc.get_region()}, State: {loc.get_state()}")

def check_location_name(location_name):
    if location_name.count(',') == 2:
        # splits the location given into 3 different parts by a comma
        three_parts = location_name.split(", ")
        if three_parts[0] in {"Blue", "Red"} and len(three_parts[2]) == 2 and three_parts[2].isupper():
            # checks if given location has a party color and contains exactly 2 upper letters for US code
            return True
        else:
            return False
    return False

def input_validation(locations):
    valid_locations = True

    for string in locations:
        if not check_location_name(string):
            valid_locations = False

    return valid_locations

# calculate the distance between two decimal coordinates
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # radius of earth in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# entering a location into the heap
def push_location(heap, location):
    heapq.heappush(heap, location)

# pops the location with the smallest sum of dist + heuristic from the heap
def pop_location(heap):
    return heapq.heappop(heap)

# removes the location chosen that is not on top of the heap
def remove_location(heap, location):
    try:
        # the index of the location to be removed
        index = heap.index(location)
        # swaps the location with the last element in the heap
        heap[index] = heap[-1]
        heap.pop()
        # if the heap is not empty, re-heapify the heap
        if index < len(heap):
            heapq.heapify(heap)
    except ValueError:
        print("Location not found in the frontier.")

def calculate_huristic(location, goal_location):

    distance = calculate_distance(location.get_coordinate(), goal_location.get_coordinate())
    huristic = distance/100  # dividing the distance in km by average speed per hour
    # rounding the result
    if huristic - int(huristic) >= 0.5:
        return int(huristic) + 1
    else:
        return int(huristic)

    #return huristic

def search_by_a_star(starting_location, goal_location):  # serching by A*
    flag = True
    explored.add(starting_location)

    if starting_location.get_region() == goal_location.get_region() and starting_location.get_state() == goal_location.get_state():
        return starting_location

    for loc in starting_location.get_neighbors():
        loc.set_dist(loc.get_dist() + 1)  # dist of one step
        loc.set_huristic(calculate_huristic(loc, goal_location))
        loc.set_father(starting_location)
        push_location(frontier_heap, loc)

    while flag:
        if len(frontier_heap) == 0:
            return "failure"
        else:
            pooped_location = pop_location(frontier_heap)
            if pooped_location.get_region() == goal_location.get_region() and pooped_location.get_state() == goal_location.get_state():
                return pooped_location
            else:
                explored.add(pooped_location)
                for child in pooped_location.get_neighbors():
                    if child not in explored and child not in frontier_heap:
                        if pooped_location.get_father().get_region() != child.get_region() and pooped_location.get_father().get_state() != child.get_state():
                            child.set_dist(pooped_location.get_dist() + 1)
                            child.set_huristic(calculate_huristic(child, goal_location))
                            child.set_father(pooped_location)
                            push_location(frontier_heap, child)
                    elif child in frontier_heap and (child.get_dist() > pooped_location.get_dist() + 1 ):
                        remove_location(frontier_heap, child)
                        child.set_dist(pooped_location.get_dist() + 1)
                        child.set_father(pooped_location)
                        push_location(frontier_heap, child)


def find_path_to_root(location):  # to find the way from the start location to the goal location
    path = []
    current = location
    if location.get_father() == "None":
        region_and_state_name = (location.get_region(), location.get_state())
        loc_full_statement = ",".join(region_and_state_name)
        return loc_full_statement

    for i in range(0, location.get_dist()+1):
        path.append(current)
        current = current.get_father()
    return path[::-1]  # reverse the path to get it from root to the location

def conveting_solution_to_format(solution, detail_output, party_color):
    path_in_format = []
    if solution == "failure":
        print("No path found")
    elif not detail_output:
        way = find_path_to_root(solution)  # finds the path from the start to goal
        for loc in way:
            region_and_state_name = (loc.get_region(), loc.get_state())
            loc_full_statement = ",".join(region_and_state_name)
            location_in_format = loc_full_statement + " (" + party_color + ")"
            path_in_format.append(location_in_format)
        return path_in_format
    else:
        way = find_path_to_root(solution)
        huristic_value = way[1].get_huristic()
        for loc in way:
            region_and_state_name = (loc.get_region(), loc.get_state())
            loc_full_statement = ",".join(region_and_state_name)
            location_in_format = loc_full_statement + " (" + party_color + ")"
            path_in_format.append(location_in_format)
        path_in_format.append(huristic_value)
        return path_in_format

# Initialize all values needed for the next search
    for loc in explored:
        loc.set_father(None)
        loc.set_dist(0)
        loc.set_huristic(0)

    for loc in frontier_heap:
        loc.set_father(None)
        loc.set_dist(0)
        loc.set_huristic(0)

    frontier_heap.clear()
    explored.clear()

def print_path_in_format (path, detail_output):
    max_length = max(len(sublist) for sublist in path)
    nodes_lists = [[] for _ in range(max_length)]
    i = 0

    if not detail_output:
        for lst in nodes_lists:
            for sublist in path:
                if i < len(sublist):
                    lst.append(sublist[i])
                else:
                    lst.append(sublist[len(sublist)-1])
            print(f"{{{'; '.join(map(str, lst))}}}")
            i += 1
    else:
        nodes_lists = [[] for _ in range(max_length-1)]
        i = 0
        # fills the nodes_lists with elements from each position
        for i in range(max_length - 1):
            for sublist in path:
                if i < len(sublist)-1:
                    nodes_lists[i].append(sublist[i])
                else:
                    nodes_lists[i].append(sublist[len(sublist)-2])

        # creates a list of the last elements of each sublist- the heuristic value
        last_elements = ["Heuristic:"]
        for sublist in path:
            if sublist:  # ensures that the sublist is not empty
                last_elements.append(sublist[-1])
            else:
                last_elements.append("no path found")

        # remove the first element ("Heuristic") from the heuristic values
        heuristic_values = last_elements[1:]
        # insert the heuristic values list after the second position
        nodes_lists.insert(2, heuristic_values)

        # print the resulting lists
        for index, result_list in enumerate(nodes_lists):
            if index == 2:  #for the "Heuristic" list
                print(f"Heuristic: {{{'; '.join(map(str, result_list))}}}")
            else:
                print(f"{{{'; '.join(map(str, result_list))}}}")

def find_the_best_assignment(starting_locations,goal_locations):
    distances = []
    for loc1 in starting_locations:
        loc_distances = []
        for loc2 in goal_locations:
            distance = calculate_huristic(loc1, loc2)
            loc_distances.append(distance)
        distances.append(loc_distances)

    # number of locations/goals
    n = len(goal_locations)
    # generate all possible permutations of goal indices
    all_permutations = permutations(range(n))

    # evaluate each permutation to find the best one
    min_total_distance = float('inf')
    best_permutation = None
    for perm in all_permutations:
        total_distance = sum(distances[i][perm[i]] for i in range(n))
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_permutation = perm

    return best_permutation

def validation_check(starting_locations):
    # validation check to see if each starting location exist
    for string in starting_locations:
        input_county_state = ", ".join(string.split(", ")[1:])  # extracts the county and state from input
        found_location = False
        for location_name, location_obj in locations.items():
            if input_county_state == location_name:
                found_location = True
                location_obj.set_is_starting_location(True)
        if not found_location:
            print("Location not found:", input_county_state)

def start_search(starting_locations, goal_locations, search_method, detail_output):
    Locations = read_from_file()
    path_to_goal = []
    validation_check(starting_locations)

    blue_starting_locations = []
    blue_goal_locations = []
    red_starting_locations = []
    red_goal_locations = []

    for string in starting_locations:
        splitted_string = string.split(", ")
        region_state = string.split(", ")[1:]
        # joins the region and state to form the key
        key = ", ".join(region_state)

        if splitted_string[0] == "Blue":
            blue_starting_loc = locations.get(key, None)
            blue_starting_locations.append(blue_starting_loc)
        else:
            red_starting_loc = locations.get(key, None)
            red_starting_locations.append(red_starting_loc)

    for string in goal_locations:
        splitted_string = string.split(", ")
        region_state = string.split(", ")[1:]
        # joins the region and state to form the key
        key = ", ".join(region_state)

        if splitted_string[0] == "Blue":
            blue_goal_loc = locations.get(key, None)
            blue_goal_locations.append(blue_goal_loc)
        else:
            red_goal_loc = locations.get(key, None)
            red_goal_locations.append(red_goal_loc)

    # finds the best assignment of start and goal destinations
    best_blue_assignment = find_the_best_assignment(blue_starting_locations, blue_goal_locations)
    best_red_assignment = find_the_best_assignment(red_starting_locations, red_goal_locations)

    if search_method == 1:
        for i in best_blue_assignment:
            solution = search_by_a_star(blue_starting_locations[i], blue_goal_locations[best_blue_assignment[i]])
            if solution != "failure":
                solution_in_format = conveting_solution_to_format(solution, detail_output, "B")
                path_to_goal.append(solution_in_format)
            else:
                path_to_goal.append(["No path found"])

        for i in best_red_assignment:
            solution = search_by_a_star(red_starting_locations[i], red_goal_locations[best_red_assignment[i]])
            if solution != "failure":
                solution_in_format = conveting_solution_to_format(solution, detail_output, "R")
                path_to_goal.append(solution_in_format)
            else:
                path_to_goal.append(["No path found"])

        print_path_in_format(path_to_goal, detail_output)

def find_path(starting_locations, goal_locations, search_method, detail_output):

    valid_starting_locations = input_validation(starting_locations)
    valid_goal_locations = input_validation(goal_locations)

    if valid_starting_locations and valid_goal_locations:
        start_search(starting_locations, goal_locations, search_method, detail_output)
    else:
        print("invalid_input")

