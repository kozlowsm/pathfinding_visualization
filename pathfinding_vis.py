import pygame
import pygame_gui
import random
import time
from queue import PriorityQueue

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQOISE = (64, 224, 208)

DARK_BLUE = (54, 86, 115)
# DARK_BLUE = (14, 111, 201)
# DARK_BLUE = (11, 87, 158)
LIGHT_BLUE = (112, 187, 255)


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.prim_neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == DARK_BLUE

    def is_open(self):
        return self.color == LIGHT_BLUE

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GREEN

    def is_end(self):
        return self.color == RED

    def is_path(self):
        return self.color == YELLOW

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = GREEN

    def make_closed(self):
        self.color = DARK_BLUE

    def make_open(self):
        self.color = LIGHT_BLUE

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = RED

    def make_path(self):
        self.color = YELLOW

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # UP
        if self.row > 0 and not grid[self.row-1][self.col].is_barrier():
            self.neighbors.append(grid[self.row-1][self.col])
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row+1][self.col].is_barrier():
            self.neighbors.append(grid[self.row+1][self.col])
        # LEFT
        if self.col > 0 and not grid[self.row][self.col-1].is_barrier():
            self.neighbors.append(grid[self.row][self.col-1])
        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col+1].is_barrier():
            self.neighbors.append(grid[self.row][self.col+1])

    # used for generating a maze with prims algorithm
    def update_prim_neighbors(self, grid):
        self.prim_neighbors = []
        # UP
        if self.row > 0 and not grid[self.row-2][self.col].is_barrier():
            self.neighbors.append(grid[self.row-2][self.col])
        # DOWN
        if self.row < self.total_rows - 2 and not grid[self.row+2][self.col].is_barrier():
            self.neighbors.append(grid[self.row+2][self.col])
        # LEFT
        if self.col > 0 and not grid[self.row][self.col-2].is_barrier():
            self.neighbors.append(grid[self.row][self.col-2])
        # RIGHT
        if self.col < self.total_rows - 2 and not grid[self.row][self.col+2].is_barrier():
            self.neighbors.append(grid[self.row][self.col+2])


class Grid:
    def __init__(self, rows, width):
        self.start = None
        self.end = None
        self.rows = rows
        self.width = width
        self.height = width
        self.path_len = 0
        self.total_time = 0.00
        self.grid = []

        self.make_grid()

    # Initialize grid full of nodes

    def make_grid(self):
        self.grid = []
        gap = self.width // self.rows
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.rows):
                node = Node(i, j, gap, self.rows)
                self.grid[i].append(node)

    # Draw the grid lines to the screen
    def draw_grid(self, win_surface):
        gap = self.width // self.rows
        for i in range(self.rows):
            pygame.draw.line(win_surface, BLACK, (0, i*gap),
                             (self.width, i*gap))
            pygame.draw.line(win_surface, BLACK, (i*gap, 0),
                             (i*gap, self.width))

    # Draws nodes and lines to the screen
    def draw(self, win_surface):
        for row in self.grid:
            for node in row:
                node.draw(win_surface)

        self.draw_grid(win_surface)
        pygame.display.update()

    # Returns the node from the graph that the user clicks on
    def get_clicked_position(self, pos):
        gap = self.width // self.rows
        y, x = pos

        row = y // gap
        col = x // gap

        return row, col

    # Remove all node color fill from the grid
    def clear_all(self):
        self.start = None
        self.end = None
        self.make_grid()

    # Keeps barriers, removes all other nodes
    def clear_paths(self):
        self.start = None
        self.end = None
        grid_copy = self.grid[:]
        for row in grid_copy:
            for node in row:
                if node.color != BLACK:
                    node.reset()
        self.grid = grid_copy

    def clear_searched(self):
        grid_copy = self.grid[:]
        for row in grid_copy:
            for node in row:
                if node.is_open() or node.is_closed() or node.is_path():
                    node.reset()
        self.grid = grid_copy

    # Generate a random grid pattern
    def randomize_grid(self, percent_full):
        self.start = None
        self.end = None
        self.make_grid()
        for row in self.grid:
            for node in row:
                barrier_prob = random.random()
                if barrier_prob < percent_full:
                    node.make_barrier()

    def reconstruct_path(self, came_from, win_surface):
        if isinstance(came_from, list):
            for node in came_from:
                node.make_path()
                pygame.time.delay(5)
                self.draw(win_surface)
        else:
            current = self.end
            while current in came_from:
                current = came_from[current]
                current.make_path()
                pygame.time.delay(5)
                self.draw(win_surface)

    def get_path_len(self, came_from):
        self.path_len = 0

        if isinstance(came_from, list):
            self.path_len = len(came_from) - 1
        else:
            current = self.end
            while current in came_from:
                current = came_from[current]
                self.path_len += 1

    # Hueristic function for A* and Greedy-Best
    def h(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1-x2) + abs(y1-y2)

    def get_algorithm(self, option):
        algorithms = {
            'A*': self.a_star,
            'DIJKSTRA': self.dijkstra,
            'GREEDY-BEST': self.greedy_best,
            'DFS': self.dfs,
            'BFS': self.bfs
        }

        function = algorithms.get(option)
        return function

    def a_star(self, draw, win_surface):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, self.start))
        came_from = {}
        g_score = {spot: float('inf') for row in self.grid for spot in row}
        g_score[self.start] = 0
        f_score = {spot: float('inf') for row in self.grid for spot in row}
        f_score[self.start] = self.h(self.start.get_pos(), self.end.get_pos())

        open_set_hash = {self.start}
        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == self.end:
                self.end.make_end()
                self.reconstruct_path(came_from, win_surface)
                self.end.make_end()
                self.start.make_start()
                return came_from

            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + \
                        self.h(neighbor.get_pos(), self.end.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        neighbor.make_open()

                draw()
            if current != self.start:
                current.make_closed()

        return {}

    def dijkstra(self, draw, win_surface):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, self.start))
        came_from = {}
        g_score = {spot: float('inf') for row in self.grid for spot in row}
        g_score[self.start] = 0
        f_score = {spot: float('inf') for row in self.grid for spot in row}
        f_score[self.start] = self.h(self.start.get_pos(), self.end.get_pos())

        open_set_hash = {self.start}
        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == self.end:
                self.end.make_end()
                self.reconstruct_path(came_from, win_surface)
                self.end.make_end()
                self.start.make_start()
                return came_from

            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = 0
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        neighbor.make_open()

                draw()
            if current != self.start:
                current.make_closed()

        return {}

    def greedy_best(self, draw, win_surface):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, self.start))
        came_from = {}
        g_score = {spot: float('inf') for row in self.grid for spot in row}
        g_score[self.start] = 0
        f_score = {spot: float('inf') for row in self.grid for spot in row}
        f_score[self.start] = self.h(self.start.get_pos(), self.end.get_pos())

        open_set_hash = {self.start}
        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == self.end:
                self.end.make_end()
                self.reconstruct_path(came_from, win_surface)
                self.end.make_end()
                self.start.make_start()
                return came_from

            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = self.h(
                        neighbor.get_pos(), self.end.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        neighbor.make_open()

                draw()
            if current != self.start:
                current.make_closed()

        return {}

    def dfs(self, draw, win_surface):
        explored = {}  # RED
        stack = [[self.start]]

        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            path = stack.pop()  # pop the first node in the queue
            node = path[-1]  # get the last node from the path

        # if node not in explored:
            for neighbor in node.neighbors:
                if neighbor not in explored:
                    explored[neighbor] = True
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
                    neighbor.make_open()
                    if neighbor == self.end:
                        self.end.make_end()
                        self.start.make_start()
                        self.reconstruct_path(new_path, win_surface)
                        self.start.make_start()
                        self.end.make_end()
                        return new_path

            explored[node] = True
            if node != self.start and node != self.end:
                node.make_closed()
            draw()

        return {}

    def bfs(self, draw, win_surface):
        explored = {}  # RED
        queue = [[self.start]]

        while queue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            path = queue.pop(0)  # pop the first node in the queue
            node = path[-1]  # get the last node from the path

        # if node not in explored:
            for neighbor in node.neighbors:
                if neighbor not in explored:
                    explored[neighbor] = True
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    neighbor.make_open()
                    if neighbor == self.end:
                        self.end.make_end()
                        self.reconstruct_path(new_path, win_surface)
                        self.start.make_start()
                        self.end.make_end()
                        return new_path

            explored[node] = True
            if node != self.start and node != self.end:
                node.make_closed()
            draw()

        return {}


class VisualizerApp:
    def __init__(self, rows, height):
        pygame.init()
        pygame.display.set_caption('Pathfinding Algorithms')

        self.rows = rows
        self.height = height
        # Total width always 300 larger than height so grid is always square
        self.width = height + 300

        self.flags = pygame.SCALED
        self.window_surface = pygame.display.set_mode(
            (self.width, self.height))
        self.manager = pygame_gui.UIManager(
            (self.width, self.height), 'theme.json')

        self.grid_rect = pygame.draw.rect(
            self.window_surface, (0, 0, 0), pygame.Rect((0, 0), (600, 600)))
        self.menu_rect = pygame.draw.rect(
            self.window_surface, (30, 30, 30), pygame.Rect((600, 0), (300, 600)))

        # Menu buttons
        self.algo_selection_text = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect(650, 10, 200, 50), "Select Algorithm: ", manager=self.manager, object_id="#algo_selection")
        self.run_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (800, 50), (50, 50)), text='Run', manager=self.manager)
        self.clear_all_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (650, 150), (100, 50)), text='Clear All', manager=self.manager)
        self.clear_paths_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (750, 150), (100, 50)), text='Clear Nodes', manager=self.manager)
        self.clear_searched_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (675, 200), (150, 50)), text="Clear Searched", manager=self.manager)
        self.randomize_grid_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (675, 500), (150, 50)), text='Randomize Grid', manager=self.manager)
        self.path_len_text = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect(650, 285, 200, 30), "Path Length: 0", manager=self.manager)
        self.total_time_text = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect(650, 315, 200, 30), "Total Time (s): 0.00", manager=self.manager)
        self.no_path_text = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect(650, 340, 200, 50), "", manager=self.manager, object_id="#no_path")

        # Dropdown menu and options
        self.algo_options = ['A*', 'BFS', 'DFS', 'DIJKSTRA', 'GREEDY-BEST']
        self.selection = self.algo_options[0]
        # Create the button for dropdown
        self.algo_dropdown_button = pygame_gui.elements.UIDropDownMenu(options_list=self.algo_options, starting_option=self.selection,
                                                                       relative_rect=pygame.Rect((650, 50), (150, 50)), manager=self.manager)

        self.grid_system = Grid(rows, self.height)

        self.clock = pygame.time.Clock()
        self.is_running = True

    def run(self):
        while self.is_running:
            self.grid_system.draw(self.window_surface)
            time_delta = self.clock.tick(60) / 1000.0

            # Handle all the events -- Button clicks and mouse actions
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if pygame.mouse.get_pressed()[0]:  # LEFT mouse button
                    pos = pygame.mouse.get_pos()
                    if pos[0] < self.height:
                        row, col = self.grid_system.get_clicked_position(pos)
                        node = self.grid_system.grid[row][col]
                        if not self.grid_system.start and node != self.grid_system.end:
                            self.grid_system.start = node
                            self.grid_system.start.make_start()

                        elif not self.grid_system.end and node != self.grid_system.start:
                            self.grid_system.end = node
                            self.grid_system.end.make_end()

                        elif node != self.grid_system.end and node != self.grid_system.start:
                            node.make_barrier()

                elif pygame.mouse.get_pressed()[2]:  # RIGHT
                    pos = pygame.mouse.get_pos()
                    if pos[0] < self.height:
                        row, col = self.grid_system.get_clicked_position(pos)
                        node = self.grid_system.grid[row][col]
                        node.reset()
                        if node == self.grid_system.start:
                            self.grid_system.start = None
                        elif node == self.grid_system.end:
                            self.grid_system.end = None

                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.run_button and self.grid_system.start and self.grid_system.end:
                            for row in self.grid_system.grid:
                                for node in row:
                                    node.update_neighbors(
                                        self.grid_system.grid)
                            algorithm = self.grid_system.get_algorithm(
                                self.selection)
                            start_time = time.time()
                            came_from = algorithm(
                                lambda: self.grid_system.draw(self.window_surface), self.window_surface)
                            end_time = time.time()
                            self.grid_system.get_path_len(came_from)
                            self.path_len_text.set_text(
                                f"Path Length: {self.grid_system.path_len}")
                            self.total_time_text.set_text(
                                f"Total Time (s): {end_time - start_time:.3f}")
                            if self.grid_system.path_len == 0:
                                self.no_path_text.set_text(
                                    "Path does not exist")
                            else:
                                self.no_path_text.set_text("")
                        if event.ui_element == self.clear_all_button:
                            self.grid_system.clear_all()
                        if event.ui_element == self.clear_paths_button:
                            self.grid_system.clear_paths()
                        if event.ui_element == self.clear_searched_button:
                            self.grid_system.clear_searched()
                        if event.ui_element == self.randomize_grid_button:
                            self.grid_system.randomize_grid(.35)

                    if event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                        self.selection = event.text

                self.manager.process_events(event)
            pygame.draw.rect(self.window_surface, (30, 30, 30),
                             pygame.Rect((600, 0), (300, 600)))
            self.manager.update(time_delta)
            pygame.draw.line(self.window_surface, (90, 90, 90),
                             (600, 120), (900, 120), width=3)

            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
        pygame.quit()


if __name__ == '__main__':
    app = VisualizerApp(50, 600)
    app.run()
