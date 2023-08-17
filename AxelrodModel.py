from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from collections import deque
import matplotlib.pyplot as plt
import numpy as np


def calculate_stats(model):
    grid = model.grid
    agent_features = []
    for agent in model.schedule.agents:
        agent_features.append(tuple(agent.feature))
    max_cluster_size, num_clusters = find_clusters(
        agent_features, grid.width, grid.height
    )

    return max_cluster_size / model.agents_num


def find_clusters(agent_features, width, height):
    grid = np.zeros((height, width))
    visited = np.zeros((height, width), dtype=bool)
    clusters = []
    max_domain_size = 0

    for y in range(height):
        for x in range(width):
            if not visited[y][x]:
                feature = agent_features[y * width + x]
                domain = bfs(agent_features, visited, feature, x, y, width, height)
                clusters.append(domain)
                max_domain_size = max(max_domain_size, len(domain))

    num_domains = len(clusters)

    return max_domain_size, num_domains


def bfs(agent_features, visited, target_feature, start_x, start_y, width, height):
    queue = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    cluster = []

    while queue:
        x, y = queue.popleft()
        cluster.append((x, y))

        neighbors = get_neighbors(x, y, width, height)
        for nx, ny in neighbors:
            if (
                not visited[ny][nx]
                and agent_features[ny * width + nx] == target_feature
            ):
                queue.append((nx, ny))
                visited[ny][nx] = True

    return cluster


def get_neighbors(x, y, width, height):
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < width - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < height - 1:
        neighbors.append((x, y + 1))
    return neighbors


class AxelrodAgent(Agent):
    def __init__(self, unique_id, features, traits, model):
        super().__init__(unique_id, model)
        self.feature = np.random.randint(low=0, high=traits, size=features)

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=False)
        random_neighbor = self.random.choice(neighbors)
        probability = (
            self.model.feature_num
            - np.count_nonzero(self.feature - random_neighbor.feature)
        ) / self.model.feature_num
        if probability > np.random.rand():
            index = np.random.choice(self.model.feature_num)
            self.feature[index] = random_neighbor.feature[index]


class AxelrodModel(Model):
    def __init__(self, N, width, height, features, traits):
        self.agents_num = N
        self.feature_num = features

        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        self.fig, self.ax = plt.subplots()
        self.img = None

        for id in range(self.agents_num):
            agent = AxelrodAgent(id, features, traits, self)
            x = id % width
            y = id // width
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        self.datacollector = DataCollector(model_reporters={"Smax/N": calculate_stats})

    def step(self, display_plot=True):
        self.datacollector.collect(self)
        random_agent = self.random.choice(self.schedule.agents)
        random_agent.step()
        # self.schedule.step()
        # self.display_features()
        if display_plot:
            self.display_plot()

    def display_features(self):
        for agent in self.schedule.agents:
            print(f"Agent ID: {agent.unique_id}, Agent featrues: {agent.feature}")

    def map_vector_to_color(self, vector):
        vector_id = hash(tuple(vector))
        rng = np.random.RandomState(vector_id % (2**32 - 1))
        color = rng.rand(3)

        return color

    def display_plot(self):
        color_list = []
        for agent in self.schedule.agents:
            color_list.append(self.map_vector_to_color(agent.feature))

        color_matrix = np.array(color_list).reshape(
            self.grid.width, self.grid.height, 3
        )

        if self.img is None:
            self.img = self.ax.imshow(color_matrix)
            self.ax.axis("off")
        else:
            self.img.set_data(color_matrix)
            plt.pause(0.001)
            plt.draw()

    def get_color_matrix(self):
        color_list = []
        for agent in self.schedule.agents:
            color_list.append(self.map_vector_to_color(agent.feature))

        color_matrix = np.array(color_list).reshape(
            self.grid.width, self.grid.height, 3
        )

        return color_matrix
