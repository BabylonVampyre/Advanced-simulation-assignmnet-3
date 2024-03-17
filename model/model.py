import mesa
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link, Intersection
import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max

# Get the delay time of a bridge to be used in the data collector
def get_delay(agent):
    if type(agent) == Bridge:
        return agent.get_delay_time()
    else:
        return None
# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Only straight paths in the Demo are added into the dict;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    file_name = ('../data/N1N2.csv')

    def __init__(self, seed=None,scenario = [0,[0,0,0,0]], x_max=500, y_max=500, x_min=0, y_min=0):

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []

        #set seed
        self.random.seed(seed)
        #Save scenario to give to the agents in the initiation
        self.scenario = scenario
        self.possible_catagories = ['A','B','C','D']

        self.generate_model()
        self.generate_network()
        self.model_reporters = {}
        self.agent_reporters = {}
        self.model_vars = {}
        self._agent_records = {}
        self.tables = {}
        self.datacollector = mesa.DataCollector()

        # data collector of delay time and vehicle driving time when the vehicle has arrived at the sink
        self.datacollector = mesa.DataCollector(model_reporters={},
                                                agent_reporters={"Delay time": lambda a: get_delay(
                                                    a) if a.__class__.__name__ == 'Bridge' else None,
                                                                 "Driving time of cars leaving": lambda
                                                                     a: a.vehicle_removed_driving_time if a.__class__.__name__ == 'Sink' or a.__class__.__name__ == 'SourceSink' else None})

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv(self.file_name)

        # a list of names of roads to be generated
        # TODO You can also read in the road column to generate this list automatically
        # roads = ['N1', 'N2']
        roads = df['road'].unique()

        df_objects_all = []
        for road in roads:
            # Select all the objects on a particular road in the original order as in the cvs
            df_objects_on_road = df[df['road'] == road]

            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)

                """
                Set the path 
                1. get the serie of object IDs on a given road in the cvs in the original order
                2. add the (straight) path to the path_ids_dict
                3. put the path in reversed order and reindex
                4. add the path to the path_ids_dict so that the vehicles can drive backwards too
                """
                path_ids = df_objects_on_road['id']
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids
                path_ids = path_ids[::-1]
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids

        # put back to df with selected roads so that min and max and be easily calculated
        df = pd.concat(df_objects_all)
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():  # index, row in ...

                # create agents according to model_type
                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                else:
                    name = name.strip()

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], name, row['road'])
                    self.sinks.append(agent.unique_id)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], name, row['road'], row['condition'],scenario=self.scenario)
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                elif model_type == 'intersection':
                    if not row['id'] in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return self.path_ids_dict[source, sink]

    # TODO
    def get_route(self, source):
        return self.get_straight_route(source)

    def get_straight_route(self, source):
        """
        pick up a straight route given an origin
        """
        return self.path_ids_dict[source, None]

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()
        self.datacollector.collect(self)

    def generate_network(self):
        """
        Make the network using networkx, load in csv file
        """
        # make a clean networkx graph
        network = nx.Graph()
        # load in the file of the network (using bangladesh model)
        df_network = pd.read_csv(self.file_name)

        # A source sink or bridge is a node ->
        # find them in the csv and add them using for loop, position is the lon and lat
        for _, row in df_network.iterrows():
            model_type = row['model_type'].strip()
            model_lat = row['lat']
            model_lon = row['lon']
            model_ID = row["id"]
            # print(model_type)
            if model_type == "sourcesink":
                network.add_node(model_ID, pos=(model_lat,model_lon))
            elif model_type == "bridge":
                network.add_node(model_ID, pos=(model_lat, model_lon))
            elif model_type == "intersection":
                network.add_node(model_ID, pos=(model_lat, model_lon))

        # Add the links between the nodes using the index of the previous and upcoming row of the dataframe.
        for index, row in df_network.iterrows():
            model_type = row['model_type'].strip()
           # model_weight = row["length"]
            if model_type == "link":
                previous_id = df_network.at[index-1, "id"]
                upcoming_id = df_network.at[index+1, "id"]
                network.add_edge(previous_id, upcoming_id)# model_weight = length)

        # #plot the network, with the ID as a label for the node
        # nx.draw(network, with_labels=True)
        # plt.show()


# EOF -----------------------------------------------------------
