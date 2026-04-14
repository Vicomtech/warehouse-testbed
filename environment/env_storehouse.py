from math import ceil, prod
from pathlib import Path
from statistics import mean
from time import time
import csv
import itertools
import json
import logging
import operator
import os

import gymnasium as gym
import numpy as np
from colorama import Back, Fore, Style
from gymnasium import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.stats import poisson
from skimage.morphology import flood_fill

CONF_NAME = "realistic_warehouse"
MAX_NUM_BOXES = 6
MIN_NUM_BOXES = 2
FEATURE_NUMBER = 3
MAX_INVALID = 10
MAX_MOVEMENTS = 100
MIN_CNN_LEN = 32
MIN_SB3_SIZE = 32
EPISODE = 0
NUM_RANDOM_STATES = 2000
PATH_REWARD_PROPORTION = 0.0
ENV_VERSION = "v2.0"

TYPE_CODIFICATION = {"A": 100, "B": 200}
TYPE_COMB_CODIFICATION = {0: 0, 1: 50, 2: 100, 3: 150, 4: 200, 5: 255}

class Score:
    """Stores episode-level performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.delivered_boxes = 0
        self.filled_orders = 0
        self.steps = 0
        self.box_ages = []
        self.non_optimal_material = 0
        self.timer = 0
        self.max_id = 0
        self.total_orders = 0
        self.seed = 0
        self.returns = []
        self.discounted_return = 0
        self.num_invalid = 0
        self.trapped = 0

    def print_header(self) -> str:
        return (
            f"Score,"
            f"Discounted return,"
            f"Delivered Boxes,"
            f"Filled orders,"
            f"Total orders,"
            f"Mean box ages,"
            f"FIFO violation,"
            f"Invalid actions,"
            f"max_id,"
            f"Steps,"
            f"time,"
            f"Seed,"
            f"trapped"
            f"\n"
        )

    def print_score(self) -> str:
        return (
            f"{sum(self.returns)},"
            f"{self.discounted_return},"
            f"{self.delivered_boxes},"
            f"{self.filled_orders},"
            f"{self.total_orders},"
            f"{mean(self.box_ages)},"
            f"{self.non_optimal_material / max(1, self.delivered_boxes) * 100},"
            f"{self.num_invalid},"
            f"{self.max_id},"
            f"{self.steps},"
            f"{self.timer},"
            f"{self.seed},"
            f"{self.trapped}"
            f"\n"
        )


class Box:
    """Represents a single box stored or transported in the warehouse."""

    def __init__(self, id: int, position: tuple, type: str = "A", age: int = 1):
        self.id = id  
        self.type = type
        self.age = age
        self.position = position
    def update_age(self, num_steps: int = 1):
        """Increase the age of the box by at least one step."""
        self.age += max(num_steps, 1)

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.position == other.position and self.age == other.age and self.id == other.id and self.age == other.age

    def __repr__(self) -> str:
        return f"Box(id={self.id}, type={self.type}, age={self.age})"


class Agent:
    """Represents the mobile warehouse agent."""

    def __init__(self, initial_position: tuple, got_item: int = 0):
        self.position = initial_position
        self.got_item = got_item

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.position == other.position and self.got_item == other.got_item


class Entrypoint:
    def __init__(self, position: tuple, type_information: dict, rng: list):
        self.rng = rng
        self.type_information = type_information
        self.position = position
        self.material_queue = []

    def create_new_material(self, max_id: int):
        """Represents a warehouse entrypoint where new boxes can be generated."""
        box_type = self.rng[0].choice(list(self.type_information.keys()))
        prob = self.type_information[box_type]["create"]
        if self.rng[0].choice([True, False], p=[prob, 1 - prob]):
            material = Box(max_id, self.position, box_type)
            self.material_queue.append(material)
            max_id += 1
        return max_id

    def get_item(self) -> Box:
        """Generate a new box according to the configured creation probabilities."""
        try:
            return self.material_queue.pop(0)
        except IndexError as ex:
            logging.error(f"Error at get_item in Entrypoint {self.position}")
            raise IndexError from ex

    def update_entrypoint(self, max_id, steps: int = 1):
        """Return and remove the first available box from the queue."""
        for material in self.material_queue:
            material.update_age(max(1, steps))
        return self.create_new_material(max_id)

    def reset(self):
        """Clear the entrypoint queue."""
        self.material_queue = []


class Delivery:
    """Represents a delivery request waiting to become active."""

    def __init__(
        self,
        prob: int,
        num_boxes: int,
        type: str,
        rng: list,
        timer: int = 0,
        ready: bool = False,
    ):
        self.type = type
        self.prob = prob
        self.num_boxes = num_boxes
        self.timer = timer
        self.ready = ready
        self.rng = rng

    def update_timer(self, step: int = 1):
        """Advance the internal timer and activate the request when ready."""
        self.timer += step
        if not self.ready:
            prob = poisson.cdf(self.timer, self.prob)
            if prob:
                self.ready = True

    def __repr__(self) -> str:
        return (
            f"Delivery(type={self.type}, num_boxes={self.num_boxes}, timer={self.timer}, ready={self.ready}, prob={self.prob})"
        )

    def __eq__(self, other):
        if isinstance(other, Delivery):
            return (
                self.type == other.type
                and self.prob == other.prob
                and self.num_boxes == other.num_boxes
                and self.timer == other.timer
                and self.ready == other.ready
            )


class Outpoints:
    """Manages delivery requests associated with warehouse outpoints."""

    def __init__(self, outpoints: list, type_information: dict, delivery_prob: dict, rng: list):  
        self.outpoints = outpoints
        self.type_information = type_information
        self.delivery_prob = delivery_prob
        self.max_num_boxes = MAX_NUM_BOXES
        self.min_num_boxes = MIN_NUM_BOXES
        self.delivery_schedule = []
        self.last_delivery_timers = np.inf
        self.rng = rng
        self.new_deliveries_this_step = []

    def reset(self):
        """Reset delivery state."""
        self.delivery_schedule = []
        self.last_delivery_timers = np.inf
        self.new_deliveries_this_step = []

    def update_timers(self, steps: int = 1):
        """Advance the timers of all pending deliveries."""
        self.last_delivery_timers += max(1, steps)
        for delivery in self.delivery_schedule:
            delivery.update_timer(max(0, max(1, steps)))

    def create_order(self, type: str) -> dict:
        """Create a delivery request for a given material type."""
        num_boxes = self.rng[0].integers(self.min_num_boxes, self.max_num_boxes + 1)
        return Delivery(type=type, prob=self.type_information[type]["deliver"], num_boxes=num_boxes, rng=self.rng)

    def create_delivery(self) -> dict:
        """Randomly create a new delivery request."""
        prob = self.delivery_prob
        if not self.rng[0].choice([True, False], p=[prob, 1 - prob]):
            return None

        box_type = self.rng[0].choice(list(self.type_information.keys()))
        order = self.create_order(box_type)
        self.delivery_schedule.append(order)
        self.last_delivery_timers = 0
        self.new_deliveries_this_step.append(order)

        op_position = self.rng[0].choice(self.outpoints)
        order.op_position = op_position
        return order

    def consume(self, box: Box) -> int:
        """
                Try to consume a delivered box.

                Returns:
                    0: box not consumed
                    1: box consumed, order still pending
                    2: box consumed and order completed
                """
        for ii, order in enumerate(self.delivery_schedule):
            if box.type == order.type and order.ready:
                self.delivery_schedule[ii].num_boxes -= 1
                if self.delivery_schedule[ii].num_boxes < 1:
                    del self.delivery_schedule[ii]
                    return 2
                return 1
        return 0


class Storehouse(gym.Env):
    """
    Custom Gymnasium environment for warehouse logistics.

    The environment models:
    - warehouse layouts
    - entrypoints where boxes are generated
    - outpoints where delivery requests are fulfilled
    - a mobile agent that collects, stores, and delivers boxes

    It is designed for benchmarking reinforcement learning and heuristic
    methods under configurable and reproducible scenarios.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        logname: str = "log/log",
        logging: bool = False,
        save_episodes: bool = False,
        transpose_state: bool = False,
        max_steps: int = MAX_MOVEMENTS,
        conf_name: str = CONF_NAME,
        augment: bool = False,
        random_start: bool = False,
        normalized_state: bool = False,
        path_reward_weight: float = PATH_REWARD_PROPORTION,
        seed: int = None,
        reward_function: int = 0,
        gamma: float = 0.99,
        record_scenario = False,
        scenario_path = "scenario_seed_X"
    ):
        """
            Initialize the warehouse environment.

            Args:
                logname: Base path for log files.
                logging: Whether to enable metric logging.
                save_episodes: Whether to store episode traces.
                transpose_state: Whether to return channel-first states.
                max_steps: Maximum number of actions per episode.
                conf_name: Configuration name loaded from conf.json.
                augment: Whether to enlarge the state for CNN-based models.
                random_start: Whether to use random initial states.
                normalized_state: Whether to normalize state values.
                path_reward_weight: Weight assigned to path-based reward shaping.
                seed: Random seed for reproducibility.
                reward_function: Reward function selector.
                gamma: Discount factor.
                record_scenario: Whether to export scenario event logs.
                scenario_path: Base path for scenario CSV files.
            """

        env_parameters = {k: v for k, v in locals().items() if k != "self"}

        if reward_function == 0:
            self.get_reward = self.get_reward

        self.signature = {}
        self.original_seed = seed
        self.rng = [np.random.default_rng(self.original_seed)]

        self.max_id = 1
        self.gamma = gamma
        self.max_steps = max_steps
        self.log_flag = logging
        self.path_reward_weight = path_reward_weight

        self.load_conf(conf_name)
        self.conf_name = conf_name

        self.augmented = augment
        self.random_start = random_start
        self.normalized_state = normalized_state
        self.feature_number = FEATURE_NUMBER

        self.score = Score()
        self.episode = []
        self.available_actions = []

        self.logname = Path(logname)
        self.save_episodes = save_episodes
        self.transpose_state = transpose_state
        self.last_box_from_EP = False

        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        if self.augmented:
            self.augment_factor = ceil(MIN_SB3_SIZE / min(self.grid.shape))
            size = tuple(dimension * self.augment_factor for dimension in self.grid.shape)
        else:
            size = self.grid.shape

        all_positions = [
            (i, j)
            for i in range(self.grid.shape[0])
            for j in range(self.grid.shape[1])
        ]

        if hasattr(self, "outside_cells"):
            self.valid_positions = [pos for pos in all_positions if pos not in self.outside_cells]
        else:
            self.valid_positions = all_positions

        self.action_space = spaces.Discrete(len(self.valid_positions))
        self.observation_space = spaces.Box(
            low=0.0, high=255.0, shape=(size[0], size[1], self.feature_number), dtype=np.uint8
        )

        self.material = {}
        self.agents = [Agent((0, 0)) for _ in range(self.num_agents)]

        self.ter = False
        self.tru = False
        self.action = None
        self.floor_graph = None
        self.path = []
        self.num_actions = 0
        self.current_return = 0
        self.action_mask = np.zeros(len(list(range(self.action_space.n))))
        self.full_storage_flag = False

        # Scenario logging
        self.record_scenario = record_scenario
        self.scenario_path = scenario_path
        self.episode_id = 0
        self.current_step = 0

        if record_scenario:
            layout_name = conf_name
            self.idScenario = f"sc_{layout_name}_s{self.original_seed:03d}"

            if scenario_path is not None:
                self.scenario_path = scenario_path
            else:
                self.scenario_path = os.path.join("results", self.idScenario)

            os.makedirs(os.path.dirname(self.scenario_path), exist_ok=True)
        else:
            self.scenario_path = scenario_path

        self.csv_file_config = None
        self.csv_writer_config = None
        self.csv_file_steps = None
        self.csv_writer_steps = None

        if self.random_start:
            self.random_initial_states = self.create_random_initial_states(NUM_RANDOM_STATES)

        if save_episodes:
            self.episode_folder = self.logname / "episodes"
            self.episode_folder.mkdir(parents=True, exist_ok=True)

        if self.log_flag:
            self.create_logfile(env_parameters)

    def create_logfile(self, env_params):
        """Create CSV and JSON files used to store environment metrics and parameters."""
        self.logname.mkdir(parents=True, exist_ok=True)

        self.metrics_log = f"{str(self.logname / self.logname.name)}_metrics.csv"
        with open(self.metrics_log, "a") as f:
            f.write(self.score.print_header())

        with open(self.logname / "env_parameters.json", "w") as f:
            json.dump(
                {**env_params, "initial_rng": self.rng[0].bit_generator.state},
                f,
                indent=4
            )

    def load_conf(self, conf: str = CONF_NAME):
        """
        Load environment configuration from conf.json.

        Args:
            conf: Configuration key to load from the JSON file.
        """
        with open(os.path.join(os.path.dirname(__file__), "conf.json"), "r") as f:
            current_conf = json.load(f)[conf]

        self.grid = np.zeros(current_conf["grid"])
        conf = current_conf["conf"]

        self.type_information = conf["material_types"]

        self.entrypoints = [
            Entrypoint(
                position=eval(ii),
                type_information=self.type_information,
                rng=self.rng
            )
            for ii in conf["entrypoints"]
        ]

        self.outpoints = Outpoints(
            [eval(ii) for ii in conf["outpoints"]],
            type_information=self.type_information,
            delivery_prob=conf["delivery_prob"],
            rng=self.rng,
        )

        self.num_agents = conf["num_agents"]  # Numero de agentes

        if "forbidden" in conf:
            self.outern_crown = [tuple(pos) for pos in conf["forbidden"]]
        else:
            self.outern_crown = self.declare_outern_crown()
        if "outside" in conf:
            self.outside_cells = [tuple(pos) for pos in conf["outside"]]

    def declare_outern_crown(self):
        """
        Return the outer crown of the grid, excluding entrypoints and outpoints.

        This region is used as a restricted or overflow area depending on
        the environment configuration.
        """
        shape_x, shape_y = self.grid.shape
        crown = []
        for ii in range(shape_x):
            crown.extend(((ii, 0), (ii, shape_y - 1)))
        for ii in range(shape_y):
            crown.extend(((0, ii), (shape_x - 1, ii)))
        crown = list(set(crown))
        for op in self.outpoints.outpoints:
            crown.remove(op)
        for ep in self.entrypoints:
            crown.remove(ep.position)
        return crown

    def outpoints_consume(self):
        """
        Consume boxes currently placed on outpoints if they match an active
        and ready delivery request.
        """
        for outpoint in self.outpoints.outpoints:
            if self.grid[outpoint] > 0:
                try:
                    status = self.outpoints.consume(self.material[self.grid[outpoint]])
                    if status == 1:
                        logging.info("Material consumed")
                    elif status == 2:
                        logging.info("Order completed")
                        self.score.filled_orders += 1
                    self.score.box_ages.append(self.material[self.grid[outpoint]].age)
                    del self.material[self.grid[outpoint]]
                    self.grid[outpoint] = 0
                except Exception as e:
                    logging.error(f"Unexpected error at consuming the material at outpoint {outpoint}: {e}")
                    raise Exception from e

    @staticmethod
    def prepare_grid(matrix: np.array, start: tuple, end: tuple, whitelist: list = None) -> Grid:
        """
        Prepare a grid for pathfinding.

        Cells in `whitelist`, as well as the start and end positions, are forced
        to be traversable.
        """
        prepared_matrix = np.array(matrix, dtype="int16")
        prepared_matrix[start] = 0
        prepared_matrix[end] = 0

        for cell in whitelist:
            prepared_matrix[cell] = 0
        return Grid(matrix=np.negative(prepared_matrix) + 1)

    def find_path_cost(self, start_position, end_position) -> int:
        """
        Compute the path cost between two positions using A*.

        Returns:
            Path length minus one. Returns 0 when start and end are the same.
        """
        grid = self.prepare_grid(
            self.grid,
            start_position,
            end_position,
            whitelist=[ep.position for ep in self.entrypoints] + self.outpoints.outpoints,
        )

        start = grid.node(*reversed(start_position))
        end = grid.node(*reversed(end_position))
        path, runs = self.finder.find_path(start, end, grid)

        self.path = path
        return len(path) - 1

    @staticmethod
    def __get_age_factor(age):
        """
        Legacy age-based weighting function.

        Age is bounded to [0, 500] and mapped to a value in [0, 1].
        """
        bound = 500
        bounded_age = min(max(abs(age), 0), bound) / bound
        return (1 - bounded_age) ** 2 + (bounded_age) ** 2

    @staticmethod
    def get_age_factor(age, old_age):
        """
        Compute a normalized age difference between two boxes.

        Returns a value in [0, 1].
        """
        bound = 100
        return min(max(abs(age - old_age), 0), bound) / bound

    def get_oldest_box(self, box_type) -> int:
        """
        Return the oldest available box of the given type, considering both
        stored boxes and the head element of each entrypoint queue.
        """
        candidates = [
                         material for material in self.material.values() if material.type == box_type
                     ] + [
                         ep.material_queue[0]
                         for ep in self.entrypoints
                         if len(ep.material_queue)
                            and ep.material_queue[0].type == box_type
                            and self.path
                            and self.path[0] != ep.position
                     ]

        if not candidates:
            return None

        return max(candidates, key=operator.attrgetter("age"))

    def delivery_reward(self, box):
        """
        Reward associated with delivering a box.

        A penalty is applied when the delivered box is not the oldest available
        box of the same type.
        """
        min_rew = -0.5
        oldest_box = self.get_oldest_box(box.type)

        age_factor = self.get_age_factor(box.age, oldest_box.age)
        self.score.delivered_boxes += 1

        if box.id != oldest_box.id:
            self.score.non_optimal_material += 1
            return min_rew * age_factor
        return 0.0

    def get_entrypoints_with_items(self):
        """
        Return the list of entrypoints whose queue is not empty.
        """
        try:
            entrypoints_with_items = [ep for ep in self.entrypoints if len(ep.material_queue) > 0]
        except IndexError:
            entrypoints_with_items = []
        return entrypoints_with_items

    def __LEGACY_get_macro_action_reward(self, ag: Agent, box: Box = None) -> float:
        """
        Legacy macro-level reward function kept for backward compatibility.

        It rewards or penalizes actions depending on whether there are ready
        deliveries, available boxes, and whether the agent is positioned at
        an outpoint with a deliverable box.
        """
        if self.get_ready_to_consume_types():
            if ag.position in self.outpoints.outpoints:
                return self.delivery_reward(box) if box is not None else -1  # cambiado a -1
            return -0.9 if len(self.material) or self.get_entrypoints_with_items() else 0
        elif not self.get_ready_to_consume_types() and self.get_entrypoints_with_items():
            return -0.9
        else:
            return 0

    @staticmethod
    def normalize_path_cost(cost: int, grid_shape: tuple) -> float:
        """
        Normalize path cost into a negative penalty proportional to grid size.
        """
        return -cost / (prod(grid_shape) / 2)
 
    def __new_reward(self):
        """
        Penalize the presence of aging boxes in the environment.

        The reward is based on the total age of all boxes currently stored
        in the warehouse and in entrypoint queues.
        """
        return max(
            -0.9,
            (
                -sum(
                    [box.age for box in self.material.values()]
                    + [box.age for sublist in [ep.material_queue for ep in self.entrypoints] for box in sublist]
                )
                / 10000
            ),
        )

    def __get_reward_2(self, move_status: int, ag: Agent) -> float:
        """
        Alternative reward function based on aging-box penalty.
        """
        new_r = self.__new_reward()
        if move_status == 0:
            return -0.5 + new_r
        elif move_status == 2 and ag.position in self.outpoints.outpoints:
            self.score.delivered_boxes += 1
            return 0
        else:
            return new_r

    def __get_reward_1(self, move_status: int, ag: Agent) -> float:
        """
       Simple reward function:
       - +1 for a successful delivery
       - -1 otherwise
       """
        if move_status == 2 and ag.position in self.outpoints.outpoints:
            self.score.delivered_boxes += 1
            return 1
        return -1

    def check_idle(self):
        """
        Return True when the agent is effectively idle:
        - it is not carrying any box,
        - there are no ready deliveries that can be fulfilled,
        - and no entrypoint currently has boxes available.
        """
        consumable_types = self.get_ready_to_consume_types()
        return (
            not self.agents[0].got_item
            and (not consumable_types or not consumable_types & {box.type for box in self.material.values()})
            and not self.get_entrypoints_with_items()
        )

    def get_reward(self, move_status: int) -> float:
        """
        Main reward function.

        Move status codes:
            0: invalid action
            1: take from entrypoint
            2: take from grid
            3: idle/procrastination
            4: drop at outpoint
            5: drop in storage/grid
            6: drop in outer crown because storage is full
        """
        if move_status == 0:
            return -1
        elif move_status == 4:
            return self.delivery_reward(self.material[self.grid[self.agents[0].position]])
        elif move_status == 6:
            return -0.8
        elif self.check_idle():
            return 0
        elif move_status == 1:
            return -0.7
        elif move_status == 2:
            return -0.5
        elif move_status == 3:
            return -0.9
        elif move_status == 5:
            return -0.5
        else:
            return -1

    def __LEGACY_get_reward(self, move_status, ag, box) -> float:
        """
        Legacy reward function kept for backward compatibility.
        """
        if move_status == 0:
            return -1
        if move_status == 1 and not self.get_ready_to_consume_types():
            return 0

        macro_action_reward = self.__LEGACY_get_macro_action_reward(ag, box)
        weighted_reward = macro_action_reward

        return weighted_reward

    def log(self):
        """
        Store episode-level metrics and, optionally, the full episode trace.
        """
        self.score.box_ages += [box.age for box in self.material.values()]
        self.score.max_id = self.max_id
        self.score.seed = self.original_seed
        self.score.discounted_return = np.mean(
            [
                sum(self.gamma**ii * ret for ii, ret in enumerate(self.score.returns[jj:]))
                for jj in range(len(self.score.returns))
            ]
        )
        if not len(self.score.box_ages):
            self.score.box_ages.append(0)
        with open(self.metrics_log, "a") as f:
            f.write(self.score.print_score())
        if self.save_episodes:
            with open(f"{self.episode_folder / self.logname.name}_episode_{EPISODE}.json", "w") as f:
                json.dump(self.episode, f)

    @staticmethod
    def normalize_age(age: int, oldest_box_age: int) -> float:
        """
        Map a box age to the [0, 255] range for state representation.
        """
        return ceil(min(max(age, 1), oldest_box_age) / oldest_box_age * 255)

    def normalize_type(self, type: str) -> int:
        """
        Map a material type (e.g., 'A', 'B') to its encoded value.
        """
        return TYPE_CODIFICATION[type]

    def get_ready_to_consume_types(self) -> dict:
        """
        Return the set of material types associated with ready delivery requests.
        """
        try:
            return {order.type for order in self.outpoints.delivery_schedule if order.ready}
        except IndexError:
            return {}

    def check_reachable(self, position: tuple, maze: np.array) -> bool:
        """
        Return True if the given cell is adjacent to a reachable cell.

        A cell is considered reachable if at least one of its four neighbors
        has value 0.5 in the flood-filled maze.
        """
        adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for delta in adjacent_cells_delta:
            adjacent_cell = tuple(map(operator.add, position, delta))
            try:
                if maze[adjacent_cell] == 0.5:
                    return True
            except IndexError:
                continue
        return False

    def get_available_actions(self) -> list:
        """
        Compute the boolean action mask for the current agent state.

        Valid actions depend on:
        - flood-filled reachable cells,
        - whether the agent is carrying a box,
        - current delivery requests,
        - storage capacity constraints,
        - forbidden and outside cells.
        """
        agent = self.agents[0]
        maze = np.array(self.grid)
        for ep in self.entrypoints:
            maze[ep.position] = 0
        for op in self.outpoints.outpoints:
            maze[op] = 0
        maze[agent.position] = 0
        maze = flood_fill(maze, agent.position, 0.5, connectivity=1)

        if self.check_full_storage():
            for cell in self.outern_crown:
                maze[cell] = 0.5
        else:
            for cell in self.outern_crown:
                maze[cell] = 0
        # MANUAL
        #for cell in self.outern_crown:
        #    maze[cell] = 0.5  # siempre se puede caminar por la corona
        #if self.check_full_storage():
        #    pass  # si está lleno, podrá además dejar cajas (ya gestionado en otro método)

        for cell in self.outside_cells:
            maze[cell] = 0
        if agent.got_item:
            for ep in self.entrypoints:
                maze[ep.position] = 0
            if self.material[agent.got_item].type not in self.get_ready_to_consume_types():
                for op in self.outpoints.outpoints:
                    maze[op] = 0
        else:
            available_boxes = [pos for pos in np.argwhere(self.grid > 0) if self.check_reachable(pos, maze)]
            for pos in available_boxes:
                maze[tuple(pos)] = 0.5
            for op in self.outpoints.outpoints:
                maze[op] = 0
            for ep in self.entrypoints:
                if len(ep.material_queue) == 0:
                    maze[ep.position] = 0

        maze[agent.position] = 0

        mask = np.zeros(len(self.valid_positions), dtype=bool)
        for idx, pos in enumerate(self.valid_positions):
            if maze[pos] == 0.5:
                mask[idx] = True
        self.action_mask = mask

        # Fallback safeguard: ensure at least one valid action
        if not self.action_mask.any():
            self.action_mask[:] = True  # habilita todo como fallback

        return self.action_mask

    def set_signature(self, signature: dict) -> None:
        """
        Restore a previously saved environment snapshot.
        """
        _, _ = self.reset(options={"force_clean": True})

        self.ter = signature["ter"]
        self.tru = signature["tru"]
        self.rng[0].bit_generator.state = signature["rng"]

        self.agents = [Agent(agent["pos"], agent["item_id"]) for agent in signature["agents"]]

        self.material = {box["id"]: Box(box["id"], box["pos"], box["type"], box["age"]) for box in signature["boxes"]}

        self.outpoints.delivery_schedule = [
            Delivery(type=el.type, prob=el.prob, num_boxes=el.num_boxes, timer=el.timer, ready=el.ready, rng=self.rng)
            for el in signature["outpoints"]["delivery_schedule"]
        ]
        self.outpoints.last_delivery_timers = signature["outpoints"]["last_delivery_timers"]
        for ep, info in zip(self.entrypoints, signature["entrypoints"]):
            ep.material_queue = [Box(el.id, el.position, el.type, el.age) for el in info["material_queue"]]
            ep.position = info["pos"]
        self.num_actions = signature["num_actions"]
        for box_id, box in list(self.material.items()) + [
            (queue[0].id, queue[0]) for queue in [ep.material_queue for ep in self.entrypoints if len(ep.material_queue) > 0]
        ]:
            self.grid[box.position] = box_id
        if self.agents[0].got_item:
            self.grid[self.agents[0].position] = 0
        self.max_id = signature["max_id"]
        self.signature = signature

    def get_signature(self) -> dict:
        """
        Return a full snapshot of the current environment state.
        """
        return {
            "max_id": self.max_id,
            "ter": self.ter,
            "tru": self.tru,
            "rng": self.rng[0].bit_generator.state,
            "boxes": [
                {
                    "id": id_box,
                    "pos": box.position,
                    "age": box.age,
                    "type": box.type,
                }
                for id_box, box in self.material.items()
            ],
            "agents": [
                {
                    "pos": agent.position,
                    "item": self.material[agent.got_item].type if agent.got_item > 0 else 0,
                    "item_id": int(agent.got_item) if agent.got_item > 0 else 0,
                }
                for agent in self.agents
            ],
            "entrypoints": [
                {
                    "pos": ep.position,
                    "material_queue": [Box(el.id, el.position, el.type, el.age) for el in ep.material_queue],
                }
                for ep in self.entrypoints
            ],
            "outpoints": {
                "pos": list(self.outpoints.outpoints),
                "accepted_types": list(set(self.get_ready_to_consume_types())),
                "delivery_schedule": [
                    Delivery(type=el.type, prob=el.prob, num_boxes=el.num_boxes, timer=el.timer, ready=el.ready, rng=self.rng)
                    for el in self.outpoints.delivery_schedule
                ],
                "last_delivery_timers": self.outpoints.last_delivery_timers,
            },
            "num_actions": self.num_actions,
        }

    def save_state_simplified(self, reward: int, action: tuple):
        state = self.get_signature()
        self.episode.append(
            {
                "step": action,
                "num_actions": state["num_actions"],
                "reward": reward,
                "cum_reward": self.current_return,
                "path": self.path,
                "state": {
                    "agents": state["agents"],
                    "outpoints": {"pos": state["outpoints"]["pos"], "accepted_types": state["outpoints"]["accepted_types"]},
                    "entrypoints": {"pos": [ep["pos"] for ep in state["entrypoints"]]},
                    "boxes": state["boxes"],
                },
            }
        )

    @staticmethod
    def augment_state(box_grid, age_grid, agent_grid, augment_factor) -> np.array:
        """
        Upscale the state representation by repeating each cell value.

        This is useful when training CNN-based agents that require a minimum
        input size.
        """
        return np.array(
            [np.kron(grid, np.ones((augment_factor, augment_factor))) for grid in np.array([box_grid, age_grid, agent_grid])]
        )

    def mix_state(self, box_grid, age_grid, agent_grid):
        """
        Return the state either in its original resolution or augmented form.
        """
        return (
            self.augment_state(box_grid, age_grid, agent_grid, self.augment_factor)
            if self.augmented
            else np.array([box_grid, age_grid, agent_grid])
        )

    @staticmethod
    def normalize_state(state_mix):
        """
        Normalize state values to the [0, 1] range.
        """
        for ii, matrix in enumerate(state_mix):
            state_mix[ii] = matrix / 255
        return state_mix

    @staticmethod
    def type_to_int(box_type: str) -> int:
        """
        Convert a box type such as 'A', 'B', 'C', ... into 0, 1, 2, ...
        """
        return ord(box_type) - ord("A")

    def normalize_type_combination(self, ready_to_consume_types: list, num_types: int) -> int:
        """
        Encode a set of ready-to-consume box types into a single integer value.

        The encoding is based on the binary combination of active types.
        """
        num = sum([2 ** self.type_to_int(consume_type) for consume_type in ready_to_consume_types] + [0])
        return TYPE_COMB_CODIFICATION[num]

    def construct_age_grid(self, age_grid):
        """
        Fill the age channel of the state.

        Ages are assigned to:
        - boxes currently stored in the warehouse
        - the first box in each entrypoint queue
        """
        oldest_boxes = {
            box_type: oldest.age
            for box_type in self.type_information.keys()
            if (oldest := self.get_oldest_box(box_type)) is not None
        }
        for box in list(self.material.values()) + [ep.material_queue[0] for ep in self.entrypoints if ep.material_queue]:
            if box.type in oldest_boxes:
                age_grid[box.position] = self.normalize_age(box.age, oldest_boxes[box.type])
        return age_grid

    def construct_box_grid(self, box_grid):
        """
        Fill the box-type channel of the state.

        This includes:
        - boxes currently stored in the warehouse
        - the first box in each entrypoint queue
        - boxes currently carried by agents when standing on entrypoints
        - encoded ready delivery types on outpoints
        """
        for box in list(self.material.values()) + [ep.material_queue[0] for ep in self.get_entrypoints_with_items()]:
            box_grid[box.position] = self.normalize_type(box.type)
        for agent in self.agents:
            if agent.position in [ep.position for ep in self.entrypoints] and agent.got_item:
                box_grid[agent.position] = self.normalize_type(self.material[agent.got_item].type)
        ready_to_consume_types = self.get_ready_to_consume_types()
        for pos in self.outpoints.outpoints:
            box_grid[pos] = self.normalize_type_combination(ready_to_consume_types, len(self.type_information))
        return box_grid

    def construct_agent_grid(self, agent_grid):
        """
        Fill the agent channel of the state.

        Encoding:
        - 64: agent without a box
        - 128: agent carrying type A
        - 192: agent carrying type B
        """
        for agent in self.agents:
            if agent.got_item:
                mat_type = self.material[agent.got_item].type
                if mat_type == "A":
                    agent_grid[agent.position] = 128
                elif mat_type == "B":
                    agent_grid[agent.position] = 192
            else:
                agent_grid[agent.position] = 64

        return agent_grid

    def initialize_grids(self):
        """
        Initialize the three state channels:
        - box type grid
        - age grid
        - agent grid
        """
        return (
            np.zeros(self.grid.shape, dtype=np.uint8),
            np.zeros(self.grid.shape, dtype=np.uint8),
            np.zeros(self.grid.shape, dtype=np.uint8),
        )

    def construct_grids(self):
        """
        Construct all state grids from the current environment state.
        """
        box_grid, age_grid, agent_grid = self.initialize_grids()
        return self.construct_box_grid(box_grid), self.construct_age_grid(age_grid), self.construct_agent_grid(agent_grid)

    def get_state(self) -> list:
        """
        Build and return the current environment state.

        Depending on configuration, the state may be:
        - augmented for CNN-based agents
        - normalized to [0, 1]
        - returned either channel-first or channel-last
        """
        box_grid, age_grid, agent_grid = self.construct_grids()
        state_mix = self.mix_state(box_grid, age_grid, agent_grid)
        size = state_mix[0].shape
        if self.normalized_state:
            state_mix = self.normalize_state(state_mix)
        return (state_mix if self.transpose_state else state_mix.reshape(size + (self.feature_number,))).astype("uint8")

    def check_full_storage(self):
        """
        Return True if the internal storage area is completely full.
        """
        x_size, y_size = self.grid.shape
        storage_filter = np.zeros((x_size - 4, y_size - 4))
        mask = np.pad(storage_filter, ((1, 1), (1, 1)), "constant", constant_values=(1, 1))
        mask = np.pad(mask, ((1, 1), (1, 1)), "constant", constant_values=(0, 0))
        storage = self.grid[mask.astype(bool)]
        self.full_storage_flag = all(storage > 0)
        return self.full_storage_flag

    def assert_movement(self, ag: Agent, movement: tuple) -> int:
        """
        Validate whether a movement is legal.

        A movement is valid if:
        - it is inside grid bounds
        - indices are non-negative
        - it is allowed by the current action mask
        - there is a valid path to the target position
        """
        try:
            _ = self.grid[movement]
            assert all(ii >= 0 for ii in movement)
            assert self.action_mask[self.denorm_action(
                movement)], f"Action not allowed by mask (mask[{movement}]={self.action_mask[self.denorm_action(movement)]})"
            assert self.find_path_cost(ag.position, movement) >= 0
        except (AssertionError, IndexError) as e:
            self.score.num_invalid += 1
            return 0
        return 1

    def move_agent(self, ag: Agent, movement: tuple) -> int:  # Este método realiza el movimiento de un agente y devuelve un código según el tipo de acción que ha hecho
        """
        Execute an agent movement and return the resulting action code.

        Returns:
            0: invalid action
            1: successful pickup from an entrypoint
            2: successful pickup from the warehouse grid
            3: move to an empty cell without carrying a box
            4: successful drop at an outpoint
            5: successful drop in the warehouse grid
            6: drop in the outer crown when storage is full
        """
        if not self.assert_movement(ag, movement):
            return 0
        if self.grid[movement] > 0:
            return self.take_item(ag, movement)
        else:
            return self.drop_item(ag, movement)

    def drop_item(self, ag, movement):
        """
        Handle the case where the agent moves to an empty cell.

        If the agent is not carrying a box, the action is considered an idle move.
        If the agent is carrying a box, it is dropped either in storage, at an
        outpoint, or in the outer crown when storage is full.
        """
        if not ag.got_item:
            ag.position = movement
            return 3
        ag.position = movement
        if self.full_storage_flag and movement in self.outern_crown:
            self.material[ag.got_item].position = movement
            return 6
        self.grid[ag.position] = ag.got_item
        self.material[ag.got_item].position = movement
        ag.got_item = 0
        return 4 if movement in self.outpoints.outpoints else 5

    def take_item(self, ag, movement):
        """
        Handle the case where the agent moves to an empty cell.

        If the agent is not carrying a box, the action is considered an idle move.
        If the agent is carrying a box, it is dropped either in storage, at an
        outpoint, or in the outer crown when storage is full.
        """
        if ag.got_item:
            return 0
        ag.position = movement
        ag.got_item = self.grid[ag.position]
        self.grid[ag.position] = 0
        return 1 if movement in [ep.position for ep in self.entrypoints] else 2

    def _step(self, action: tuple, render=False) -> tuple:
        """
        Internal step function operating on coordinate-based actions.

        It updates the environment, computes rewards, advances timers, creates
        new delivery requests when applicable, and returns the Gym-compatible
        transition tuple.
        """
        action = (int(action[0]), int(action[1]))
        self.last_action = self.denorm_action(action)
        self.num_actions += 1

        info = {"Steps": self.num_actions}
        agent = self.agents[0]
        ter = False
        tru = False

        if self.num_actions >= self.max_steps:
            ter = True
            reward = 0
            info["done"] = "Max movements achieved. Well done!"

            if self.log_flag:
                self.log()
            return self.return_result(reward, ter, tru, info, action)

        self.score.steps += 1

        if not ter or tru:
            reward, move_status = self.act(agent, action, info)
        else:
            info["Info"] = "Done. Please reset the environment"
            reward = -1e3
            return self.return_result(reward, ter,  tru, info, action)

        self.outpoints_consume()
        self.update_timers()

        order = self.outpoints.create_delivery()
        if order is not None and self.log_flag:
            self.score.total_orders += 1

        if not any(self.get_available_actions()):
            ter = True
            reward = -1e3
            info["done"] = "Not any valid actions found. Reset."
            self.score.trapped += 1
            return self.return_result(reward, ter, tru, info, action)

        if render:
            self.render()

        if self.record_scenario:
            self._log_step_events()

        return self.return_result(reward, ter, tru, info, action)

    def close_scenario_logs(self):
        """
        Close open CSV files used for scenario logging.
        """
        if hasattr(self, "csv_file_config") and self.csv_file_config:
            self.csv_file_config.close()
        if hasattr(self, "csv_file_steps") and self.csv_file_steps:
            self.csv_file_steps.close()

    def update_timers(self):
        """
        Advance environment time based on the last planned path length.

        This updates:
        - the global episode timer
        - the age of all stored boxes
        - delivery request timers
        - entrypoint queues and newly generated boxes
        """
        steps = len(self.path) - 1  
        self.score.timer += steps
        for box in self.material.values():
            box.update_age(steps)
        self.outpoints.update_timers(steps)
        for entrypoint in self.entrypoints:
            self.max_id = entrypoint.update_entrypoint(max_id=self.max_id, steps=steps)  
            try:
                self.grid[entrypoint.position] = entrypoint.material_queue[0].id
            except IndexError as ex:
                self.grid[entrypoint.position] = 0

    def act(self, agent, action, info):
        """
        Execute the agent action and compute the associated reward.

        This method also updates the environment state when a box is picked up
        from an entrypoint queue.
        """
        carried_box_id = agent.got_item
        box = None
        if agent.got_item:
            box = self.material[agent.got_item]

        move_status = self.move_agent(agent, action)

        if move_status == 2:
            box = self.material[agent.got_item]

        if move_status == 1:
            box = [entrypoint for entrypoint in self.entrypoints if entrypoint.position == agent.position][0].get_item()
            self.material[box.id] = box

        reward = self.__LEGACY_get_reward(move_status, agent, box)
        self.score.returns.append(reward)

        return reward, move_status

    def return_result(self, reward, ter, tru, info, action):
        """
        Package the current transition into the Gymnasium return format.
        """
        self.last_r = reward
        self.current_return += reward
        self.ter = ter
        self.tru = tru

        info["timer"] = self.score.timer
        info["delivered"] = self.score.delivered_boxes
        info["outpoint queue"] = {
            t: sum((deliver.num_boxes) for deliver in self.outpoints.delivery_schedule if deliver.type == t and deliver.ready)
            for t in self.get_ready_to_consume_types()
        }
        info["orders"] = [deliver for deliver in self.outpoints.delivery_schedule]

        for entrypoint in self.entrypoints:
            info[f"EP{entrypoint.position}"] = list(entrypoint.material_queue)

        self.last_info = info
        self.save_state_simplified(reward, action)

        return self.get_state(), reward, ter, tru, info

    def norm_action(self, action_idx: int) -> tuple:
        """
        Convert a discrete action index into a valid grid position.
        """
        assert 0 <= action_idx < len(self.valid_positions)
        return self.valid_positions[action_idx]

    def denorm_action(self, action: tuple) -> int:
        """
        Convert a valid grid position into its discrete action index.
        """
        return self.valid_positions.index(action)

    def step(self, action: int) -> tuple:
        """
        Standard Gymnasium step function using a discrete action index.
        """
        self.action = self.norm_action(action)
        assert action == self.denorm_action(self.action)

        state, reward, ter, tru, info = self._step(self.action)
        return state, reward, ter, tru, info

    def create_random_box(self, position: tuple, type: str = None, age: int = None):
        """
        Create a random box at the given position.

        If type or age are not provided, they are sampled randomly.
        """
        box = Box(
            id=self.max_id,
            position=position,
            type=type or self.rng[0].choice(list(self.type_information.keys())),
            age=age or self.rng[0].choice(range(1, 100)),
        )
        self.max_id += 1
        return box

    def assign_order_to_material(self):
        """
        Create delivery requests based on the current material stored
        in the warehouse.
        """
        def decomposition(i):
            while i > 0:
                try:
                    n = self.rng[0].integers(MIN_NUM_BOXES, min(i, MAX_NUM_BOXES) + 1)
                except ValueError:
                    n = self.rng[0].integers(1, min(i, MAX_NUM_BOXES) + 1)
                yield n
                i -= n

        for type, info in self.type_information.items():
            num_boxes_type = len([box for box in self.material.values() if box.type == type])
            num_boxes_distribution = decomposition(num_boxes_type)

            for num_boxes in num_boxes_distribution:
                self.outpoints.delivery_schedule.append(
                    Delivery(type=type, prob=info["deliver"], num_boxes=num_boxes, rng=self.rng)
                )

    def create_random_initial_states(self, num_states) -> list:
        """
        Pre-generate random initial environment states for reproducible training.
        """
        self.rng[0] = np.random.default_rng(self.original_seed)
        states = []

        print("Creating random states...")
        t0 = time()

        for _ in range(num_states):
            self.reset_random()
            states.append(self.get_signature())

        print(f"Finished! Created {num_states} states in {time() - t0}s")
        return states

    def set_search(self):
        """
        Disable logging and episode saving for search-based methods such as AlphaZero.
        """
        self.log_flag = False
        self.save_episodes = False

    def reset(self, *, seed=None, options=None) -> tuple:
        """
        Reset the environment to its initial state.

        Options:
            render: whether to render the initial state
            force_clean: if True, skip random initial state restoration
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng[0] = np.random.default_rng(seed)

        if self.original_seed:
            self.rng[0] = np.random.default_rng(self.original_seed)


        render = options.get("render", False) if options else False
        force_clean = options.get("force_clean", False) if options else False

        global EPISODE
        EPISODE += 1

        self.max_id = 1
        random_flag = self.random_start

        if self.original_seed is not None:
            self.rng[0] = np.random.default_rng(self.original_seed)
        else:
            self.rng[0] = np.random.default_rng()

        self.signature = {}
        self.episode = []
        self.grid = np.zeros(self.grid.shape)
        self.num_actions = 0
        self.current_return = 0
        self.material = {}
        self.last_action = 0
        self.last_r = 0
        self.last_info = {}

        self.outpoints.reset()  # Deja vacíos los outpoints
        self.full_storage_flag = False

        for entrypoint in self.entrypoints:
            entrypoint.reset()

        if random_flag and not force_clean:
            self.set_signature(self.rng[0].choice(self.random_initial_states))
        else:
            self.agents = [Agent(initial_position=(3, 3)) for _ in range(self.num_agents)]

        self.ter = False
        self.tru = False
        self.score.reset()
        self.number_actions = 0

        if not any(self.get_available_actions()):
            return self.reset(options={"render": render})

        if render:
            self.render()

        return self.get_state(), {}

    def reset_random(self):
        """
        Reset the environment and populate it with a random initial configuration.
        """
        _, _ = self.reset(options={"force_clean": True})

        box_probability = 0.3

        self.agents = [
            Agent(
                (1, 1),
                got_item=self.rng[0].choice([0, self.max_id]),
            )
            for _ in range(self.num_agents)
        ]
        if self.agents[0].got_item:
            self.material[self.agents[0].got_item] = self.create_random_box(position=self.agents[0].position)

        forbidden_positions = [ep.position for ep in self.entrypoints] + list(self.outpoints.outpoints)

        for row, col in itertools.product(range(self.grid.shape[0]), range(self.grid.shape[1])):
            if (row, col) == self.agents[0].position:
                continue
            if (row, col) in forbidden_positions:
                continue
            if (row, col) in self.outern_crown:
                continue
            if (row, col) in self.outside_cells:
                continue
            if self.rng[0].random() < box_probability:
                max_id = self.max_id
                self.material[max_id] = self.create_random_box((row, col))

        for box in list(self.material.values()):
            self.grid[box.position] = box.id

        if self.agents[0].got_item:
            self.grid[self.agents[0].position] = 0

        if self.record_scenario:
            self._init_scenario_csv()
            self._log_initial_state()

        self.current_step = 0

    def _init_scenario_csv(self, only_steps: bool = False):
        """
        Initialize scenario-level CSV files.

        If only_steps is False, both the static configuration CSV and the
        per-step CSV are created. Otherwise, only the per-step CSV is created.
        """
        base = self.scenario_path

        if not only_steps:
            config_file = os.path.join(base, f"{self.idScenario}_config.csv")
            self.csv_file_config = open(config_file, mode="w", newline="")
            self.csv_writer_config = csv.writer(self.csv_file_config)
            self.csv_writer_config.writerow([
                "layout",
                "grid_shape",
                "create_A",
                "deliver_A",
                "create_B",
                "deliver_B",
                "delivery_prob",
                "entrypoints_positions",
                "outpoints_positions",
                "forbidden_positions",
                "outside_positions",
            ])

            grid_shape = f"{self.grid.shape[0]}x{self.grid.shape[1]}"
            entrypoints_positions = [[int(ep.position[0]), int(ep.position[1])] for ep in self.entrypoints]
            outpoints_positions = [[int(pos[0]), int(pos[1])] for pos in self.outpoints.outpoints]
            forbidden_positions = getattr(self, "outern_crown", [])
            outside_positions = getattr(self, "outside_cells", [])

            create_A = self.type_information.get("A", {}).get("create", None)
            deliver_A = self.type_information.get("A", {}).get("deliver", None)
            create_B = self.type_information.get("B", {}).get("create", None)
            deliver_B = self.type_information.get("B", {}).get("deliver", None)
            delivery_prob = getattr(self.outpoints, "delivery_prob", None)

            self.csv_writer_config.writerow([
                getattr(self, "conf_name", "unknown"),
                grid_shape,
                create_A,
                deliver_A,
                create_B,
                deliver_B,
                delivery_prob,
                json.dumps(entrypoints_positions),
                json.dumps(outpoints_positions),
                json.dumps(forbidden_positions),
                json.dumps(outside_positions),
            ])

            self.csv_file_config.close()

        steps_file = os.path.join(base, f"{self.idScenario}_steps.csv")
        self.csv_file_steps = open(steps_file, mode="w", newline="")
        self.csv_writer_steps = csv.writer(self.csv_file_steps)

        self.csv_writer_steps.writerow([
            "step",
            "seed",
            "boxes_initial",
            "agent_state_initial",
            "entrypoints_state_initial",
            "initial_orders",
            "entrypoints_events",
            "outpoints_orders_created",
        ])

    def _log_initial_state(self):
        """
        Log the initial scenario state as step 0 in the per-step CSV.
        """
        boxes = [
            {
                "pos": [int(box.position[0]), int(box.position[1])],
                "type": box.type,
                "id": int(box_id),
                "age": int(box.age),
            }
            for box_id, box in self.material.items()
        ]

        if self.agents[0].got_item:
            carried_box = self.material[self.agents[0].got_item]
            got_item_data = {
                "id": int(carried_box.id),
                "type": carried_box.type,
                "age": int(carried_box.age),
            }
        else:
            got_item_data = 0

        agent_state = {
            "pos": [int(self.agents[0].position[0]), int(self.agents[0].position[1])],
            "got_item": got_item_data
        }

        initial_orders = [
            {
                "op": [int(o.position[0]), int(o.position[1])],
                "type": o.type,
                "pending": int(o.pending)
            }
            for o in self.outpoints.delivery_schedule
        ]

        entrypoints_state = []
        for ep in self.entrypoints:
            head_box = ep.material_queue[0] if ep.material_queue else None
            entrypoints_state.append({
                "ep": [int(ep.position[0]), int(ep.position[1])],
                "head": {
                    "type": head_box.type,
                    "id": int(head_box.id)
                } if head_box else None
            })

        self.csv_writer_steps.writerow([
            0,
            self.original_seed,
            json.dumps(boxes),
            json.dumps(agent_state),
            json.dumps(entrypoints_state),
            json.dumps(initial_orders),
            json.dumps([]),
            json.dumps([])
        ])

    def _log_step_events(self):
        """
        Log dynamic events for the current step:
        - newly generated entrypoint boxes
        - newly created delivery requests
        """
        self.current_step += 1

        entrypoint_events = []
        for ep in self.entrypoints:
            if ep.material_queue:
                last_box = ep.material_queue[-1]
                last_box_id = int(last_box.id)

                if not hasattr(ep, "last_logged_id") or ep.last_logged_id != last_box_id:
                    entrypoint_events.append({
                        "ep": [int(ep.position[0]), int(ep.position[1])],
                        "type": last_box.type,
                        "id": last_box_id
                    })
                    ep.last_logged_id = last_box_id

        outpoints_orders_created = []

        for d in self.outpoints.new_deliveries_this_step:
            outpoints_orders_created.append({
                "op": [int(d.op_position[0]), int(d.op_position[1])],
                "type": d.type,
                "num_boxes": int(d.num_boxes)
            })

        self.outpoints.new_deliveries_this_step = []

        self.csv_writer_steps.writerow([
            self.current_step,
            self.original_seed,
            "", "", "", "",
            json.dumps(entrypoint_events),
            json.dumps(outpoints_orders_created)
        ])

    @staticmethod
    def encode(num: int) -> str:  # PARA EL RENDER
        """
        Encode a numeric material identifier into its letter representation.
        """
        return " " if num == 0 else [key for key, value in TYPE_CODIFICATION.items() if num == value][0]

    @staticmethod
    def decode(letter: str) -> int:  # PARA EL RENDER
        """
        Decode a material letter into its numeric representation.
        """
        return TYPE_CODIFICATION[letter]

    def render_state(self, dark=True):
        """
        Render the current state as an image using matplotlib.
        """
        from matplotlib import pyplot as plt

        state = (
            np.flip(np.rot90(np.transpose(self.get_state().reshape((self.feature_number,) + self.grid.shape)), k=3), axis=1)
            / 255.0
        )

        if not dark:
            state = abs(state - 1)

        plt.clf()
        plt.imshow(state)
        plt.draw()
        plt.pause(10e-10)

    def render(self):
        """
        Render the warehouse as a text-based colored grid.

        Color guide:
        - Green: entrypoint
        - Cyan: empty-handed agent
        - Blue: agent carrying a box
        - Magenta/White: outpoint
        - Red: restricted cells
        - Black: outside cells
        """
        maze = "" + "+"

        for _ in range(self.grid.shape[1] * 2 - 1):
            maze += "-"
        maze += "+\n"
        for r, row in enumerate(self.grid):
            maze += "|"
            for e, element in enumerate(row):
                if (r, e) in getattr(self, "outside_cells", []):
                    encoded_el = f"{Back.BLACK}{Fore.WHITE}X{Style.RESET_ALL}"
                elif (r, e) in getattr(self, "outern_crown", []):
                    encoded_el = f"{Back.RED}{Fore.RED}█{Style.RESET_ALL}"
                elif element == 0:
                    encoded_el = " "
                elif (r, e) in [entrypoint.position for entrypoint in self.entrypoints]:
                    encoded_el = [ep.material_queue[0].type for ep in self.entrypoints if ep.position == (r, e)][0]
                else:
                    encoded_el = self.material[element].type

                try:
                    for agent in self.agents:
                        if agent.position == (r, e):
                            if agent.got_item:
                                encoded_el = f"{Back.BLUE}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                            else:
                                encoded_el = f"{Back.CYAN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in [entrypoint.position for entrypoint in self.entrypoints]:
                        encoded_el = f"{Back.GREEN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in self.outpoints.outpoints:
                        if self.get_ready_to_consume_types():
                            encoded_el = f"{Back.MAGENTA}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                        else:
                            encoded_el = f"{Back.WHITE}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                except Exception:
                    pass

                maze += encoded_el
                if e < self.grid.shape[1] - 1:
                    maze += ":"
            maze += "|\n"
        maze += "+"
        for _ in range(self.grid.shape[1] * 2 - 1):
            maze += "-"

        maze += "+\n"
        print(maze)

    def seed(self, seed: int = ...) -> list:
        """
        Seed the environment spaces and internal RNG.
        """
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.rng[0] = np.random.default_rng(seed)
        self.original_seed = seed
        return [seed]


if __name__ == "__main__":
    import numpy as np

    env = Storehouse(
        logging=True,
        random_start=True,
        save_episodes=False,
        max_steps=100,
        seed=5,
        record_scenario=False,   # Set to True to export scenario CSV logs
        scenario_path="scenario_seed_5",
    )

    _, _ = env.reset(options={"render": True})
    done = False
    t = 0

    while not done and t < 105:
        print(f"\n--- STEP {t} ---")
        env.render()

        mask = env.get_available_actions()
        print(f"Valid actions: {np.sum(mask)}")

        valid_positions = [pos for pos, m in zip(env.valid_positions, mask) if m]
        print("Valid positions:", valid_positions)

        valid_actions = [i for i, m in enumerate(mask) if m]
        if not valid_actions:
            print("No valid actions available. Terminating episode.")
            break

        action = np.random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action taken: {env.norm_action(action)} | Reward: {reward:.3f}")

        done = terminated or truncated
        t += 1

    env.close_scenario_logs()
