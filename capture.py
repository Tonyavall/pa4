import typing

from pacai.agents.greedy import GreedyFeatureAgent
from pacai.capture.gamestate import GameState
from pacai.core.action import Action, STOP
from pacai.core.agentinfo import AgentInfo
from pacai.core.board import Position
from pacai.core.features import FeatureDict
from pacai.search.distance import DistancePreComputer

MAX_DIST: float = 9999.0

class StudentCaptureAgent(GreedyFeatureAgent):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self._distances = DistancePreComputer()
        self._frontier: list[Position] = []
        self._initial_position: Position | None = None

    def game_start(self, initial_state: GameState) -> None:
        super().game_start(initial_state)

        self._distances = DistancePreComputer()
        self._distances.compute(initial_state.board)
        self._initial_position = initial_state.board.get_agent_initial_position(self.agent_index)
        self._frontier = self._build_frontier(initial_state)

    def _build_frontier(self, state: GameState) -> list[Position]:
        board = state.board
        frontier_col = (board.width // 2) - 1

        if (self._initial_position is not None) and (self._initial_position.col >= (board.width / 2)):
            frontier_col = board.width // 2

        frontier: list[Position] = []

        for row in range(board.height):
            position = Position(row, frontier_col)

            if (board.is_wall(position)):
                continue

            frontier.append(position)

        if (not frontier) and (self._initial_position is not None):
            frontier.append(self._initial_position)

        return frontier

    def _distance(
        self,
        start: Position | None, end: Position | None,
        default: float = MAX_DIST
    ) -> float:
        if (start is None) or (end is None):
            return default

        return float(self._distances.get_distance_default(start, end, default))

    def _closest_distance(
            self,
            start: Position | None,
            targets: typing.Iterable[Position],
            default: float = MAX_DIST
    ) -> float:
        if (start is None):
            return default

        best = default
        for target in targets:
            best = min(best, self._distance(start, target, default))

        return best

    def _distance_to_frontier(self, position: Position | None) -> float:
        if (position is None) or (len(self._frontier) == 0):
            return 0.0

        return min(self._distance(position, entry) for entry in self._frontier)

class AttackerAgent(StudentCaptureAgent):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self.weights.update({
            'bias': 1.0,
            'score': 150.0,
            'distance_to_food': -2.5,
            'distance_to_ghost': 1.8,
            'ghost_near': -200.0,
            'stopped': -80.0,
            'reverse': -4.0,
            'on_home_side': -10.0,
        })

    def compute_features(self, state: GameState, action: Action) -> FeatureDict:
        state = typing.cast(GameState, state)
        features: FeatureDict = FeatureDict()

        features['bias'] = 1.0
        features['score'] = float(state.get_normalized_score(self.agent_index))
        features['stopped'] = float(action == STOP)
        features['on_home_side'] = float(state.is_ghost(agent_index = self.agent_index))

        features['reverse'] = 0.0
        agent_actions = state.get_agent_actions(self.agent_index)

        if (len(agent_actions) > 1):
            features['reverse'] = float(action == state.get_reverse_action(agent_actions[-2]))

        position = state.get_agent_position(self.agent_index)

        if (position is None):
            return features

        food = state.get_food(agent_index = self.agent_index)

        if (len(food) > 0):
            food_distance = self._closest_distance(position, food, MAX_DIST)
            features['distance_to_food'] = float(min(food_distance, 20.0))
        else:
            features['distance_to_food'] = 0.0

        ghosts = state.get_nonscared_opponent_positions(agent_index = self.agent_index)
        ghost_distance = self._closest_distance(position, ghosts.values(), MAX_DIST)
        capped_ghost_distance = float(min(ghost_distance, 12.0))
        features['distance_to_ghost'] = capped_ghost_distance
        features['ghost_near'] = float(ghost_distance <= 3.0)

        return features

class GuardAgent(StudentCaptureAgent):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.weights.update({
            'bias': 1.0,
            'on_home_side': 60.0,
            'stopped': -60.0,
            'reverse': -3.0,
            'num_invaders': -1200.0,
            'distance_to_invader': -25.0,
            'frontier_distance': -2.5,
        })

    def compute_features(self, state: GameState, action: Action) -> FeatureDict:
        state = typing.cast(GameState, state)
        features: FeatureDict = FeatureDict()

        features['bias'] = 1.0
        features['stopped'] = float(action == STOP)
        features['on_home_side'] = float(state.is_ghost(agent_index = self.agent_index))

        features['reverse'] = 0.0
        agent_actions = state.get_agent_actions(self.agent_index)

        if (len(agent_actions) > 1):
            features['reverse'] = float(action == state.get_reverse_action(agent_actions[-2]))

        position = state.get_agent_position(self.agent_index)

        if (position is None):
            return features

        invaders = state.get_invader_positions(agent_index = self.agent_index)
        features['num_invaders'] = float(len(invaders))

        if (len(invaders) > 0):
            distance = self._closest_distance(position, invaders.values(), MAX_DIST)
            features['distance_to_invader'] = float(distance)
            features['frontier_distance'] = 0.0
        else:
            features['distance_to_invader'] = 0.0
            features['frontier_distance'] = float(self._distance_to_frontier(position))

        return features

def create_team() -> list[AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    module_name = __name__
    offensive = AgentInfo(name = f"{module_name}.AttackerAgent")
    defensive = AgentInfo(name = f"{module_name}.GuardAgent")

    return [offensive, defensive]
