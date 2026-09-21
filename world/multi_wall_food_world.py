from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .food_world import WorldObstacle
from .geometry import distance
from .room_food_world import RoomFoodWorld, RoomFoodWorldConfig


@dataclass(frozen=True)
class MultiWallFoodWorldConfig(RoomFoodWorldConfig):
    """Two divider walls whose doorways form a straight or zigzag route."""

    size: float = 12.0
    max_steps: int = 160
    door_width: float = 2.0
    wall_spacing: float = 3.2
    door_offset: float = 2.4
    door_jitter: float = 0.45
    use_position_sensors: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.wall_spacing <= self.wall_thickness:
            raise ValueError("wall_spacing must exceed wall_thickness.")
        if self.wall_spacing >= self.size * 0.6:
            raise ValueError("wall_spacing must leave two usable outer rooms.")
        if self.door_offset <= 0:
            raise ValueError("door_offset must be positive.")
        if self.door_jitter < 0:
            raise ValueError("door_jitter cannot be negative.")
        door_extent = self.door_offset + self.door_jitter + self.door_width / 2
        if door_extent >= self.size / 2:
            raise ValueError("Door positions must stay inside the world boundary.")


@dataclass(frozen=True)
class MultiWallScenarioSignature:
    """Categorical factors balanced across the three experiment splits."""

    orientation: str
    start_side: str
    first_door_side: str
    door_pattern: str


@dataclass(frozen=True)
class _MultiWallLayout:
    orientation: str
    start_sign: int
    wall_positions: tuple[float, float]
    door_centers: tuple[float, float]
    door_pattern: str
    obstacles: tuple[WorldObstacle, WorldObstacle, WorldObstacle, WorldObstacle]


class MultiWallFoodWorld(RoomFoodWorld):
    """Navigation through two unseen doorways without absolute coordinates."""

    def __init__(self, config: MultiWallFoodWorldConfig | None = None) -> None:
        super().__init__(config or MultiWallFoodWorldConfig())
        self.config: MultiWallFoodWorldConfig

    def scenario_signature_for_seed(self, seed: int) -> MultiWallScenarioSignature:
        layout = self._make_layout(random.Random(seed))
        first_center = self._door_centers_in_travel_order(layout)[0]
        return MultiWallScenarioSignature(
            orientation=layout.orientation,
            start_side="positive" if layout.start_sign > 0 else "negative",
            first_door_side="positive" if first_center > 0 else "negative",
            door_pattern=layout.door_pattern,
        )

    def _make_layout(self, rng: random.Random) -> _MultiWallLayout:
        orientation = self._draw_orientation(rng)
        start_sign = -1 if rng.random() < 0.5 else 1
        first_door_sign = -1 if rng.random() < 0.5 else 1
        door_pattern = "aligned" if rng.random() < 0.5 else "alternating"
        second_door_sign = (
            first_door_sign if door_pattern == "aligned" else -first_door_sign
        )

        midpoint = rng.uniform(-self.config.size * 0.04, self.config.size * 0.04)
        half_spacing = self.config.wall_spacing / 2
        wall_positions = (midpoint - half_spacing, midpoint + half_spacing)
        travel_positions = self._ordered_positions(wall_positions, start_sign)
        travel_centers = (
            first_door_sign * self._door_magnitude(rng),
            second_door_sign * self._door_magnitude(rng),
        )
        center_by_position = dict(zip(travel_positions, travel_centers))
        door_centers = tuple(center_by_position[position] for position in wall_positions)

        obstacles: list[WorldObstacle] = []
        for wall_position, door_center in zip(wall_positions, door_centers):
            obstacles.extend(self._wall_obstacles(wall_position, door_center, orientation))

        return _MultiWallLayout(
            orientation=orientation,
            start_sign=start_sign,
            wall_positions=wall_positions,
            door_centers=door_centers,
            door_pattern=door_pattern,
            obstacles=(obstacles[0], obstacles[1], obstacles[2], obstacles[3]),
        )

    def _door_magnitude(self, rng: random.Random) -> float:
        return self.config.door_offset + rng.uniform(
            -self.config.door_jitter,
            self.config.door_jitter,
        )

    def _wall_obstacles(
        self,
        wall_position: float,
        door_center: float,
        orientation: str,
    ) -> tuple[WorldObstacle, WorldObstacle]:
        half = self.config.size / 2
        wall_half = self.config.wall_thickness / 2
        door_low = door_center - self.config.door_width / 2
        door_high = door_center + self.config.door_width / 2
        if orientation == "vertical":
            return (
                WorldObstacle(
                    wall_position - wall_half,
                    -half,
                    wall_position + wall_half,
                    door_low,
                ),
                WorldObstacle(
                    wall_position - wall_half,
                    door_high,
                    wall_position + wall_half,
                    half,
                ),
            )
        return (
            WorldObstacle(
                -half,
                wall_position - wall_half,
                door_low,
                wall_position + wall_half,
            ),
            WorldObstacle(
                door_high,
                wall_position - wall_half,
                half,
                wall_position + wall_half,
            ),
        )

    def _place_agent_and_food(
        self,
        rng: random.Random,
        layout: _MultiWallLayout,
    ) -> tuple[float, float, float, float]:
        first_door, second_door = self._door_centers_in_travel_order(layout)
        axis_low = self.config.size * 0.36
        axis_high = self.config.size * 0.42
        agent_axis = layout.start_sign * rng.uniform(axis_low, axis_high)
        food_axis = -layout.start_sign * rng.uniform(axis_low, axis_high)
        lane_jitter = self.config.size * 0.035
        agent_lane = -first_door + rng.uniform(-lane_jitter, lane_jitter)
        food_lane = -second_door + rng.uniform(-lane_jitter, lane_jitter)
        if layout.orientation == "vertical":
            return agent_axis, agent_lane, food_axis, food_lane
        return agent_lane, agent_axis, food_lane, food_axis

    @staticmethod
    def _route_distance(
        agent_x: float,
        agent_y: float,
        food_x: float,
        food_y: float,
        layout: _MultiWallLayout,
    ) -> float:
        if layout.orientation == "vertical":
            agent_axis, food_axis = agent_x, food_x
        else:
            agent_axis, food_axis = agent_y, food_y

        direction = 1 if food_axis > agent_axis else -1
        doorway_pairs = [
            (position, center)
            for position, center in zip(layout.wall_positions, layout.door_centers)
            if min(agent_axis, food_axis) <= position <= max(agent_axis, food_axis)
        ]
        doorway_pairs.sort(key=lambda pair: pair[0], reverse=direction < 0)

        points = [(agent_x, agent_y)]
        for position, center in doorway_pairs:
            points.append(
                (position, center)
                if layout.orientation == "vertical"
                else (center, position)
            )
        points.append((food_x, food_y))
        return sum(
            distance(left_x, left_y, right_x, right_y)
            for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
        )

    @staticmethod
    def _ordered_positions(
        wall_positions: tuple[float, float],
        start_sign: int,
    ) -> tuple[float, float]:
        return tuple(sorted(wall_positions, reverse=start_sign > 0))

    def _door_centers_in_travel_order(
        self,
        layout: _MultiWallLayout,
    ) -> tuple[float, float]:
        center_by_position = dict(zip(layout.wall_positions, layout.door_centers))
        return tuple(
            center_by_position[position]
            for position in self._ordered_positions(
                layout.wall_positions,
                layout.start_sign,
            )
        )
