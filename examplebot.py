"""
examplebot.py
=============
Reinforcement learning training script for a Rocket League bot using RLGym v2
and RLGym-PPO. Defines the renderer, state mutators, reward functions, and
the main training entry point.
 
Training is structured in progressive stages (1 -> 1.25 -> 1.5 -> 2 -> 2.25 -> 2.5 -> 3),
gradually shifting the reward signal from basic navigation to goal-scoring
and game-sense behaviours as the agent accumulates experience.
"""

import os
import json
import socket
import torch
import numpy as np
import warnings
import functools
from typing import List, Dict, Any

from ClassDeprecation import deprecated_class

from rlgym.api import RewardFunction, AgentID, Renderer, StateMutator, RLGym
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_MAX_SPEED, CAR_MAX_SPEED, CAR_MAX_ANG_VEL, BACK_NET_Y, BALL_RADIUS
from rlgym.rocket_league.math import rand_vec3, rand_uvec3, normalize
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator, FixedTeamSizeMutator
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import RAMP_HEIGHT
from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator

# Project name, used for organising checkpoint directories.
# Changing this will start a new run.
project_name = "ExampleBot" 

# Prefer GPU acceleration if CUDA is available; fall back to CPU otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu" 


#=========================================
# @section: Renderer
#=========================================

# Default UDP target for RocketSimVis — localhost on a fixed port.
DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 9273

# Ordered action axis names; used to label control values in JSON payloads.
BUTTON_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")

class RocketSimVisRenderer(Renderer[GameState]):
    """
    Streams game state to a RocketSimVis visualiser over UDP as JSON.
 
    Each rendered frame serialises ball physics, car states, and boost pad
    availability, then fires-and-forgets the packet to the configured endpoint.
    Rendering is intentionally disabled during training (see `render=False` in
    the Learner config) to avoid the SPS hit.
    """

    def __init__(self, udp_ip=DEFAULT_UDP_IP, udp_port=DEFAULT_UDP_PORT):
        # UDP socket -- connectionless, so we never need to worry about teardown.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_ip = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def write_physobj(physobj):
        """
        Serialise a physics object (ball or car) to a JSON-compatible dict.
 
        Returns position, forward/up orientation vectors, linear velocity,
        and angular velocity - everything the visualiser needs to reconstruct
        the object's pose in 3-D space.
        """
        return {
            'pos': physobj.position.tolist(),
            'forward': physobj.forward.tolist(),
            'up': physobj.up.tolist(),
            'vel': physobj.linear_velocity.tolist(),
            'ang_vel': physobj.angular_velocity.tolist()
        }

    @staticmethod
    def write_car(car: Car, controls=None):
        """
        Serialise a car's state to a JSON-compatible dict.
 
        Includes team affiliation, physics, boost level, ground/air/demo status,
        and flip availability. If control inputs are provided (e.g. from
        `shared_info`), they are appended so the visualiser can overlay inputs.
 
        Parameters
        ----------
        car:
            The Car object to serialise.
        controls:
            Optional raw control array or dict. Arrays are zipped against
            BUTTON_NAMES before inclusion.
        """
        j = {
            'team_num': int(car.team_num),
            'phys': RocketSimVisRenderer.write_physobj(car.physics),
            'boost_amount': car.boost_amount,
            'on_ground': bool(car.on_ground),
            "has_flipped_or_double_jumped": bool(car.has_flipped or car.has_double_jumped),
            'is_demoed': bool(car.is_demoed),
            'has_flip': bool(car.can_flip)
        }
        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = {k: float(v) for k, v in zip(BUTTON_NAMES, controls)}
            j['controls'] = controls
        return j

    def render(self, state: GameState, shared_info: Dict[str, Any]) -> Any:
        """
        Encode and transmit the current game state as a UDP packet.
 
        Pulls per-agent control inputs from `shared_info` so the visualiser
        can display what actions the policy is actually taking.
        """
        controls = shared_info.get("controls", {})
        j = {
            'ball_phys': self.write_physobj(state.ball),
            'cars': [self.write_car(car, controls.get(agent_id)) for agent_id, car in state.cars.items()],
            # Pad is active when its respawn timer has reached zero.
            'boost_pad_states': (state.boost_pad_timers <= 0).tolist()
        }
        self.sock.sendto(json.dumps(j).encode('utf-8'), (self.udp_ip, self.udp_port))

    def close(self):
        # Nothing to clean up as the OS will reclaim the socket on exit
        pass

# @endsection

#=========================================
# @section: State Setters
#=========================================

class RandomPhysicsMutator(StateMutator[GameState]):  
    """
    Randomises the position, velocity, and orientation of the ball and all cars
    at episode reset.
 
    Spawn positions are constrained to avoid: goal mouths, arena corners,
    and any zone where a wall meets the ceiling/floor (ramp areas). Cars are
    additionally capped to the lower sixth of the arena vertically so they
    don't spawn mid-air at absurd heights. Yes, we tried letting them spawn
    anywhere. No, it wasn't fun to watch.
    """

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 100 # Minimum distance from any arena boundary, in UU.
        goal_line_y = 5120
        min_goal_dist = 2000 # Keep the ball away from the goal mouth on spawn.

        # Index 0 is the ball; subsequent indices are cars.
        for i, po in enumerate([state.ball] + [car.physics for car in state.cars.values()]):
            while True:
                # Cars are restricted to the lower sixth of the arena to avoid
                # spawning at ceiling height, which is... less than ideal.
                max_z = (CEILING_Z - padding) if i == 0 else ((CEILING_Z / 6) - padding)
                new_pos = np.random.uniform(
                    [-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, 0 + padding],
                    [SIDE_WALL_X - padding, BACK_WALL_Y - padding, max_z]
                )

                # Reject ball positions too close to either goal.
                if i == 0 and (abs(new_pos[1]) > goal_line_y - min_goal_dist):
                    continue
                # Reject positions that land inside the rounded corner geometry.
                if abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - padding:
                    continue

                # Reject positions in the ramp transition zone (wall meets floor/ceiling).
                close_to_wall = (abs(new_pos[0]) >= SIDE_WALL_X - RAMP_HEIGHT or
                                 abs(new_pos[1]) >= BACK_WALL_Y - RAMP_HEIGHT or
                                 abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - RAMP_HEIGHT)
                close_to_floor_or_ceiling = (new_pos[2] <= RAMP_HEIGHT or new_pos[2] >= CEILING_Z - RAMP_HEIGHT)

                if close_to_wall and close_to_floor_or_ceiling:
                    continue
                break

            po.position = new_pos
            # Random velocity up to ~2300 UU/s (roughly car max speed) for the ball.
            po.linear_velocity = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)

            if i > 0:
                # Construct a valid orthonormal rotation matrix for the car.
                fw = rand_uvec3()
                up = rand_uvec3()
                right = normalize(np.cross(up, fw))
                up = normalize(np.cross(fw, right))
                po.rotation_mtx = np.stack([fw, right, up])
    
class RandomStateMutator(StateMutator[GameState]):
    """
    Blends standard kickoff starts with fully randomised physics states.
 
    The 60/40 split between kickoffs and random physics keeps the agent
    practising realistic game openings while also exposing it to a wide
    variety of mid-game scenarios it might not naturally encounter.
    """

    def __init__(self):
        self.mutator = WeightedSampleMutator.from_zipped(
            (KickoffMutator(), 0.6),        # Standard kickoff - keep it grounded.
            (RandomPhysicsMutator(), 0.4),  # Chaos mode - anything goes.
            (AerialMutator(), 0.2)          # 20% of resets are high aerials.
        )

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.mutator.apply(state, shared_info)

# Yeah hate me for it but I had a bug so AI wrote this.
# Sue me.
class AerialMutator(StateMutator[GameState]):
    # --- Tunable Constants ---
    BALL_X_RANGE = (-1000, 1000)
    BALL_Y_RANGE = (-1500, 1500)
    BALL_Z_RANGE = (1000, 1400)      # Lowered max Z to prevent instant ceiling hits
    BALL_VZ_RANGE = (400, 800)       # Adjusted to keep ball in play longer
    CAR_Y_OFFSET_RANGE = (1500, 2500)
    CAR_INITIAL_SPEED = 800.0
    CAR_REST_Z = 17.0
    EPSILON = 1e-8

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        assert state.cars, "AerialMutator requires at least one car"

        # 1. Spawn Ball
        ball_x = np.random.uniform(*self.BALL_X_RANGE)
        ball_y = np.random.uniform(*self.BALL_Y_RANGE)
        ball_z = np.random.uniform(*self.BALL_Z_RANGE)

        state.ball.position = np.array([ball_x, ball_y, ball_z])
        state.ball.linear_velocity = np.array([
            np.random.uniform(-200, 200),
            np.random.uniform(-200, 200),
            np.random.uniform(*self.BALL_VZ_RANGE)
        ])
        state.ball.angular_velocity = np.zeros(3)

        # 2. Spawn Cars
        for agent, car in state.cars.items():
            side = -1 if not car.is_orange else 1
            
            # Position relative to ball
            car_x = ball_x + np.random.uniform(-500, 500)
            car_y = ball_y + (np.random.uniform(*self.CAR_Y_OFFSET_RANGE) * side)
            
            car.physics.position = np.array([car_x, car_y, self.CAR_REST_Z])
            
            # Rotation and Velocity
            target_pos = state.ball.position
            car_pos = car.physics.position
            
            if np.linalg.norm(target_pos - car_pos) > self.EPSILON:
                # Use the robust look_at logic
                car.physics.rotation_mtx = self.safe_look_at(car_pos, target_pos)
                
                # Momentum toward ball
                forward_vec = car.physics.rotation_mtx[:, 0] # Extract forward from column
                car.physics.linear_velocity = forward_vec * self.CAR_INITIAL_SPEED
            else:
                # Fallback to identity matrix if car/ball are overlapping
                car.physics.rotation_mtx = np.eye(3)
                car.physics.linear_velocity = np.zeros(3)

            car.physics.angular_velocity = np.zeros(3)
            car.boost_amount = 1.0

    def safe_normalize(self, v):
        norm = np.linalg.norm(v)
        return v / (norm + self.EPSILON)

    def safe_look_at(self, source, target):
        forward = self.safe_normalize(target - source)
        world_up = np.array([0.0, 0.0, 1.0])

        # Handle degenerate case where car is directly under ball
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])

        right = self.safe_normalize(np.cross(world_up, forward))
        up = self.safe_normalize(np.cross(forward, right))

        # RLGym 2 / RocketSim: Column 0=F, 1=R, 2=U
        return np.column_stack([forward, right, up])

# @endsection

#=========================================
# @section: Rewards
#=========================================

@deprecated_class("AdvancedTouchReward is deprecated. Use StableTouchReward instead.")
class AdvancedTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    [DEPRECATED] Use StableTouchReward instead.

    Rewards the agent for making contact with the ball, with an optional bonus
    proportional to the acceleration imparted on impact.
 
    Parameters
    ----------
    touch_reward:
        Flat reward per touch (or per-touch-count if `use_touch_count` is True).
    acceleration_reward:
        Scalar applied to the normalised change in ball speed caused by the touch.
    use_touch_count:
        If True, scales the flat touch reward by the raw touch count rather than
        clamping to a binary "did touch / did not touch".
    """

    def __init__(self, touch_reward: float = 0.0, acceleration_reward: float = 1.0, use_touch_count: bool = False):
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.use_touch_count = use_touch_count
        self.prev_ball_vel = np.zeros(3)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # FIX: Store copy of velocity, not object reference - otherwise
        #      prev_ball_vel would silently track the live state.
        self.prev_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        ball_vel = state.ball.linear_velocity
        
        for agent in agents:
            touches = state.cars[agent].ball_touches
            if touches > 0:
                if not self.use_touch_count:
                    touches = 1 # Binary: reward once per step, regardless of touch count
                # Normalise delta-v so the reward is bounded in [0, 1] before scaling
                acceleration = np.linalg.norm(ball_vel - self.prev_ball_vel) / BALL_MAX_SPEED
                rewards[agent] += self.touch_reward * touches
                rewards[agent] += acceleration * self.acceleration_reward

        # Must copy here too -- same reference-vs-copy pitfall as in reset()
        self.prev_ball_vel = ball_vel.copy()
        return rewards


class StableTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A simple, bulletproof touch reward (I hope).
    Grants a flat reward if the agent touches the ball during the step.
    Removes delta-v calculations to prevent the "Wall Pinch" reward exploit.
    """
    def __init__(self, touch_reward: float = 1.0):
        self.touch_reward = touch_reward

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        return {agent: self.touch_reward if state.cars[agent].ball_touches > 0 else 0.0 for agent in agents}


class FaceBallReward(RewardFunction):
    """
    Rewards the agent for orienting its car toward the ball.
 
    Computed as the dot product of the car's forward vector and the unit vector
    pointing from the car to the ball. Returns values in [-1, 1]; a score of 1
    means the car is pointing directly at the ball.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            direction_to_ball = state.ball.position - car.physics.position
            norm = np.linalg.norm(direction_to_ball)

            # FIX: Division by zero protection. If the car is somehow inside the ball,
            #      default to facing along the X axis rather than exploding.
            if norm > 1e-5:
                direction_to_ball /= norm
            else:
                direction_to_ball = np.array([1.0, 0.0, 0.0])

            dot_product = np.dot(car.physics.forward, direction_to_ball)
            rewards[agent] = float(dot_product)
        return rewards
                        

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent proportionally to how fast it is moving toward the ball.
 
    Uses perspective-corrected (inverted) physics for orange-team cars so that
    the direction calculations remain consistent regardless of field orientation.
    Reward is clamped to [0, 1] - driving away from the ball yields no penalty,
    just no reward.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Use inverted physics for orange so the coordinate frame is consistent
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            
            pos_diff = (ball_physics.position - car_physics.position)
            # FIX: Division by zero protection - avoid NaN if car spawns on top of ball
            dist_to_ball = max(1e-5, np.linalg.norm(pos_diff))
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(car_physics.linear_velocity, dir_to_ball)
            # Clamp to zero so reversing away doesn't produce a negative reward
            rewards[agent] = max(speed_toward_ball / CAR_MAX_SPEED, 0.0)
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """
    Simple binary reward that fires whenever the car is airborne.
 
    Used in early training stages to encourage the agent to leave the ground
    and explore aerial movement, which is critical for advanced play.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for the ball moving toward the opponent's goal.
 
    The target goal is selected based on team side, and the reward is normalised
    by BALL_MAX_SPEED so it stays in [0, 1]. Negative velocities (ball moving
    away from goal) are clamped to zero.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            # Orange attacks the blue goal (negative Y); blue attacks the orange goal (positive Y).
            goal_y = -BACK_NET_Y if state.cars[agent].is_orange else BACK_NET_Y
            pos_diff = np.array([0, goal_y, 0]) - state.ball.position
            # FIX: Division by zero protection
            dist = max(1e-5, np.linalg.norm(pos_diff))
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(state.ball.linear_velocity, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / BALL_MAX_SPEED, 0.0)
        return rewards


# Save Boost Reward (Mid/Late Stage)
class SaveBoostReward(RewardFunction[AgentID, GameState, float]):
    """
    Gently rewards the agent for maintaining a healthy boost reserve.
 
    Uses a square-root scale so the marginal reward for hoarding more boost
    decreases as the tank fills — we want the bot to conserve boost, not
    become a cowardly boost-hoarder who never commits. You know the type.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None: 
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(np.sqrt(state.cars[agent].boost_amount)) for agent in agents}


# Air Touch Reward (Late Stage)
class AirTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for making aerial contact with the ball above a minimum height.
 
    Reward scales linearly with height (0 at `min_height`, 1 at ceiling) to
    incentivise high, committed aerials over cheap low-air touches.
 
    Parameters
    ----------
    min_height:
        Minimum ball height (Z) required for the touch to count. Defaults to
        two ball radii above the ground which is just enough to rule out ground bounces.
    """

    def __init__(self, min_height=BALL_RADIUS * 2):
        self.min_height = min_height

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None: 
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0 and not car.on_ground and car.physics.position[2] > self.min_height:
                # Reward scales 0 to 1 based on height; higher aerials = bigger reward.
                rewards[agent] = float(car.physics.position[2] / CEILING_Z)
        return rewards


# Event Reward for Conceding (Mid/Late Stage)
class EventReward(RewardFunction[AgentID, GameState, float]):
    """
    Applies a one-time penalty when the agent's team concedes a goal.
 
    Intentionally omits a scoring bonus - that is handled by GoalReward - so
    the two concerns remain independently tunable. The concede penalty weight
    is deliberately reduced in later training stages to promote aggression over
    risk-averse play.
 
    Parameters
    ----------
    concede:
        Reward delta applied when a goal is conceded. Should be negative.
    """

    def __init__(self, concede=-1.0):
        self.concede = concede

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        
        # Check if the ball crossed the goal line on either side
        blue_conceded = state.ball.position[1] < -BACK_WALL_Y
        orange_conceded = state.ball.position[1] > BACK_WALL_Y

        for agent in agents:
            car = state.cars[agent]
            if car.is_orange and orange_conceded: 
                rewards[agent] += self.concede
            elif not car.is_orange and blue_conceded: 
                rewards[agent] += self.concede
                
        return rewards


class RecoveryReward(RewardFunction[AgentID, GameState, float]):
    """
    Grants a continuous reward based on how 'flat' the car lands.
    1.0 for perfect, 0.0 for landing on the side/roof.
    """
    def __init__(self):
        self.was_in_air = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.was_in_air = {agent: not initial_state.cars[agent].on_ground for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            currently_in_air = not car.on_ground
            
            if self.was_in_air.get(agent, False) and not currently_in_air:
                # max(0, up[2]) means a perfect landing is 1.0, 90-degrees is 0.0
                rewards[agent] = max(0.0, float(car.physics.up[2]))
            else:
                rewards[agent] = 0.0
                
            self.was_in_air[agent] = currently_in_air
            
        return rewards


class OutOfPositionPenalty(RewardFunction[AgentID, GameState, float]):
    """
    A smooth gradient penalty. Barely penalizes close follow-ups, 
    but heavily penalizes camping on the wrong side of the ball.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball_y = state.ball.position[1]
            car_y = car.physics.position[1]
            
            # Calculate how far "ahead" of the ball the car is
            if car.is_orange:
                distance_ahead = max(0.0, ball_y - car_y)
            else:
                distance_ahead = max(0.0, car_y - ball_y)
                
            # Scale the penalty: 1000 units ahead = -0.01 penalty. 
            # It's a gentle slope, not a brick wall.
            rewards[agent] = -0.01 * (distance_ahead / 1000.0)
                
        return rewards


class BoostEfficiencyReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards holding boost, but penalizes using boost
    when the car is already at or near max speed (supersonic).
    """
    def __init__(self):
        self.last_boost = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_boost = {a: initial_state.cars[a].boost_amount for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            current_boost = car.boost_amount
            speed = np.linalg.norm(car.physics.linear_velocity)

            # Calculate passive reward for having boost (sqrt scale)
            boost_spent = max(0.0, self.last_boost.get(agent, 0.0) - current_boost)

            # Gentle passive reward for having boost (sqrt scale)
            reward = float(np.sqrt(current_boost)) * 0.02

            # Supersonic is ~2200. If they are boosting while > 2100, they are wasting it.
            if boost_spent > 0 and speed > 2100:
                reward -= boost_spent * 2.0 # penalty for waste

        return rewards

# @endsection

#=========================================
# Training Script
#=========================================

def build_rlgym_v2_env():
    """
    Construct and return a fully configured RLGym v2 environment wrapped for
    compatibility with the RLGym-PPO Learner.
 
    Environment configuration
    -------------------------
    - 1v1 match with opponent spawning enabled.
    - Actions repeated for 8 physics ticks (~0.1s at 120hz) to reduce the
      effective action frequency and make exploration more tractable.
    - Episode ends on goal, or truncates after 20s of no ball contact / 5 min
      of total game time.
    - Observations are normalised using DefaultObs with field-scale coefficients.
 
    Reward staging
    --------------
    Reward functions are staged across training to shift the agent's focus from
    basic navigation to goal-scoring and game-sense. Only one stage is active
    at a time - the others are commented out. A bridge was introduced after Stage 1 -> 2 transitions
    caused catastrophic reward collapse. Whoopsies!
    """

    spawn_opponents = True
    team_size = 1
    action_repeat = 8               # Physics ticks per action step.
    no_touch_timeout_seconds = 20   # Reset if nobody touches the ball for this long
    game_timeout_seconds = 300      # Hard episode length cap (5 minutes)

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )
    
    # ---------------------------------------------------------
    # STAGE 1: < 100_000_000 Steps
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.15), 
    #     (SpeedTowardBallReward(), 5.0), 
    #     (FaceBallReward(), 1.0), 
    #     (AdvancedTouchReward(touch_reward=1.0, acceleration_reward=0.0, use_touch_count=True), 50.0), 
    # )

    # NOTE: After training the bot for a few hours, I found out that 
    #       switching from Stage 1 to Stage 2 after ~100M timesteps
    #       completely collapses the model. So we're implementing
    #       "bridge" stages to (hopefully) avoid this.
    # UPDATE: it didn't work

    # ---------------------------------------------------------
    # STAGE 1.25: Ugh.
    # NOTE: Make sure that the `policy_lr` and `critic_lr` are
    #       both set to 1.5e-4 and NOT 1e-4 or anything else :3
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.1),
    #     (SpeedTowardBallReward(), 5.0), 
    #     (FaceBallReward(), 1.2),        
    #     (VelocityBallToGoalReward(), 8.0), 
    #     (StableTouchReward(touch_reward=15.0), 1.0), # The new, exploit-free (hopefully) touch reward
    #     (GoalReward(), 100.0), 
    #     (EventReward(concede=-1.0), 5.0)
    # )

    # ---------------------------------------------------------
    # STAGE 1.5: Whoopsies!
    # NOTE: Make sure that the `policy_lr` and `critic_lr` are
    #       both set to 2e-4 and NOT 1e-4 :3
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.1),
    #     (SpeedTowardBallReward(), 2.0),      # Fading: Was 5.0
    #     (FaceBallReward(), 0.6),             # Fading: Was 1.2
    #     (VelocityBallToGoalReward(), 20.0),  # BOOSTED: Directing the ball is priority #1
    #     (StableTouchReward(touch_reward=8.0), 1.0), # Lowered: Touches are common now
    #     (GoalReward(), 200.0),               # DOUBLED: Scoring is the ultimate win
    #     (EventReward(concede=-10.0), 5.0)    # INCREASED: Stronger punishment for conceding
    # )
    
    # ---------------------------------------------------------
    # STAGE 1.75: I'm getting tired of this.
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.1),
    #     (SpeedTowardBallReward(), 0.5),      # SLASHED: Force Blue to find a new path
    #     (FaceBallReward(), 0.3),             # SLASHED: No more "staring" points
    #     (VelocityBallToGoalReward(), 15.0),  # SLIGHTLY LOWERED: Still good, but not a salary
    #     (StableTouchReward(touch_reward=2.0), 1.0), # CRIPPLED: Touches are now just breadcrumbs
    #     (GoalReward(), 1500.0),              # MASSIVE BOOST: One goal > 2 minutes of circling
    #     (EventReward(concede=-20.0), 10.0)   # INCREASED
    # )

    # ---------------------------------------------------------
    # STAGE 1.85: The Polish (Fixing positioning, landings, and boost)
    #             < 500_000_000
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.02),                # Lowered: Stop jumping just to land!
    #     (RecoveryReward(), 4.0),              # Lowered & Smoothed: Nice landings are a bonus, not a career
    #     (OutOfPositionPenalty(), 1.0),        # Smoothed: Bot won't be scared to push anymore
    #     (SaveBoostReward(), 0.5),             # NEW: Teach them to hold onto the shiny yellow juice
    #     (SpeedTowardBallReward(), 0.5), 
    #     (VelocityBallToGoalReward(), 20.0), 
    #     (StableTouchReward(touch_reward=5.0), 1.0), 
    #     (GoalReward(), 1500.0), 
    #     (EventReward(concede=-20.0), 10.0)
    # )

    # ---------------------------------------------------------
    # STAGE 2.0: Efficiency & Aerial Era (500M - 800M)
    # ---------------------------------------------------------
    reward_fn = CombinedReward(
        (BoostEfficiencyReward(), 2.0),       # NEW: Penalize supersonic boost waste
        (AirTouchReward(min_height=300), 5.0),# NEW: Actually reward leaving the ground
        (VelocityBallToGoalReward(), 15.0),   # Main driver of game-sense
        (GoalReward(), 1500.0),               # The ultimate objective
        (EventReward(concede=-20.0), 10.0),   # The ultimate penalty
        
        # --- THE NERFED CRUTCHES ---
        (RecoveryReward(), 0.5),              # Was 4.0. You know how to land, stop farming it.
        (OutOfPositionPenalty(), 0.2),        # Was 1.0. Just a tiny reminder to rotate.
        (StableTouchReward(touch_reward=1.0), 0.5), # Slashed. Touches must have a purpose now.
        (SpeedTowardBallReward(), 0.1),       # Barely exists, just breaks midfield paralysis.
    )

    # ---------------------------------------------------------
    # STAGE 2.5
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.05),
    #     (SpeedTowardBallReward(), 1.5),      # Slowly fading the "beginner" crutch
    #     (FaceBallReward(), 0.5),             # Slowly fading the "beginner" crutch
    #     (VelocityBallToGoalReward(), 12.0),  # Keep the "Direction" priority high
    #     (AdvancedTouchReward(touch_reward=0.2, acceleration_reward=5.0), 20.0), 
    #     (GoalReward(), 75.0),
    #     (SaveBoostReward(), 1.0),            # NEW: Introduce the concept of boost
    #     (EventReward(concede=-5.0), 10.0)
    # )

    # ---------------------------------------------------------
    # STAGE 3: 1_000_000_000+ Steps
    # ---------------------------------------------------------
    # reward_fn = CombinedReward(
    #     (InAirReward(), 0.02),
    #     (SpeedTowardBallReward(), 1.0),
    #     (FaceBallReward(), 0.2),
    #     (VelocityBallToGoalReward(), 5.0),
    #     (AirTouchReward(), 8.0), 
    #     (SaveBoostReward(), 3.0),
    #     (AdvancedTouchReward(touch_reward=0.0, acceleration_reward=8.0), 15.0),  
    #     (GoalReward(), 15.0),
    #     (EventReward(concede=-1.0), 15.0),  # Decreased penalty to promote aggression
    # )

    # Observation builder: normalises all inputs to roughly [-1, 1] using
    # physical field limits as scaling denominators
    obs_builder = DefaultObs(
        zero_padding=3,
        pos_coef=np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / CAR_MAX_SPEED,
        ang_vel_coef=1 / CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    ) 

    # Ensure team sizes are fixed before applying the randomised state mutator
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size if spawn_opponents else 0),
        RandomStateMutator() 
    )  

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer() # Disabled at runtime via render=False in Learner
    )

    return RLGymV2GymWrapper(rlgym_env)

if __name__ == "__main__":
    from rlgym_ppo import Learner

    # Number of parallel environment workers. More workers = more experience
    # per second, at the cost of RAM and CPU. Tune to your hardware
    n_proc = 32
    # Minimum batch of agents required before inference is dispatched
    # Set to ~90% of n_proc to keep GPU utilisation high without stalling
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Find all numerically-named checkpoint subdirectories.
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.isdigit()]
    # FIX: Load highest numerically, not lexicographically - "9" > "10" as strings,
    #      which would load the wrong checkpoint after the 10th save
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(checkpoint_files, key=int)) if checkpoint_files else None

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,            # Swap in a WandB/custom logger here if desired
        ppo_batch_size=200_000,         # Total steps collected before each PPO update
        ts_per_iteration=200_000,       # FIX: Must match ppo_batch_size
        exp_buffer_size=300_000,        # Experience replay buffer capacity
        ppo_minibatch_size=50_000,      # Mini-batch size within each PPO epoch
        ppo_ent_coef=0.005,              # FIX: Entropy coefficient -- golden value per guide
        ppo_epochs=2,                   # FIX: Low epoch count per guide to prevent overfitting
        policy_layer_sizes=[512, 512, 512],  # Three-layer MLP for the policy network
        critic_layer_sizes=[512, 512, 512],  # Matching architecture for the value network
        policy_lr=5e-5,                 # FIX: 2e-4 Early, drop to 1e-4 Mid, 0.8e-4 Later
        critic_lr=5e-5,                 # Kept in sync with policy_lr for now
        render=True,                    # FIX: Rendering MUST be False during training to preserve SPS
        gae_gamma=0.995,                # High discount factor as we care about long-term reward
        n_checkpoints_to_keep=1000,     # Keep plenty of checkpoints for rollback if needed
        render_delay=0.047,             # Target ~21fps during visualisation runs
        add_unix_timestamp=False,       # Use sequential checkpoint names, not timestamps
        checkpoint_load_folder=checkpoint_load_folder,  # Resume from latest checkpoint if available.
        checkpoints_save_folder=checkpoint_folder,                  
        standardize_returns=True,       # Normalise advantages, I strongly recommend it
        standardize_obs=False,          # Observations are already manually normalised above
        save_every_ts=10_000_000,       # Checkpoint every 10M timesteps.
        timestep_limit=50_000_000_000,  # Hard training cap at 50B steps.
        log_to_wandb=False,             # Toggle for Weights & Biases logging.
        device=device,
    ) 
    learner.learn()
