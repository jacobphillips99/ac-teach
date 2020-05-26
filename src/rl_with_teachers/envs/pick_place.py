import gym
from gym import spaces
from gym.envs.registration import register
from gym.envs.robotics.fetch import pick_and_place
from rl_with_teachers.teachers.pick_place import *
import random
import numpy as np

class FetchOneGoalPickPlaceEnv(pick_and_place.FetchPickAndPlaceEnv):
    """
    A mujoco task with a fetch robot that needs to pick up a cube and move it to a
    goal point (or just push it there).
    """
    OBJECT_START = np.array([1.25, 0.55])
    OBJECT_START_WITH_Z = np.array([1.25, 0.55, 0.425])
    ROBOT_START = np.array([1.34184371, 0.75, 0.5])
    def __init__(self,
                 sparse=True,
                 goal=np.array([1.45, 0.55, 0.425]),
                 better_dense_reward=False, OBJECT_START_WITH_Z=OBJECT_START_WITH_Z):
        self.goal = goal
        self.better_dense_reward = better_dense_reward
        self.last_object_start = OBJECT_START_WITH_Z
        self.OBJECT_START_WITH_Z = OBJECT_START_WITH_Z
        super().__init__('sparse' if sparse else 'dense')
        self.is_sparse = sparse

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype='float32')

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obj_pos = obs['observation'][3:6]

        #if self.is_sparse:
            #if reward == 0:
                #reward = 1.0
            #else:
                #reward = 0.0
            # reward/=10.0
        #elif self.better_dense_reward:
            # Better dense reward is different from default Fetch
            # dense reward of l2; ours increades from 0 to 1
            # as object nears goal
            # obj_goal = obs['desired_goal']
            #obj_goal = [1.45, .55, 0.425]

            ### give dense reward for distance from block to goal
            #goal_dist = np.linalg.norm(obj_goal - obj_pos)
            #reward = 1. - np.tanh(10.0 * goal_dist)

        reward = -1.0
        done = np.linalg.norm([1.45, .55, 0.425] - obj_pos) + abs(.425-obj_pos[2]) < 0.01
        object_goal = np.array(obs['desired_goal'])
        obs_to_ret = np.concatenate([obs['observation'], object_goal])
        return obs_to_ret, reward, done, info

    def reset(self, repeat=False):
        obs = super().reset()
        self._reset_sim(repeat)
        object_goal = np.array(obs['desired_goal'])
        obs_to_ret = np.concatenate([obs['observation'], object_goal])
        return obs_to_ret

    def _reset_sim(self, repeat=False):
        self.sim.set_state(self.initial_state)

        object_xpos = np.array(FetchOneGoalPickPlaceEnv.OBJECT_START)
        # object_xpos[0]+=(random.random()-0.5)/10.0
        # object_xpos[1]+=(random.random()-0.5)/10.0

        if repeat:
            object_xpos = self.last_object_start

        self.last_object_start = object_xpos

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()

        gripper_offset = (np.random.random(3)-0.5)/10
        gripper_offset[2] = gripper_offset[2]/2
        gripper_target = self.sim.data.get_site_xpos('robot0:grip') + gripper_offset
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
        return True

    def _sample_goal(self):
        goal = self.goal.copy()
        return goal

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [OptimalPickPlaceAgent(self.goal, noise)]
        elif type == 'pick&place':
            picker = PickAgent(noise, return_to_start=True)
            placer = PlaceAgent(self.goal, noise)
            teachers = [picker, placer]
            if drop:
                teachers = [picker]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        else:
            raise ValueError('Not a valid teacher type '+type)

register(
        id='OneGoalPickPlaceEnv-v0',
        entry_point='rl_with_teachers.envs:FetchOneGoalPickPlaceEnv',
        max_episode_steps=100,
        )

class FetchOneGoalPickPlaceSparseEnv(FetchOneGoalPickPlaceEnv):
    def __init__(self, sparse=True):
        super().__init__(True)

register(
        id='OneGoalPickPlaceSparseEnv-v0',
        entry_point='rl_with_teachers.envs:FetchOneGoalPickPlaceSparseEnv',
        max_episode_steps=100,
        )

class FetchOneGoalPickPlaceDenseEnv(FetchOneGoalPickPlaceEnv):
    def __init__(self):
        super().__init__(False)

register(
        id='OneGoalPickPlaceDenseEnv-v0',
        entry_point='rl_with_teachers.envs:FetchOneGoalPickPlaceDenseEnv',
        max_episode_steps=100,
        )

class FetchOneGoalPickPlaceBetterDenseEnv(FetchOneGoalPickPlaceEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True)

register(
        id='FetchOneGoalPickPlaceBetterDense-v0',
        entry_point='rl_with_teachers.envs:FetchOneGoalPickPlaceBetterDenseEnv',
        max_episode_steps=100,
        )

class FetchFarGoalPickPlaceDenseEnv(FetchOneGoalPickPlaceEnv):
    def __init__(self):
        super().__init__(False, goal=np.array([1.45, 0.75, 0.425]))

register(
        id='FarGoalPickPlaceDenseEnv-v0',
        entry_point='rl_with_teachers.envs:FetchFarGoalPickPlaceDenseEnv',
        max_episode_steps=100,
        )

#new for us to use
class PickPlaceRandomGoalEnv(FetchOneGoalPickPlaceEnv):
    def __init__(self):                      # Y can range .45 < y < 1.0
        super().__init__(sparse=False, goal=np.array([1.45, .55, 0.425]), better_dense_reward=True)

register(
        id='PickPlaceRandomGoal-v0',
        entry_point='rl_with_teachers.envs:PickPlaceRandomGoalEnv',
        max_episode_steps=100,
        )
