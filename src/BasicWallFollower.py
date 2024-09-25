#!/usr/bin/env python3

"""
ROS node for wall following using Q-learning.
"""

# imports
import rospy
import random
import math
import pickle
import argparse
import tf.transformations
from pathlib import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState


class BasicWallFollower:
    # front state constants
    FRONT_VERY_CLOSE = 'fvc'
    """State value for very close proximity in front region"""
    FRONT_CLOSE = 'fc'
    """State value for close proximity in front region"""
    FRONT_MEDIUM = 'fm'
    """State value for medium proximity in front region"""
    FRONT_FAR = 'ff'
    """State value for far proximity in front region"""
    FRONT_STATES = [FRONT_VERY_CLOSE, FRONT_CLOSE, FRONT_MEDIUM, FRONT_FAR]
    """List of states for front region"""

    # front right state constants
    FRONT_RIGHT_CLOSE = 'frc'
    """State value for close proximity in front-right region"""
    FRONT_RIGHT_FAR = 'frf'
    """State value for far proximity in front-right region"""
    FRONT_RIGHT_STATES = [FRONT_RIGHT_CLOSE, FRONT_RIGHT_FAR]
    """List of states for front-right region"""

    # right state constants
    RIGHT_VERY_CLOSE = 'rvc'
    """State value for very close proximity in right region"""
    RIGHT_CLOSE = 'rc'
    """State value for close proximity in right region"""
    RIGHT_MEDIUM = 'rm'
    """State value for medium proximity in right region"""
    RIGHT_FAR = 'rf'
    """State value for far proximity in right region"""
    RIGHT_VERY_FAR = 'rvf'
    """State value for very far proximity in right region"""
    RIGHT_STATES = [RIGHT_VERY_CLOSE, RIGHT_CLOSE, RIGHT_MEDIUM, RIGHT_FAR, RIGHT_VERY_FAR]
    """List of states for right region"""

    # back-right state constants
    BACK_RIGHT_CLOSE = 'brc'
    """State value for close proximity in back right region"""
    BACK_RIGHT_FAR = 'brc'
    """State value for close proximity in back right region"""
    BACK_RIGHT_STATES = [BACK_RIGHT_CLOSE, BACK_RIGHT_FAR]

    # left state constants
    LEFT_CLOSE = 'lc'
    """State value for close proximity in left region"""
    LEFT_FAR = 'lf'
    """State value for very close proximity in left region"""
    LEFT_STATES = [LEFT_CLOSE, LEFT_FAR]
    """List of states for left region"""

    # action constants
    FORWARD = 'gf'
    """Action value for moving forward"""
    TURN_LEFT = 'tl'
    """Action value for turning left"""
    TURN_RIGHT = 'tr'
    """Action value for turning right"""
    VALID_ACTIONS = [FORWARD, TURN_LEFT, TURN_RIGHT]
    """List of valid actions that can be undertaken"""

    def __init__(self, learning_rate: float = 0.2,
                 discount_factor: float = 0.8,
                 e_cutoff: int = 500,
                 use_preset: bool = False,
                 region_size: float = 30.0,
                 q_table_path: str = None,
                 qi_table_path: str = None,
                 cr_graph_path: str = None,
                 rc_graph_path: str = None,
                 lc_graph_path: str = None,
                 fc_graph_path: str = None,
                 verbose: bool = False,
                 test: bool = False,
                 fsp: int = None):
        """
        ROS node which uses laser scan data with a Q-table to follow a right wall.
        :param learning_rate: The learning rate (alpha) to use in Q-table updating
        :param discount_factor: The discount factor (gamma) to use in Q-table updating
        :param e_cutoff: The number of episodes needed to minimize epsilon value
        :param use_preset: Whether to use a preset Q-table instead of generating one; should not be set if
         ``q_table_path`` is specified
        :param region_size: Angle to use for arc region size
        :param q_table_path: The path to use for loading/dumping the latest Q-table
        :param qi_table_path: The path to use for dumping Q-table instances
        :param cr_graph_path: The path to use for loading/dumping the cumulative reward graph's data
        :param rc_graph_path: The path to use for loading/dumping the right-close learning graph's data
        :param lc_graph_path: The path to use for loading/dumping the left-close learning graph's data
        :param fc_graph_path: The path to use for loading/dumping the front-close learning graph's data
        :param verbose: Whether the node should log additional information
        :param test: Whether the node should run in test mode (no modification of Q-table/graphs, always
         choosing based on Q table)
        :param fsp: Whether to use a fixed starting position instead of a random one
        """
        # store arc region size
        self.region_size = region_size

        # save verbose setting
        self.verbose = verbose

        # save test setting
        self.test = test

        # save learning rate, discount factor and e cutoff
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_cutoff = e_cutoff

        # save preset flag
        self.use_preset = use_preset

        # save paths for later use
        if q_table_path is not None:  # latest Q table
            self.q_table_path = Path(q_table_path)
        else:
            # create dump path using hash
            new_q_path = f"q_tables/{hash(self)}_q_table.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for latest Q-table provided, using hash path " + new_q_path)
            self.q_table_path = Path(new_q_path)
        if qi_table_path is not None:  # q table instance
            self.qi_table_path = qi_table_path
        else:
            new_qi_path = f"q_tables/instances/{hash(self)}_q_table"
            if self.verbose:
                rospy.logwarn(f"No path start for instance Q-tables provided, using hash path " + new_qi_path)
            self.qi_table_path = new_qi_path
        if cr_graph_path is not None:  # cumulative reward graph
            self.cr_graph_path = Path(cr_graph_path)
        else:
            # create dump path using hash
            new_crg_path = f"graphs/{hash(self)}_cr_graph.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for CR graph provided, using hash path " + new_crg_path)
            self.crg_graph_path = Path(new_crg_path)
        if rc_graph_path is not None:  # RC learning graph
            self.rc_graph_path = Path(rc_graph_path)
        else:
            # create dump path using hash
            new_rcg_path = f"graphs/{hash(self)}_rc_learning_graph.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for RC learning graph provided, using hash path " + new_rcg_path)
            self.rc_graph_path = Path(new_rcg_path)
        if lc_graph_path is not None:  # LC learning graph
            self.lc_graph_path = Path(lc_graph_path)
        else:
            # create dump path using hash
            new_lcg_path = f"graphs/{hash(self)}_lc_learning_graph.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for LC learning graph provided, using hash path " + new_lcg_path)
            self.lc_graph_path = Path(new_lcg_path)
        if fc_graph_path is not None:  # FC learning graph
            self.fc_graph_path = Path(fc_graph_path)
        else:
            # create dump path using hash
            new_fcg_path = f"graphs/{hash(self)}_fc_learning_graph.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for FC learning graph provided, using hash path " + new_fcg_path)
            self.fcg_graph_path = Path(new_fcg_path)

        # generate starting states
        self._create_start_states()

        # initialize node
        rospy.init_node('wall_follower')

        # do not use fixed start position during training
        if not test and fsp is not None:
            rospy.logwarn("Fixed starting point set in training mode, which is not allowed; turning off")
            fsp = None

        # set fsp flag (and index if applicable)
        if fsp is None:
            self.fixed_start = False
        else:
            self.fixed_start = True
            self.fixed_start_index: int = fsp

        # initialize episode counter
        self.episode_count = 0

        # Q table
        if not use_preset:
            # try to load Q-table
            loaded = False
            if q_table_path is not None:
                rospy.loginfo("Attempting to load Q table from %s" % q_table_path)
                if self.q_table_path.is_file():  # assert path exists and goes to a file
                    try:
                        self.load_q_table(self.q_table_path)
                        loaded = True
                        rospy.loginfo("Q table loaded")
                    except IOError as e:
                        rospy.logerr(f"Could not load Q table from {q_table_path}:\n{e}")
                    except pickle.UnpicklingError as e:
                        rospy.logerr(f"Could not load Q table from {q_table_path}:\n{e}")
                else:
                    rospy.logwarn(f"No Q table file at %s" % q_table_path)

            # create empty Q table if loading failed or was not attempted
            # only supported during training, abort if loading failed or not attempted when trying to init. in
            # testing mode
            if not loaded:
                # assert that node is in training mode
                if not test:
                    rospy.loginfo("Creating blank Q table, this may take a moment")
                    self.q_table = _init_q_table()
                else:
                    # no Q table for testing, abort
                    raise IOError("Could not load Q table when a non-blank Q table is required for testing, aborting; "
                                  "if q_table_path was set, there will likely be an error message above this exception "
                                  "that details why the Q table was unable to load")
        else:
            # manual fill
            if q_table_path is not None:
                rospy.logwarn("use_preset flag is set but q_table_path is also set; defaulting to preset")
            rospy.loginfo("Creating preset Q table, this may take a moment")
            self.q_table = _manual_fill_q_table()

        # learning graphs
        if not use_preset:
            # create cumulative reward graph counters
            self.cr_episode_counter = 0  # episodes since cumulative reward graph creations
            self.cumulative_reward = 0  # cumulative reward

            # create right close learning graph counters
            self.rc_episode_counter = 0  # episodes since right close learning graph creation
            self.correct_rc_answers = 0  # number of correct choices in right close scenario
            self.total_rc_answers = 0  # number of right close scenarios encountered

            # create left close learning graph counters
            self.lc_episode_counter = 0  # episodes since left close learning graph creation
            self.correct_lc_answers = 0  # number of correct choices in left close scenario
            self.total_lc_answers = 0  # number of left close scenarios encountered

            # create front close learning graph counters
            self.fc_episode_counter = 0  # episodes since right close learning graph creation
            self.correct_fc_answers = 0  # number of correct choices in front close scenario
            self.total_fc_answers = 0  # number of front close scenarios encountered

            # try to load CR graph
            loaded = False
            if cr_graph_path is not None:
                rospy.loginfo("Attempting to load CR graph from %s" % cr_graph_path)
                if self.cr_graph_path.is_file():
                    try:
                        self.load_graph_data(self.cr_graph_path, 'cr')
                        loaded = True
                        rospy.loginfo("Graph loaded")
                    except IOError as e:
                        rospy.logerr(f"Could not load CR graph from {rc_graph_path}:\n{e}")
                    except pickle.PicklingError as e:
                        rospy.logerr(f"Could not load CR graph from {rc_graph_path}:\n{e}")
                else:
                    rospy.logwarn("No CR graph file at %s" % rc_graph_path)

            # create empty graph data holder if loading failed or was not attempted
            if not loaded:
                rospy.loginfo("Creating blank CR graph")
                self.cr_graph_data = []

            # try to load RC graph
            loaded = False
            if rc_graph_path is not None:
                rospy.loginfo("Attempting to load RC learning graph from %s" % rc_graph_path)
                if self.rc_graph_path.is_file():
                    try:
                        self.load_graph_data(self.rc_graph_path, 'rc')
                        loaded = True
                        rospy.loginfo("Graph loaded")
                    except IOError as e:
                        rospy.logerr(f"Could not load RC learning graph from {rc_graph_path}:\n{e}")
                    except pickle.PicklingError as e:
                        rospy.logerr(f"Could not load RC learning graph from {rc_graph_path}:\n{e}")
                else:
                    rospy.logwarn("No RC learning graph file at %s" % rc_graph_path)

            # create empty graph data holder if loading failed or was not attempted
            if not loaded:
                rospy.loginfo("Creating blank RC learning graph")
                self.rc_graph_data = []

            # try to load LC graph
            loaded = False
            if lc_graph_path is not None:
                rospy.loginfo("Attempting to load LC learning graph from %s" % lc_graph_path)
                if self.lc_graph_path.is_file():
                    try:
                        self.load_graph_data(self.lc_graph_path, 'lc')
                        loaded = True
                        rospy.loginfo("Graph loaded")
                    except IOError as e:
                        rospy.logerr(f"Could not load LC learning graph from {lc_graph_path}:\n{e}")
                    except pickle.PicklingError as e:
                        rospy.logerr(f"Could not load LC learning graph from {lc_graph_path}:\n{e}")
                else:
                    rospy.logwarn("No LC learning graph file at %s" % lc_graph_path)

            # create empty graph data holder if loading failed or was not attempted
            if not loaded:
                rospy.loginfo("Creating blank LC learning graph")
                self.lc_graph_data = []

            # try to load FC graph
            loaded = False
            if fc_graph_path is not None:
                rospy.loginfo("Attempting to load FC learning graph from %s" % fc_graph_path)
                if self.fc_graph_path.is_file():
                    try:
                        self.load_graph_data(self.fc_graph_path, 'fc')
                        loaded = True
                        rospy.loginfo("Graph loaded")
                    except IOError as e:
                        rospy.logerr(f"Could not load FC learning graph from {fc_graph_path}:\n{e}")
                    except pickle.PicklingError as e:
                        rospy.logerr(f"Could not load FC learning graph from {fc_graph_path}:\n{e}")
                else:
                    rospy.logwarn("No FC learning graph file at %s" % fc_graph_path)

            # create empty graph data holder if loading failed or was not attempted
            if not loaded:
                rospy.loginfo("Creating blank FC learning graph")
                self.fc_graph_data = []

        # create laser scan listener holder
        self.scanner = None

        # create client holders
        self.reset_client = None
        self.get_position_client = None
        self.set_position_client = None
        self.pause_client = None
        self.unpause_client = None

        # create other holders
        self.raw_state = None
        self.state = None
        self.front = 0.0
        self.front_right = 0.0
        self.right = 0.0
        self.back_right = 0.0
        self.left = 0.0
        self.prev_action = None
        self.prev_state = None

        # create state update flag
        self.state_updated = False

        # initialize action publisher
        self.action_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # initialize termination counters
        self.stuck_counter = 0
        self.good_iteration_counter = 0

        # log node creation
        rospy.loginfo("wall_follower node created")

    def __del__(self):
        # log shutdown
        rospy.loginfo("Shutting down wall follower node")

        # dump Q table & learning graphs if not preset/test mode
        if not self.use_preset and not self.test:
            self.dump_q_table(self.q_table_path)
            self.dump_all_graph_data()

        # kill laser scan listener
        if self.scanner is not None:
            self.scanner.unregister()

        # kill action publisher
        self.action_publisher.unregister()

        # shutdown node
        rospy.signal_shutdown("Node terminated by code")

    def _update_q_table(self, initial_state, end_state, action, reward):
        """
        Updates the Q-table based on the results of an action taken by the agent.
        :param initial_state: The state of the agent before performing the given action
        :param end_state: The state of the agent after performing the given action
        :param action: The action taken by the agent that resulted in ``end_state``
        :param reward: The reward calculated after performing the given action
        """
        # determine old Q value
        old_q_value = self._q(initial_state, action)

        # determine estimate of optimal future value
        max_future_q = -float('inf')
        for possible_action in BasicWallFollower.VALID_ACTIONS:
            future_q = self._q(end_state, possible_action)
            if future_q > max_future_q:
                max_future_q = future_q

        # calculate new Q value
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q_value)

        # store new Q value in table
        if self.verbose:
            rospy.loginfo("New Q-value for state \"%s\": %f", create_state_key(initial_state), new_q_value)
        self.q_table[create_state_key(initial_state)][action] = new_q_value

    def _q(self, state, action):
        """
        Calculates the Q-value for performing the specified action in the given state.
        :param state: The current state of the agent
        :param action: The action to be performed
        :return: The Q-value for the given action in the given state
        """
        # get state key
        state_key = create_state_key(state)

        # return Q table value
        return self.q_table[state_key][action]

    def _discretize(self):
        """
        Performs discretization of arc region ranges.
        :return: ``True`` if
        """
        # front state
        if self.front < 0.5:
            front_state = BasicWallFollower.FRONT_VERY_CLOSE
        elif self.front < 0.6:
            front_state = BasicWallFollower.FRONT_CLOSE
        elif self.front <= 1.2:
            front_state = BasicWallFollower.FRONT_MEDIUM
        else:
            front_state = BasicWallFollower.FRONT_FAR

        # front-right state
        if self.front_right <= 1.2:
            front_right_state = BasicWallFollower.FRONT_RIGHT_CLOSE
        else:
            front_right_state = BasicWallFollower.FRONT_RIGHT_FAR

        # right state
        if self.right < 0.2:
            right_state = BasicWallFollower.RIGHT_VERY_CLOSE
        elif self.right < 0.3:
            right_state = BasicWallFollower.RIGHT_CLOSE
        elif self.right < 0.5:
            right_state = BasicWallFollower.RIGHT_MEDIUM
        elif self.right <= 1.2:
            right_state = BasicWallFollower.RIGHT_FAR
        else:
            right_state = BasicWallFollower.RIGHT_VERY_FAR

        # back-right state
        if self.back_right <= 1.2:
            back_right_state = BasicWallFollower.BACK_RIGHT_CLOSE
        else:
            back_right_state = BasicWallFollower.BACK_RIGHT_FAR

        # left state
        if self.left <= 0.5:
            left_state = BasicWallFollower.LEFT_CLOSE
        else:
            left_state = BasicWallFollower.LEFT_FAR

        # compile into state tuple
        self.state = (front_state, front_right_state, right_state, back_right_state, left_state)

    def _min_regions(self):
        """
        Uses state data to calculate the minimum ranges for each arc region.
        """
        # initialize minimum fields
        f_min = float('inf')
        fr_min = float('inf')
        r_min = float('inf')
        br_min = float('inf')
        l_min = float('inf')

        # find minimum range for each region
        for i in range(len(self.raw_state[0])):
            # get range
            s_range = self.raw_state[0][i]

            # determine region & if it is the minimum reading for that region
            if self.raw_state[2][i] == 'f':  # front
                if s_range <= f_min:
                    f_min = s_range
            elif self.raw_state[2][i] == 'fr':  # front-right
                if s_range <= fr_min:
                    fr_min = s_range
            elif self.raw_state[2][i] == 'r':  # right
                if s_range <= r_min:
                    r_min = s_range
            elif self.raw_state[2][i] == 'br':  # back-right
                if s_range <= br_min:
                    br_min = s_range
            elif self.raw_state[2][i] == 'l':  # left
                if s_range <= l_min:
                    l_min = s_range

        # pass to fields
        self.front = f_min
        self.front_right = fr_min
        self.right = r_min
        self.left = l_min

    def _get_angle_region(self, angle):
        if _in_angle_region(angle, 0, self.region_size):  # front
            return 'f'
        elif _in_angle_region(angle, 315, self.region_size):  # front-right
            return 'fr'
        elif _in_angle_region(angle, 270, self.region_size):  # right
            return 'r'
        elif _in_angle_region(angle, 225, self.region_size):  # back-right
            return 'br'
        elif _in_angle_region(angle, 90, self.region_size):  # left
            return 'l'
        else:  # none (will be ignored during state discretization)
            return 'n'

    def _add_to_cumulative_reward(self, reward):
        """
        Adds the given reward to the cumulative reward.
        """
        self.cumulative_reward += reward

    def _calculate_e(self):
        """
        Calculates the e-value for the current episode.
        """
        x = self.episode_count / self.e_cutoff
        if x > 1:
            x = 1
        return 0.9 + (-0.8 * x)

    def _choose_manual_action(self, state):
        print(f"State: {create_state_key(state)}")
        # get action scores
        action_scores = {}
        for action in BasicWallFollower.VALID_ACTIONS:
            action_scores[action] = self._q(state, action)
        print(f"Actions & Q-values: {action_scores}")
        a_list = ['Q']
        a_list.extend(BasicWallFollower.VALID_ACTIONS)
        a = ""
        while a not in a_list and not rospy.is_shutdown():
            a = input("Choose an action or type Q to use highest Q-value:\t").strip()
            if a not in a_list:
                print("Invalid input, try again")
        if a == "q":
            # choose the highest scoring action
            top_action = None
            top_score = -float('inf')
            for action in BasicWallFollower.VALID_ACTIONS:
                if action_scores[action] > top_score:
                    top_action = action
                    top_score = action_scores[action]

            # return chosen action
            return top_action
        elif a in BasicWallFollower.VALID_ACTIONS:
            return a
        else:
            # impossible state
            raise RuntimeError("Impossible state")

    def _choose_action(self, state):
        """
        Selects an action to undertake based on its Q score for the given state.
        :param state: The current state of the robot
        :return: The selected action
        """
        # get action scores
        action_scores = {}
        for action in BasicWallFollower.VALID_ACTIONS:
            action_scores[action] = self._q(state, action)

        # choose the highest scoring action
        top_action = None
        top_score = -float('inf')
        for action in BasicWallFollower.VALID_ACTIONS:
            if action_scores[action] > top_score:
                top_action = action
                top_score = action_scores[action]

        # return chosen action
        return top_action

    def _perform_action(self, action):
        """
        Has the robot perform the given action.
        :param action: The action to undertake
        """
        # create action message
        action_msg = Twist()

        # set action fields based on action selected
        if action == BasicWallFollower.FORWARD:
            action_msg.linear.x = 0.2
        elif action == BasicWallFollower.TURN_LEFT:
            action_msg.linear.x = 0.2
            action_msg.angular.z = _to_radians(90)
        elif action == BasicWallFollower.TURN_RIGHT:
            action_msg.linear.x = 0.2
            action_msg.angular.z = _to_radians(-90)
        else:
            # invalid action, abort
            rospy.logerr("Invalid action code \'%s\'; aborting action performance" % action)
            return

        # publish action
        self.action_publisher.publish(action_msg)

    def _create_start_states(self):
        """
        Generates a hardcoded list of 8 starting states as ModelState objects.
        :return: The list of starting states (in ``self.starting_states``)
        """
        # create empty starting state list
        starting_states = []

        # first state (index 0)
        model_state_1 = _create_model_state(-1.95, 1.95, -90.0)
        starting_states.append(model_state_1)

        # second state (index 1)
        model_state_2 = _create_model_state(-2.12, -1.47, -90.0)
        starting_states.append(model_state_2)

        # third state (index 2)
        model_state_3 = _create_model_state(-0.15, -0.96, 0.0)
        starting_states.append(model_state_3)

        # fourth state (index 3)
        model_state_4 = _create_model_state(0.14, -1.73, -90.0)
        starting_states.append(model_state_4)

        # fifth state (index 4)
        model_state_5 = _create_model_state(2.0, -1.71, 90.0)
        starting_states.append(model_state_5)

        # sixth state (index 5)
        model_state_6 = _create_model_state(1.8, -0.81, 180)
        starting_states.append(model_state_6)

        # seventh state (index 6)
        model_state_7 = _create_model_state(1.92, 0.38, 90.0)
        starting_states.append(model_state_7)

        # eighth state (index 7)
        model_state_8 = _create_model_state(0.8, 2.03, 180.0)
        starting_states.append(model_state_8)

        # send to field
        self.starting_states = starting_states

    def _set_start_state(self):
        """
        Sets the start state of the robot. If FSP is set, uses that index; otherwise selects randomly from a list of 8
        hardcoded starting states.
        """
        if self.fixed_start:
            # get chosen model state
            model_state = self.starting_states[self.fixed_start_index]
        else:
            # get random model state
            model_state = random.choice(self.starting_states)

        # send model state
        if self.verbose:
            rospy.loginfo("Setting start state to:\n" + str(model_state))
        self.set_position_client(model_state)

    def _update_termination_counters(self, previous_state, current_state):
        """
        Checks whether the agent hasn't been able to move this iteration; if so, updates an internal counter for the
        agent being stuck, otherwise updates an internal counter for the agent successfully following a wall
        :param previous_state: The previous state of the agent's model
        :param current_state: The current state of the agent's model
        """
        # check if positions are equal within tolerance
        if _states_equal(previous_state, current_state):
            # robot is stuck this iteration
            rospy.logwarn("Did not move this iteration")
            self.stuck_counter += 1
            self.good_iteration_counter = 0
        else:
            self.good_iteration_counter += 1
            self.stuck_counter = 0

    def _reset_state(self):
        """
        Resets the Gazebo simulation. Meant to be called when the robot can no longer move (i.e. it has hit a wall).
        """
        # perform reset by resetting model state
        if self.verbose:
            rospy.logwarn("State reset called")
        self._set_start_state()

    def _update_graph_data(self, state, action):
        """
        Determines if a learning graph should be updated; if so, updates the graph.
        :param state: The initial state of this iteration
        :param action: The action taken this iteration
        """
        # check for scenarios with obvious answers or obvious wrong answers
        # (turn left when right close, turn right when left close, turn instead of move forward when front close)
        if state[2] == BasicWallFollower.RIGHT_VERY_CLOSE or state[2] == BasicWallFollower.RIGHT_CLOSE:
            if self.verbose:
                rospy.loginfo("Updating RC learning graph")

            # update counters
            self.total_rc_answers += 1
            if action == BasicWallFollower.TURN_LEFT:
                self.correct_rc_answers += 1
        elif state[3] == BasicWallFollower.LEFT_CLOSE:
            if self.verbose:
                rospy.loginfo("Updating LC learning graph")

            # update counters
            self.total_lc_answers += 1
            if action == BasicWallFollower.TURN_RIGHT:
                self.correct_lc_answers += 1
        elif state[0] == BasicWallFollower.FRONT_VERY_CLOSE or state[0] == BasicWallFollower.FRONT_CLOSE:
            if self.verbose:
                rospy.loginfo("Updating FC learning graph")

            # update counters
            self.total_fc_answers += 1
            if action == BasicWallFollower.TURN_RIGHT or action == BasicWallFollower.TURN_LEFT:
                self.correct_fc_answers += 1

    def _create_graph_point(self):
        """
        Adds graph datapoints for each graph.
        """
        # add CR data point to graph
        if self.verbose:
            rospy.loginfo(f"Cumulative reward this episode: {self.cumulative_reward}")
        self.cr_graph_data.append((self.cr_episode_counter, self.cumulative_reward))
        if self.verbose:
            dp = self.cr_graph_data[len(self.cr_graph_data) - 1]
            rospy.loginfo(f"Added CR data point ({dp[0]}, {dp[1]})")

        # add RC data point to graph
        if self.total_rc_answers > 0:
            if self.verbose:
                rospy.loginfo("% correct choices for RC this episode: "
                              f"{self.correct_rc_answers / self.total_rc_answers} "
                              f"({self.correct_rc_answers}/{self.total_rc_answers})")
            self.rc_graph_data.append((self.rc_episode_counter, self.correct_rc_answers / self.total_rc_answers))
        else:
            if self.verbose:
                rospy.loginfo("% correct choices for RC this episode: 0 (0/0)")
            self.rc_graph_data.append((self.rc_episode_counter, 0))
        if self.verbose:
            dp = self.rc_graph_data[len(self.rc_graph_data) - 1]
            rospy.loginfo(f"Added RC data point ({dp[0]}, {dp[1]})")

        # add LC data point to graph
        if self.total_lc_answers > 0:
            if self.verbose:
                rospy.loginfo("% correct choices for LC this episode: "
                              f"{self.correct_lc_answers / self.total_lc_answers} "
                              f"({self.correct_lc_answers}/{self.total_lc_answers})")
            self.lc_graph_data.append((self.lc_episode_counter, self.correct_lc_answers / self.total_lc_answers))
        else:
            if self.verbose:
                rospy.loginfo("% correct choices for LC this episode: 0 (0/0)")
            self.lc_graph_data.append((self.lc_episode_counter, 0))
        if self.verbose:
            dp = self.lc_graph_data[len(self.lc_graph_data) - 1]
            rospy.loginfo(f"Added LC data point ({dp[0]}, {dp[1]})")

        # add FC data point to graph
        if self.total_fc_answers > 0:
            if self.verbose:
                rospy.loginfo("% correct choices for FC this episode: "
                              f"{self.correct_fc_answers / self.total_fc_answers} "
                              f"({self.correct_fc_answers}/{self.total_fc_answers})")
            self.fc_graph_data.append((self.fc_episode_counter, self.correct_fc_answers / self.total_fc_answers))
        else:
            if self.verbose:
                rospy.loginfo("% correct choices for FC this episode: 0 (0/0)")
            self.fc_graph_data.append((self.fc_episode_counter, 0))
        if self.verbose:
            dp = self.fc_graph_data[len(self.fc_graph_data) - 1]
            rospy.loginfo(f"Added FC data point ({dp[0]}, {dp[1]})")

    def load_q_table(self, path: Path):
        """
        Loads a Q-table from the specified Pickle file path.
        :param path: The file path to load the Q-table from (relative to current working directory)
        """
        with path.open("rb") as file:
            self.episode_count, self.q_table = pickle.load(file)

    def load_graph_data(self, path: Path, target: str):
        """
        Loads learning graph data and the last absolute iteration count from the specified Pickle file path.
        :param path: The file path to load the graph data from (relative to current working directory)
        :param target: The key for the target graph to load data into (can be 'rc', 'lc', or 'fc')
        :raise ValueError: if ``target`` is not valid
        """
        # load data
        with path.open("rb") as file:
            raw = pickle.load(file)

        # partition
        if target == 'cr':
            self.cr_episode_counter = raw['iter']
            self.cumulative_reward = raw['total']
            self.cr_graph_data = raw['data']
        elif target == 'rc':
            self.rc_episode_counter = raw['iter']
            self.correct_rc_answers = raw['correct']
            self.total_rc_answers = raw['total']
            self.rc_graph_data = raw['data']
        elif target == 'lc':
            self.lc_episode_counter = raw['iter']
            self.correct_lc_answers = raw['correct']
            self.total_lc_answers = raw['total']
            self.lc_graph_data = raw['data']
        elif target == 'fc':
            self.fc_episode_counter = raw['iter']
            self.correct_fc_answers = raw['correct']
            self.total_fc_answers = raw['total']
            self.fc_graph_data = raw['data']
        else:
            # bad target graph key
            raise ValueError("Invalid target graph key \'%s\'; must be \'cr\', \'rc\', \'lc\', or \'fc\'" % target)

    def dump_q_table(self, path: Path):
        """
        Dumps the node's Q-table to the specified file path using Pickle.
        :param path: The file path to dump the Q-table to (relative to current working directory)
        """
        # latest instance dump
        rospy.loginfo("Dumping latest Q-table to %s" % path)
        with path.open("wb") as file:
            pickle.dump((self.episode_count, self.q_table), file)

        # not-overwrite-able instance dump
        complete_q_instance_path = Path((self.qi_table_path + str(self.episode_count) + ".pkl"))
        rospy.loginfo("Dumping Q-table instance to %s" % complete_q_instance_path)
        with complete_q_instance_path.open("wb") as file:
            pickle.dump((self.episode_count, self.q_table), file)

    def dump_graph_data(self, path: Path, graph: str = None):
        """
        Dumps the node's learning graph data from the specified graph to the specified file path using Pickle.
        :param path: The file path to dump the graph data to (relative to current working directory)
        :param graph: The key of graph to dump - can be either \'rc\', \'lc\', or \'fc\'
        :raise ValueError: If an invalid graph key is provided
        """
        # construct data structure to dump
        if graph == 'cr':
            raw = {'iter': self.cr_episode_counter, 'total': self.cumulative_reward,
                   'data': self.cr_graph_data, 'type': 'cr'}
        elif graph == 'rc':
            raw = {'iter': self.rc_episode_counter, 'correct': self.correct_rc_answers,
                   'total': self.total_rc_answers, 'data': self.rc_graph_data, 'type': 'rc'}
        elif graph == 'lc':
            raw = {'iter': self.lc_episode_counter, 'correct': self.correct_lc_answers,
                   'total': self.total_lc_answers, 'data': self.lc_graph_data, 'type': 'lc'}
        elif graph == 'fc':
            raw = {'iter': self.fc_episode_counter, 'correct': self.correct_fc_answers,
                   'total': self.total_fc_answers, 'data': self.fc_graph_data, 'type': 'fc'}
        else:
            # bad graph key
            raise ValueError("Invalid graph key \'%s\'; must be \'cr\', \'rc\', \'lc\', or \'fc\'" % graph)

        # dump data
        rospy.loginfo("Dumping %s learning graph to %s" % (graph, path))
        with path.open("wb") as file:
            pickle.dump(raw, file)

    def dump_all_graph_data(self):
        """
        Dumps all learning graph data associated with the node via Pickle.
        """
        # dump CR graph
        self.dump_graph_data(self.cr_graph_path, 'cr')

        # dump RC graph
        self.dump_graph_data(self.rc_graph_path, 'rc')

        # dump LC graph
        self.dump_graph_data(self.lc_graph_path, 'lc')

        # dump FC graph
        self.dump_graph_data(self.fc_graph_path, 'fc')

    def poll_state(self):
        """
        Gets the current discretized state of the robot.
        :return: The discretized state of the robot
        """
        return self.state

    def scan_callback(self, msg):
        """
        Called when a scanner percept is sent to the node's listener.
        :param msg: The LaserScan message received
        """
        # get angle range from reading; angle max and min expected to be integers
        min_angle = int(_to_degrees(msg.angle_min))
        max_angle = int(_to_degrees(msg.angle_max))
        angle_inc = _to_degrees(msg.angle_increment)

        # get ranges from reading
        min_range = msg.range_min
        max_range = msg.range_max
        ranges = msg.ranges

        # create list of angles
        angles = _get_angles(len(ranges), min_angle, angle_inc)

        # assert angles do not exceed maximum
        actual_max = int(max(angles))
        if actual_max > max_angle:
            rospy.logerr("Expected max angle is exceeded (%d > %d); assuming bad data and aborting"
                         % (actual_max, max_angle))
            return

        # assert that there is an angle associated with each range
        if len(angles) != len(ranges):
            rospy.logerr("Mismatch between number of angles and ranges; assuming bad data and aborting")
            return

        # make sure each range is valid
        valid_ranges = []
        valid_angles = []
        for i in range(0, len(ranges)):
            # check if range is valid
            if ranges[i] < min_range or ranges[i] > max_range:
                if ranges[i] == float('inf'):
                    # set to max range
                    t_range = max_range
                else:
                    # log bad range and skip
                    rospy.logwarn('Range %d outside possible range values [%d, %d]; discarding'
                                  % (ranges[i], min_range, max_range))
                    continue
            else:
                t_range = ranges[i]

            # add to list of valid ranges
            valid_ranges.append(t_range)
            valid_angles.append(angles[i])

        # determine what arc regions a scan component falls in
        regions = []
        for angle in valid_angles:
            regions.append(self._get_angle_region(angle))

        # compile into raw state
        self.raw_state = (valid_ranges, valid_angles, regions)

        # average into regions and discretize
        self._min_regions()
        self._discretize()

        # set update flag
        self.state_updated = True

    def print_state(self):
        """
        Prints the current state
        :return:
        """
        state_str = (f"STATE: {self.state[0]} ({self.front}), {self.state[1]} ({self.front_right}), "
                     f"{self.state[2]} ({self.right}, {self.state[3]} ({self.left})")
        rospy.loginfo(state_str)

    def run(self):
        """
        Starts node functionality.
        """
        # make sure services are ready
        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
        if self.verbose:
            rospy.loginfo("All services required are ready")

        # initialize scan listener
        self.scanner = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        # initialize reset, pause, and unpause clients
        self.reset_client = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.pause_client = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # initialize position clients and set start position at random
        self.get_position_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_position_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._set_start_state()

        # initial pause
        self.pause_client()

        # create counter for total iterations
        iteration_counter = 0

        # create counter for iterations per episode
        episode_iteration_counter = 0

        # get initial e threshold
        e_threshold = self._calculate_e()

        # main loop
        if self.verbose:
            rospy.loginfo("==NEW EPISODE==")
        while not rospy.is_shutdown():
            if self.verbose:
                rospy.loginfo("Starting iteration #%d" % (episode_iteration_counter + 1))

            # get state
            self.unpause_client()
            if self.verbose:
                rospy.loginfo("Waiting for state update")
            while not self.state_updated and not rospy.is_shutdown():
                continue
            current_state = self.poll_state()
            self.pause_client()
            self.state_updated = False
            if self.verbose:
                rospy.loginfo("Got current state")

            # get model state
            current_model_state = self.get_position_client("turtlebot3_burger", "world")

            # determine whether to select action randomly or using Q-table
            if not self.use_preset and not self.test:
                if self.verbose:
                    rospy.loginfo("Current E-value: " + str(e_threshold))
                do_random = _do_random_action(e_threshold)  # determine whether to use random or Q table
            else:
                do_random = False  # always use Q table when preset or testing

            # determine action to take
            #action = self._choose_manual_action(current_state)  # teleoperation
            if do_random:
                action = _choose_random_action()
                if self.verbose:
                    rospy.loginfo("Randomly selected action key \'%s\'" % action)
            else:
                action = self._choose_action(current_state)
                if self.verbose:
                    rospy.loginfo("Selected action key \'%s\' (Q-value %d)" % (action, self._q(current_state, action)))

            # perform action
            self.unpause_client()
            if self.verbose:
                rospy.loginfo("Performing action")
            self._perform_action(action)
            rospy.sleep(0.3)  # wait for action to be completed
            self.pause_client()

            # get new state
            self.unpause_client()
            if self.verbose:
                rospy.loginfo("Waiting for state update")
            while not self.state_updated and not rospy.is_shutdown():
                continue
            new_state = self.poll_state()
            self.pause_client()
            self.state_updated = False
            if self.verbose:
                rospy.loginfo("Got new state")

            # get new model state
            new_model_state = self.get_position_client("turtlebot3_burger", "world")

            # update Q-table if not preset or test mode
            if not self.use_preset and not self.test:
                # determine reward
                reward = _calculate_reward(new_state)

                # update cumulative reward if not random action
                if not do_random:
                    self._add_to_cumulative_reward(reward)

                # update Q table
                self._update_q_table(current_state, new_state, action, reward)
                if self.verbose:
                    rospy.loginfo("Updated Q table")

            # update iteration counters
            iteration_counter += 1

            # update learning graph & CR graph if applicable
            if not self.use_preset and not do_random and not self.test:
                self._update_graph_data(current_state, action)

            # dump data every 100 iterations
            already_dumped = False
            if iteration_counter % 100 == 0 and not self.use_preset and not self.test:
                rospy.loginfo("Auto-saving Q table & learning graphs")
                self.dump_q_table(self.q_table_path)
                self.dump_all_graph_data()
                already_dumped = True

            # update condition counters
            if self.verbose:
                rospy.loginfo("Updating condition counters")
            episode_iteration_counter += 1
            self._update_termination_counters(current_model_state, new_model_state)

            # check for termination (cannot move)
            resetting = False
            if self.stuck_counter >= 3:
                # log error & perform reset
                rospy.logerr("Agent can no longer move (stuck for 3 iterations), resetting simulation")
                resetting = True
            elif self.verbose:
                rospy.loginfo("Stuck counter: %d/3" % self.stuck_counter)

            # check for termination (ran for 300 iterations)
            if episode_iteration_counter >= 300 and not resetting and not self.test:
                # log & perform reset
                rospy.logwarn("Iterations per episode limit reached, restarting simulation")
                resetting = True
            elif self.verbose:
                rospy.loginfo("Iterations per episode count: %d/300" % episode_iteration_counter)

            # check for success termination (followed wall for 1000 iterations)
            if self.good_iteration_counter >= 1000 and not self.test:
                # log success & break
                rospy.loginfo("Agent has successfully followed the wall for 1000 consecutive iterations")
                if not self.use_preset and not self.test:
                    self._create_graph_point()  # add point to learning graph
                break
            elif self.verbose:
                rospy.loginfo("Successful iteration count: %d/1000" % self.good_iteration_counter)

            # reset if to be performed
            if resetting:
                self._reset_state()  # reset state to random starting position
                self.stuck_counter = 0
                if not self.use_preset and not self.test:
                    self._create_graph_point()  # add point to learning graphs
                episode_iteration_counter = 0  # new episode
                self.episode_count += 1
                self.cr_episode_counter += 1
                self.rc_episode_counter += 1
                self.lc_episode_counter += 1
                self.fc_episode_counter += 1
                if not already_dumped and not self.use_preset and not self.test:
                    # dump q table and graph data
                    rospy.loginfo("Auto-saving Q table & learning graphs")
                    self.dump_q_table(self.q_table_path)
                    self.dump_all_graph_data()
                e_threshold = self._calculate_e()
                if self.verbose:
                    rospy.loginfo("==NEW EPISODE==\n(%d/%d)" % (self.episode_count, self.e_cutoff))

            # # sleep
            # self.unpause_client()
            # rospy.sleep(1)
            # self.pause_client()


def _to_degrees(rad):
    """
    Helper function. Converts radians to degrees.
    :param rad: The radian value to convert
    :return: The corresponding degree value
    """
    return rad * 180 / math.pi


def _to_radians(deg):
    """
    Helper function. Converts degrees to radians.
    :param deg: The degree value to convert
    :return: The corresponding degree value
    """
    return deg * math.pi / 180


def _z_rotation(angle):
    """
    Helper function. Creates a quaternion containing a rotation on the Z axis.
    :param angle: The angle (in degrees) to use for the rotation
    :return: The quaternion containing the desired rotation on the Z axis
    """
    # convert angle to radians
    rad_angle = _to_radians(angle)

    # create quaternion and return
    return tf.transformations.quaternion_from_euler(0.0, 0.0, rad_angle)


def _create_model_state(x_pos: float, y_pos: float, z_ori: float) -> ModelState:
    """
    Creates a 2D model state for the turtle.
    :param x_pos: The x-position of the turtle
    :param y_pos: The y-position of the turtle
    :param z_ori: The z-orientation of the turtle in degrees
    :return: The created model state
    """
    # create model state & set turtle model name
    model_state = ModelState()
    model_state.model_name = "turtlebot3_burger"

    # set model position
    model_state.pose.position.x = x_pos
    model_state.pose.position.y = y_pos

    # set model orientation
    orientation = _z_rotation(z_ori)
    model_state.pose.orientation.x = orientation[0]
    model_state.pose.orientation.y = orientation[1]
    model_state.pose.orientation.z = orientation[2]
    model_state.pose.orientation.w = orientation[3]

    # return filled model state
    return model_state


def _get_angles(num_angles, ang_min, ang_inc):
    """
    Helper function. Creates a list of angles based on the inputted data.
    :param num_angles: The number of angles to generate
    :param ang_min: The minimum angle in degrees
    :param ang_inc: The angle increment in degrees used for separation
    :return: A list of ``data_len`` angles within the range [``ang_min``, ``ang_min`` + ``num_angles`` * ``ang-_inc``],
     each separated by an angle of ``ang_inc``
    """
    # create angle list
    angles = []

    # get angles
    current_angle = ang_min
    for i in range(num_angles):
        # save current angle
        angles.append(current_angle)

        # increment current angle
        current_angle += ang_inc

    # return angle list
    return angles


def _in_angle_region(angle: float, region_center: float, region_size: float) -> bool:
    """
    Helper function. Determines if an angle falls into a given region using interpolation. Assumes a region is
    an arc defined by two angles in degrees: a center angle and an arc angle.
    :param angle: The angle in degrees to check
    :param region_center: The center of the arc region to check against
    :param region_size: The size of the arc region to check against
    :return: ``True`` if the angle falls into the given arc region, ``False`` otherwise
    """
    # calculate difference between region center and angle being checked
    angle_diff = ((region_center - angle) + 180 + 360) % 360 - 180

    # return whether difference puts angle within arc region
    return -region_size <= angle_diff <= region_size


def _states_equal(state_1, state_2, position_tolerance: float = 0.02, orientation_tolerance: float = 0.03) -> bool:
    """
    Helper function that checks if two model states are equal within the given tolerance.
    :param state_1: The first model state to be compared
    :param state_2: The second model state to be compared
    :param position_tolerance: The maximum allowed difference between two position elements of the states for them to
     still be considered equal
    :param orientation_tolerance: The maximum allowed difference between the orientations of the states for them to
     still be considered equal
    :return: ``True`` if the model states are equal within the given tolerance, ``False`` otherwise
    """
    # compare position
    if abs(state_2.pose.position.x - state_1.pose.position.x) > position_tolerance:
        return False
    if abs(state_2.pose.position.y - state_1.pose.position.y) > position_tolerance:
        return False

    # compare orientation
    if abs(state_2.pose.orientation.z - state_1.pose.orientation.z) > orientation_tolerance:
        return False

    # all checks passed
    return True


def _init_q_table():
    """
    Creates a Q-table for wall following filled with all zeroes.
    :return: The created Q table
    """
    # iterate through all possible combos to construct Q table
    q_table = {}
    for front_value in BasicWallFollower.FRONT_STATES:
        for front_right_value in BasicWallFollower.FRONT_RIGHT_STATES:
            for right_value in BasicWallFollower.RIGHT_STATES:
                for back_right_value in BasicWallFollower.BACK_RIGHT_STATES:
                    for left_value in BasicWallFollower.LEFT_STATES:
                        # create state key
                        state_key = create_state_key((front_value,
                                                      front_right_value,
                                                      right_value,
                                                      back_right_value,
                                                      left_value))

                        # create action dict with zeroes
                        actions = {}
                        for action in BasicWallFollower.VALID_ACTIONS:
                            actions[action] = 0.0

                        # add state and actions to Q table
                        q_table[state_key] = actions

    # return constructed Q table
    return q_table


def _manual_fill_q_table():
    """
    'Manually' fills a new Q table with preset values, with desirable actions marked as 1.
    :return: The filled Q table
    """
    # create empty Q table
    q_table = _init_q_table()

    # go forward
    forward_front_states = BasicWallFollower.FRONT_MEDIUM, BasicWallFollower.FRONT_FAR
    forward_front_right_state = BasicWallFollower.FRONT_RIGHT_FAR
    forward_right_state = BasicWallFollower.RIGHT_MEDIUM
    forward_left_state = BasicWallFollower.LEFT_FAR
    for state in forward_front_states:
        # create state key
        state_key = create_state_key((state, forward_front_right_state, forward_right_state, forward_left_state))

        # set value togo forward
        action = BasicWallFollower.FORWARD
        q_table[state_key][action] = 1

    # turn left if front-right and front are close
    tl_front_state = BasicWallFollower.FRONT_CLOSE
    tl_front_right_state = BasicWallFollower.FRONT_RIGHT_CLOSE
    for right_state in BasicWallFollower.RIGHT_STATES:
        for left_state in BasicWallFollower.LEFT_STATES:
            # create state key
            state_key = create_state_key((tl_front_state, tl_front_right_state, right_state, left_state))

            # set value to turn left
            action = BasicWallFollower.TURN_LEFT
            q_table[state_key][action] = 1

    # turn left if right is close and front is close or med
    tl_right_states = [BasicWallFollower.RIGHT_CLOSE, BasicWallFollower.RIGHT_VERY_CLOSE]
    tl_front_states = [BasicWallFollower.FRONT_CLOSE, BasicWallFollower.FRONT_MEDIUM]
    for tl_right_state in tl_right_states:
        for tl_front_state in tl_front_states:
            for front_right_state in BasicWallFollower.FRONT_RIGHT_STATES:
                for left_state in BasicWallFollower.LEFT_STATES:
                    # create state key
                    state_key = create_state_key((tl_front_state, front_right_state, tl_right_state, left_state))

                    # set value to turn left
                    action = BasicWallFollower.TURN_LEFT
                    q_table[state_key][action] = 1

    # turn right if left is close, right and front-right are far, and front is close or med
    tl_left_state = BasicWallFollower.LEFT_CLOSE
    tl_right_states = [BasicWallFollower.RIGHT_FAR, BasicWallFollower.RIGHT_VERY_FAR]
    tl_front_right_state = BasicWallFollower.FRONT_RIGHT_FAR
    for tl_front_state in tl_front_states:
        for tl_right_state in tl_right_states:
            # create state key
            state_key = create_state_key((tl_front_state, tl_front_right_state, tl_right_state, tl_left_state))

            # set value to turn right
            action = BasicWallFollower.TURN_RIGHT
            q_table[state_key][action] = 1

    # # turn right if right is very far
    # tl_right_state = BasicWallFollower.RIGHT_VERY_FAR
    # tl_front_states = [BasicWallFollower.FRONT_MEDIUM, BasicWallFollower.FRONT_FAR]
    # for tl_front_state in tl_front_states:
    #     for left_state in BasicWallFollower.LEFT_STATES:
    #         # create state key
    #         state_key = create_state_key((tl_front_state, tl_front_right_state, tl_right_state, left_state))
    #
    #         # set value to turn right
    #         action = BasicWallFollower.TURN_RIGHT
    #         q_table[state_key][action] = 1

    # return Q table
    return q_table


def _calculate_reward(state):
    """
    Determines the reward for entering a certain state.
    :param state: The state to calculate a reward for
    :return: The calculated reward
    """
    # set default reward
    reward = 0.0

    # penalize undesired states
    if state[0] is BasicWallFollower.FRONT_VERY_CLOSE \
            or state[2] is BasicWallFollower.RIGHT_VERY_CLOSE \
            or state[2] is BasicWallFollower.RIGHT_VERY_FAR \
            or state[3] is BasicWallFollower.LEFT_CLOSE:
        reward = -1.0
    # reward desired state
    elif state[2] is BasicWallFollower.RIGHT_MEDIUM:
        reward = 1.0

    # return reward
    return reward


def _do_random_action(e: float = 0.5):
    """
    Helper function. Determines whether to randomly select an action or to use the Q table.
    :param e: The threshold for choosing randomly; defaults to 0.5
     :return: ``True`` if a random action should be selected, ``False`` otherwise
    """
    return random.random() < e


def _choose_random_action():
    """
    Helper function. Chooses a random action for the agent to undertake.
    :return: The key of the randomly selected action
    """
    return random.choice(BasicWallFollower.VALID_ACTIONS)


def create_state_key(state_values):
    """
    Creates a state key for a Q-table dictionary using the given states.
    :param state_values: The list of states to use
    :return: The created state key
    """
    # create empty state key
    state_key = ""

    # fill state key
    for i in range(len(state_values)):
        # add state value to state key
        state_key += state_values[i]

        # add spacer between values
        if i < len(state_values) - 1:
            state_key += "_"

    # return state key
    return state_key


def main(q_path: str = None,
         qi_path: str = None,
         crg_path: str = None,
         rcg_path: str = None,
         lcg_path: str = None,
         fcg_path: str = None,
         preset: bool = False,
         verbose: bool = False,
         test: bool = False,
         fsp: int = None):
    """
    Main function. Handles node creation and execution.
    :param q_path: The path to use for loading and saving the latest Q-table
    :param qi_path: The path to use for saving Q-table instances
    :param crg_path: The path to use for loading and saving CR graph data
    :param rcg_path: The path to use for loading and saving RC graph data
    :param lcg_path: The path to use for loading and saving LC graph data
    :param fcg_path: The path to use for loading and saving FC graph data
    :param preset: Whether to use preset Q table (task 1)
    :param verbose: Whether the node should log additional information
    :param test: Whether the node should run in test mode (static Q table, no random choices)
    :param fsp: Whether to use a fixed starting pose and, if so
    """
    # create & execute node
    wall_follower = BasicWallFollower(q_table_path=q_path,
                                      qi_table_path=qi_path,
                                      cr_graph_path=crg_path,
                                      rc_graph_path=rcg_path,
                                      lc_graph_path=lcg_path,
                                      fc_graph_path=fcg_path,
                                      use_preset=preset,
                                      verbose=verbose,
                                      test=test,
                                      fsp=fsp)
    wall_follower.run()

    # kill node when done
    del wall_follower


# run on exec
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("__log", default=None, help="catcher for ROS backend")
    parser.add_argument("__name", default=None, help="catcher for ROS backend")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether to log additional information")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Whether to run in test mode (only use Q table & no updates)")
    parser.add_argument("--preset", action="store_true",
                        help="Whether to use preset Q-table; will override path & graph setting")
    parser.add_argument("-qp", "--q_path", default=None, type=str,
                        help="Path to use for latest Q-table loading/dumping; will dump to a hash-distinguished file "
                             "if none specified")
    parser.add_argument("-qip", "--qi_path", default=None, type=str,
                        help="Path to use for Q-table instance dumping; will dump to hash-distinguished files if none"
                             "specified")
    parser.add_argument("-crg", "--crg_path", default=None, type=str,
                        help="Path to use for cumulative reward graph loading/dumping; will dump to a"
                             " hash-distinguished file if none specified")
    parser.add_argument("-rcg", "--rcg_path", default=None, type=str,
                        help="Path to use for RC learning graph loading/dumping; will dump to a hash-distinguished "
                             "file if none specified")
    parser.add_argument("-lcg", "--lcg_path", default=None, type=str,
                        help="Path to use for LC learning graph loading/dumping; will dump to a hash-distinguished "
                             "file if none specified")
    parser.add_argument("-fcg", "--fcg_path", default=None, type=str,
                        help="Path to use for FC learning graph loading/dumping; will dump to a hash-distinguished "
                             "file if none specified")
    parser.add_argument("-fsp", "--fixed_starting_point", default=-1, type=int,
                        help="Index of starting point to use instead of random selection per episode; defaults to"
                             "-1 for none (use random); will be ignored if test mode is not enabled")
    args = parser.parse_args()

    # make sure fixed starting point is valid (in proper range) if set
    if args.fixed_starting_point != -1:
        if args.fixed_starting_point not in range(0, 8):
            raise IndexError("Invalid starting point index %d provided - must be in range [0,8)"
                             % args.fixed_starting_point)
        else:
            # set index as parameter
            fixed_starting_point = args.fixed_starting_point
    else:
        # -1 is none
        fixed_starting_point = None

    # pass to main
    main(args.q_path, args.qi_path, args.crg_path, args.rcg_path, args.lcg_path, args.fcg_path, args.preset,
         args.verbose, args.test, fixed_starting_point)
