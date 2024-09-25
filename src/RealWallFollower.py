#!/usr/bin/env python3

"""
ROS node for wall following using Q-learning.
"""

# imports
import rospy
import math
import pickle
import argparse
from pathlib import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


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
                 region_size: float = 30.0,
                 q_table_path: str = None,
                 verbose: bool = False):
        """
        ROS node which uses laser scan data with a Q-table to follow a right wall.
        :param learning_rate: The learning rate (alpha) to use in Q-table updating
        :param discount_factor: The discount factor (gamma) to use in Q-table updating
        :param region_size: Angle to use for arc region size
        :param q_table_path: The path to use for loading/dumping the latest Q-table
        :param verbose: Whether the node should log additional information
        """
        # store arc region size
        self.region_size = region_size

        # save verbose setting
        self.verbose = verbose

        # save learning rate, discount factor and e cutoff
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # save paths for later use
        if q_table_path is not None:  # latest Q table
            self.q_table_path = Path(q_table_path)
        else:
            # create dump path using hash
            new_q_path = f"q_tables/{hash(self)}_q_table.pkl"
            if self.verbose:
                rospy.logwarn(f"No path for latest Q-table provided, using hash path " + new_q_path)
            self.q_table_path = Path(new_q_path)

        # initialize node
        rospy.init_node('wall_follower')

        # try to load Q-table
        self.q_table = {}
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

            # abort if not loaded
            if not loaded:
                raise IOError("Could not load Q table when a non-blank Q table is required for testing, aborting; "
                              "if q_table_path was set, there will likely be an error message above this exception "
                              "that details why the Q table was unable to load")

        # create laser scan listener holder
        self.scanner = None

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
        self.good_iteration_counter = 0

        # log node creation
        rospy.loginfo("wall_follower node created")

    def __del__(self):
        # log shutdown
        rospy.loginfo("Shutting down wall follower node")

        # kill laser scan listener
        if self.scanner is not None:
            self.scanner.unregister()

        # kill action publisher
        self.action_publisher.unregister()

        # shutdown node
        rospy.signal_shutdown("Node terminated by code")

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
        if self.front < 0.2:
            front_state = BasicWallFollower.FRONT_VERY_CLOSE
        elif self.front < 0.3:
            front_state = BasicWallFollower.FRONT_CLOSE
        elif self.front <= 0.8:
            front_state = BasicWallFollower.FRONT_MEDIUM
        else:
            front_state = BasicWallFollower.FRONT_FAR

        # front-right state
        if self.front_right <= 0.8:
            front_right_state = BasicWallFollower.FRONT_RIGHT_CLOSE
        else:
            front_right_state = BasicWallFollower.FRONT_RIGHT_FAR

        # right state
        if self.right < 0.1:
            right_state = BasicWallFollower.RIGHT_VERY_CLOSE
        elif self.right < 0.2:
            right_state = BasicWallFollower.RIGHT_CLOSE
        elif self.right < 0.3:
            right_state = BasicWallFollower.RIGHT_MEDIUM
        elif self.right <= 0.8:
            right_state = BasicWallFollower.RIGHT_FAR
        else:
            right_state = BasicWallFollower.RIGHT_VERY_FAR

        # back-right state
        if self.back_right <= 0.8:
            back_right_state = BasicWallFollower.BACK_RIGHT_CLOSE
        else:
            back_right_state = BasicWallFollower.BACK_RIGHT_FAR

        # left state
        if self.left <= 0.2:
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
            # get range; silently discard any remaining invalids
            s_range = self.raw_state[0][i]
            if s_range <= 0:
                continue

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
        self.back_right = br_min
        self.left = l_min

    def _get_angle_region(self, angle):
        """
        Determines if an angle falls within one of the hardcoded angle regions of the algorithm.
        :param angle: The angle to use for region determination
        :return: The abbreviation key of the angle region the angle is associated with (``'n'`` if none)
        """
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

    def _choose_manual_action(self, state):
        """
        Telescopic (manual) action selection for debugging purposes.

        To this end, it also displays relevant state information - namely the state key and the Q values for
        each action in that state. Input is provided via text in console.
        :param state: The state of the agent
        :return: The selected action
        """
        # get action scores
        action_scores = {}
        for action in BasicWallFollower.VALID_ACTIONS:
            action_scores[action] = self._q(state, action)
        print(f"State: {create_state_key(state)}")
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
            action_msg.linear.x = 0.15
        elif action == BasicWallFollower.TURN_LEFT:
            action_msg.linear.x = 0.05
            action_msg.angular.z = _to_radians(50)
        elif action == BasicWallFollower.TURN_RIGHT:
            action_msg.linear.x = 0.05
            action_msg.angular.z = _to_radians(-50)
        else:
            # invalid action, abort
            rospy.logerr("Invalid action code \'%s\'; aborting action performance" % action)
            return

        # publish action
        self.action_publisher.publish(action_msg)

    def load_q_table(self, path: Path):
        """
        Loads a Q-table from the specified Pickle file path.
        :param path: The file path to load the Q-table from (relative to current working directory)
        """
        with path.open("rb") as file:
            _, self.q_table = pickle.load(file)

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
            if ranges[i] is None:
                rospy.logwarn("Null range received, discarding")
                continue
            elif ranges[i] < min_range or ranges[i] > max_range or ranges[i] <= 0:
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
        if self.verbose:
            rospy.loginfo("Region values: "
                          f"f={self.front}, fr={self.front_right}, r={self.right}, "
                          f"br={self.back_right}, {self.left}")
        self._discretize()
        if self.verbose:
            rospy.loginfo("Current state: %s", self.state)

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

        # initialize scan listener
        self.scanner = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        # create counter for total iterations
        iteration_counter = 0

        # create counter for iterations per episode
        episode_iteration_counter = 0

        # main loop
        while not rospy.is_shutdown():
            if self.verbose:
                rospy.loginfo("Starting iteration #%d" % (episode_iteration_counter + 1))

            # get state
            if self.verbose:
                rospy.loginfo("Waiting for state update")
            while not self.state_updated and not rospy.is_shutdown():
                continue
            current_state = self.poll_state()
            self.state_updated = False
            if self.verbose:
                rospy.loginfo("Got current state")

            # determine action to take
            # action = self._choose_manual_action(current_state)  # teleoperation
            action = self._choose_action(current_state)
            if self.verbose:
                rospy.loginfo("Selected action key \'%s\' (Q-value %d)" % (action, self._q(current_state, action)))

            # perform action
            if self.verbose:
                rospy.loginfo("Performing action")
            self._perform_action(action)
            rospy.sleep(0.3)  # wait for action to be completed

            # update iteration counters
            iteration_counter += 1

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
         verbose: bool = False):
    """
    Main function. Handles node creation and execution.
    :param q_path: The path to use for loading and saving the latest Q-table
    :param verbose: Whether the node should log additional information
    """
    # create & execute node
    wall_follower = BasicWallFollower(q_table_path=q_path,
                                      verbose=verbose)
    wall_follower.run()

    # kill node when done
    del wall_follower


# run on exec
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("q_path", type=str,
                        help="Path to use for latest Q-table loading")
    parser.add_argument("__log", default=None, help="catcher for ROS backend")
    parser.add_argument("__name", default=None, help="catcher for ROS backend")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether to log additional information")
    args = parser.parse_args()

    # pass to main
    main(args.q_path, args.verbose)
