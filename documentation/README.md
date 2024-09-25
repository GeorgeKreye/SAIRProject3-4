# SAIR Project 3
## Part 1
Preset Q-table that is good at following straight lines, but
does not work well with corners (which are out of scope of this
part anyway.)
### How to run
1. (Optional) Run ``roscore``
2. Run ``roslaunch sair-p3-georgekreye wallfollow.launch``
3. Run ``roslaunch sair-p3-georgekreye part1_demo.launch``
### Discretization
Displayed next to each aspect value of the discretized state is the state key
component for that particular aspect value. Each aspect region is 30 degrees in
arc length, its center at the minimum of the arc.

Front (center 0 deg.):
* Too close: range < 0.5 (``fvc``)
* Close: 0.5 <= range < 0.6 (``fc``)
* Medium: 0.6 <= range <= 1.2 (``fm``)
* Far: range > 1.2 (``ff``)

Front-Right (center 315 deg.):
* Close: range <= 1.2 (``frc``)
* Far: range > 1.2 (``frf``)

Right (center 270 deg.):
* Too close: range < 0.1 (``rvc``)
* Close: 0.1 <= range < 0.2 (``rc``)
* Medium: 0.2 <= range < 0.4 (``rm``)
* Far: 0.4 <= range <= 0.8 (``rf``)
* Too far: range > 0.8 (``rvf``)

Left (center 90 deg.):
* Close: range <= 0.5 (``lc``)
* Far: range > 0.5 (``lf``)
### Q-table
Part 1 uses a preset Q-table that has its construction hardcoded
in the method. There is no need to load a Q-table.
## Part 2
Part 2 has multiple .launch files for the stages of Q-Learing: one
for training, two for testing (one with random starting poses and
one with fixed starting poses).

Discretization remains the same, so that section will be omitted.
### How to run
####
Before running any training or testing launch files, do the following:
1. (Optional) Run ``roscore``
2. Run ``roslaunch sair-p3-georgekreye wallfollow.launch``

This will initialize the simulation.
#### Training
If starting training from scratch:
1. (Required when using defaults) Ensure there are no files occupying
the relative paths q_tables/part_2_q_table.pkl,
graphs/part_2_rc_learning_graph.pkl, graphs/part_2_lc_learning_graph.pkl,
and graphs/part_2_fc_learning_graph.pkl. If these are present, the default
launch settings will make the implementation assume a continuation of 
training.
2. Run ``roslaunch sair-p3-georgekreye part2_train.launch``. Optional
parameters are ``q_path``, which sets the file path to dump the latest Q-table
save to (defaults to q_tables/part2_q_table.pkl); ``qi_path``, which sets the
file path beginning to use in creating file paths (appending the episode \# of
the save and the .pkl extension) to dump episodic Q-table saves to
(defaults to q_tables/instances/part2_q_table); ``rcg_path``, which sets the
file path used to dump graph data for learning convergence in the right-close
scenario

If resuming training, do the same as the above, but skip step 1 if using
default parameters. The implementation will automatically recognize that
it should continue training instead of overwriting with new training, and
load the data from all parameters excluding ``qi_path``.
#### Reading a Q table
To print a Q table in a human-readable format, run <br />
``src/data_reading/read_q_table.py <q_table>``<br /> (using ``rosrun`` is not
needed). This will print out the Q-table in the console, displaying the
episode it was generated in as well as the Q-values for each state-action
pair. Each row is for a specific state (displayed in the format 
"&lt;f&gt;&#95;&lt;fr&gt;&#95;&lt;r&gt;&#95;&lt;br&gt;&#95;&lt;l&gt;", where
each aspect encased in &lt;&gt; is replaced with a state key component valid 
for that aspect - for example, "ff_frf_rvf_brf_lf"), which contains the list
of actions' Q values for that state.

#### Graphing
To plot a graph using a given data file, run <br/>
``src/data_reading/plot_graph.py <graph>``<br />  (using ``rosrun`` is not
needed). This will use ``matplotlib`` and ``numpy`` to create a line
plot using the given graph data. This can be run mid-execution of training to
show learning progress, but will need to be closed and reran to be updated.

To read graph data without plotting it visually, run <br />
``src/data_reading/read_graph.py <graph>`` <br /> (using ``rosrun`` is not
needed). This will print out the graph data in a human-readable format. This
can also be run mid-training, since it can be used to display the latest
correct choice ratio; however, it will still need to be manually closed and
reran to be kept current.
#### Finding best Q table
The Q table with the best correct choice ratio, either overall or for a
specific scenario, can be determined automatically using the script
``src/data_reading/find_best_episode.py``. However, this script only works
with using default parameters in training due to needing to iterate through
all Q-table instances saved. Non-default parameters (such a different instance
filepath format) will require manual combing of data or a separate script not
included in this implementation.

The overall best Q table in particular is determined by taking an average of
the correct choice ratios for all checked scenarios for each episode - the
highest average is considered the best overall and should be a reliable policy
for testing. To allow the user to ensure the latter is true, the best overall
Q table has its individual correct choice ratios presented to be checked against
the best Q table for each checked scenario; a large difference between the 
correct choice ratio for a scenario for the best table overall and the best table
for that scenario may suggest a need for additional training to produce a better
overall best Q table.

The cumulative reward is not used due to its behavior not being reliably
indicative of learning taking place.

#### Testing
Testing requires a Q-table file to load; if none is provided or loading fails,
node initialization will abort with an ``IOError``. The launch files for testing
will always attempt to load a Q table for this reason; it is highly recommended
to use them instead of attempting to manually run the node in testing mode
using ``rosrun``.

To test with random starting poses, run: <br /> 
``roslaunch sair-p3-georgekreye part2_test.launch``

To test with a fixed starting pose, run: <br />
``roslaunch sair-p3-georgekreye part2_test.launch starting_pos_index:=<i>`` <br />
with ``<i>`` being an integer being the index of one of the 8 hardcoded starting
positions (in the range [0,7]). 

With default options, both launch files will attempt to load a Q table at
q_tables/q_table_best.pkl; this can be changed by setting the ``q_path``
parameter. q_table_best.pkl is meant to be set to the best Q table found using the
script method detailed in the previous section.
### Q table
The Q table for part 2 is generated via training. When the training launch file
is run without a preexisting Q-table, it will initialize a new one with all Q
values being set to zero. These Q values will be altered during training.

The Q table is constructed out of nested dictionaries, with two layers being
present. The first layer handles states, while the second layer handles actions.
The keys for each layer's dictionary are strings, with the state key being of the
format "&lt;f&gt;&#95;&lt;fr&gt;&#95;&lt;r&gt;&#95;&lt;br&gt;&#95;&lt;l&gt;" with each aspect
placeholder being replaced with an abbreviation representing an aspect of the state
for that region of discretization, and the action key being an abbreviation of one of
the three actions ("gf", "tr", or "tl").

### Q learning parameters
This implementation of Q learning utilizes epsilon greedy learning and a learning
rate in addition to the discount function.

The learning rate (alpha) of the algorithm is set to 0.2, while the discount factor
(gamma) is set to 0.8. The threshold (epsilon) for selecting greedy choice over
random choice is dynamic - it starts at 0.9, and decreases to 0.1 over the course
of 600 episodes of training (though training is allowed to continue after this point). 

All three parameters are ignored in testing mode, due to alpha and gamma being
irrelevant and greedy choice always being selected while testing.
