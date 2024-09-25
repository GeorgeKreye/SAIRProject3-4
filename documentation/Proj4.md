## New state space
Displayed next to each aspect value of the discretized state is the state key
component for that particular aspect value. Each aspect region is 30 degrees in
arc length, its center at the minimum of the arc. Ranges are in meters.

Front (center 0 deg.):
* Too close: range < 0.2 (``fvc``)
* Close: 0.5 <= range < 0.3 (``fc``)
* Medium: 0.6 <= range <= 0.8 (``fm``)
* Far: range > 0.8 (``ff``)

Front-Right (center 315 deg.):
* Close: range <= 0.8 (``frc``)
* Far: range > 0.8 (``frf``)

Right (center 270 deg.):
* Too close: range < 0.1 (``rvc``)
* Close: 0.1 <= range < 0.2 (``rc``)
* Medium: 0.2 <= range < 0.3 (``rm``)
* Far: 0.4 <= range <= 0.8 (``rf``)
* Too far: range > 0.8 (``rvf``)

Back-Right (center 225 deg.):
* Close: range <= 0.8 (``brc``)
* Far: range > 0.8 (``brf``)

Left (center 90 deg.):
* Close: range <= 0.2 (``lc``)
* Far: range > 0.2 (``lf``)

## Execution
Running the code for Project 4 is simple, simply run:<br />
``roscore`` (optional)<br />
``roslaunch sair-p3-georgekreye wallfollow.launch``<br />
``roslaunch sair-p3-georgekreye proj4.launch``