# Trajectory class
[List of Documentation Files](menu.md)

Some notes to handle the Trajectory class for storing eye movement data.

[TOC]

## Kind

The idea of the Trajectory class it to save the type of the data (angle or pixel). So you know the kind of the data which is stored.
The class try to prevent accidental changes.

You can use ```get_trajectory(kind)``` to always be sure to get the correct kind of data.
If the original format of the data is not the same, it will be converted every time.
You can change the type permanently with `convert_to(kind)`.

The same is true for ```get_velocity(kind)```.

Be careful with ```xy``` or ```velocity``` property it will always return the stored type.

### Conversion

The conversion happens in `get_trajectory`.
The following types are available:

- centered
  `pixel`, `angle_rad`, `angle_deg`
- shifted in first quadrant (stimulus is center)
  `pixel_shifted`,  `pixel_image`(inverted y axis)

## Further Functionality

The trajectory class provides some more information like ```velocity``` and have multiple methods to manipulate your data.

Every change will be documented in property ```processing```

**List of important Methods**

| name | purpose |
| --- | --- |
|  ```convert_to(kind) ``` | will change type of stored data |
|  ```get_trajectory(kind) ``` | return trajectory in specific type |
|  ```get_velocity(kind) ``` |  return velocity of trajectory in specific type |
|  ```apply_filter ``` | modify data with filter |
|  ```interpolate ``` | do linear interpolation for NaN |
|  ```remove_duplicates ``` | remove unchanged values |
|  ```invert_y ```,  ```offset ```| self explaining |

## Multiple Trajectories
... can be collected in Trajectories-Class.

This class is itterable and returns arrays with data from items.

## Internal Structure

```mermaid
graph TD

TB(TrajectoryBase)
TB-->TK(TrajectoryKind)
TK-->TP(TrajectoryProcess)
TP-->TM(TrajectoryModify)
TM-->Tfov(TrajectoryFOV)
Tfov-->TC(TrajectoryConvert)
TC-->Tcal(TrajectoryCalculate)
Tcal-->TV(TrajectoryVelocity)

Tmore(TrajectoryMore)

T(Trajectory)
TV-->T
Tmore-->T
```
