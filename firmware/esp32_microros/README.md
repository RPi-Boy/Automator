# ESP32 actuator firmware (Arduino IDE + micro_ros_arduino)

Open `esp32_actuator/` as a sketch (`esp32_actuator.ino` plus `pins.h`, `stepper_driver.h/.cpp` in the same folder).
**Not compiled or run on hardware yet** - written to be uploaded; expect to fix typos on first build.

## Arduino IDE setup
- Board package: **esp32 by Espressif 2.0.x** (the LEDC calls in `stepper_driver.cpp` are the 2.x API; 3.x renamed them)
- Libraries: **micro_ros_arduino** (humble branch), **AccelStepper** (Mike McCauley)
- Board: ESP32 Dev Module. Transport is UART0 at 115200 (`set_microros_transports()`), so nothing else may print to `Serial`.

## Jetson side
```bash
source /opt/ros/humble/setup.bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
```
Close the Arduino serial monitor first (it would hold the port). Check with `ros2 topic list` that the ESP32 node `esp32_actuator` appears.

## Interface
| Topic | Type | Direction | Meaning |
|---|---|---|---|
| `/joint_commands` | Float32MultiArray `[q1..q4 rad, gripper -1..+1]` | in | absolute joint targets @ ~50 Hz from `joint_trajectory_node` |
| `/arm_target` | Float32MultiArray `[x,y,z,class,uncertainty]` | in | safety gate: uncertainty > 0.75 halts |
| `/estop` | Bool | in | true halts |
| `/estop_reset` | Bool | in | true clears the latch (refused while the hardware line is still asserted) |

## Safety behaviour
1. **Hardware e-stop**: GPIO33, active LOW, real `attachInterrupt(FALLING)`. Wire a normally-closed button to GND, or a Jetson GPIO (open-drain).
2. **Software e-stop**: uncertainty > 0.75 or `/estop = true` call the same `haltMotors()` (EN pin HIGH = drivers off, LED on, latched).
3. **Watchdog**: no `/joint_commands` for 200 ms disables the drivers; the next command re-enables them. Drivers also stay disabled until the first command after boot or after a reset.
4. The ISR only does direct GPIO register writes (no `digitalWrite`, no printing).

Disabling the drivers removes holding torque: an arm held against gravity will sag on any stop. Add brakes or a gravity-safe pose if that matters on the real arm.

## Pins (see `pins.h`)
STEP 18/19/21/22, DIR 23/26/27/32, driver EN **25** (active LOW), e-stop in 33, LED 2, gripper servo 4. GPIO12 (flash-voltage strap) is deliberately unused.

## Before powering the arm
- `JOINT_CFG[].steps_per_rad` in `stepper_driver.cpp` are **placeholders** (200 steps x 16 microsteps x an assumed gear ratio). Measure each joint's real reduction.
- There are no limit switches or encoders: the firmware assumes the arm starts at q = 0 (straight up). Power-cycle only from that pose, or add homing.
- Test with motors unloaded first and the e-stop within reach.
