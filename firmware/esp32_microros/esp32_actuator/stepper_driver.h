#pragma once
#include <Arduino.h>
#include "pins.h"

// Joint calibration. These are PLACEHOLDERS for a 1.8 deg stepper at 1/16 microstepping behind a
// gearbox: measure the real reduction of each joint and edit before powering the arm.
//   steps_per_rad = 200 * microsteps * gear_ratio / (2*pi)
struct JointConfig {
  float steps_per_rad;
  float max_speed_rad_s;   // from the URDF velocity limits
  float max_accel_rad_s2;
  float limit_lo_rad;      // from the URDF joint limits
  float limit_hi_rad;
};

extern const JointConfig JOINT_CFG[NUM_JOINTS];

// Non-blocking multi-axis trapezoidal-acceleration stepper controller (AccelStepper under the hood).
// Call update() as fast as possible from ONE task (the Arduino loop); setTarget() and the halt
// helpers may be called from any task or ISR.
class StepperDriver {
 public:
  void begin();
  void setTargetRad(const float q[NUM_JOINTS]);   // clamps to joint limits
  void update();                                   // runs the step generators
  void enableMotors();                             // drivers on (no-op while e-stop is latched)
  void disableMotors();                            // drivers off (coast)
  bool motorsEnabled() const { return enabled_; }
  float positionRad(int joint) const;
  void setHomeHere();                              // declare current pose as q = 0 (arm straight up)
 private:
  volatile bool enabled_ = false;
};

extern StepperDriver steppers;

// ---- gripper servo (LEDC PWM, 50 Hz) ----
void gripperBegin();
void gripperSet(float cmd);   // -1 = closed, +1 = open
