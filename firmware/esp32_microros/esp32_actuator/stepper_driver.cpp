#include "stepper_driver.h"
#include <AccelStepper.h>
#include <math.h>

static const float MICROSTEPS = 16.0f;
static const float STEPS_REV = 200.0f * MICROSTEPS;
static const float TWO_PI_F = 6.2831853f;

const JointConfig JOINT_CFG[NUM_JOINTS] = {
  // steps_per_rad,                 v,    a,    lo,       hi
  {STEPS_REV * 5.0f / TWO_PI_F,  2.0f, 4.0f, -3.14159f,  3.14159f},  // J1 yaw      (assumed 5:1)
  {STEPS_REV * 10.0f / TWO_PI_F, 2.0f, 4.0f, -1.5708f,   1.5708f},   // J2 shoulder (assumed 10:1)
  {STEPS_REV * 10.0f / TWO_PI_F, 2.5f, 5.0f, -2.0f,      2.0f},      // J3 elbow    (assumed 10:1)
  {STEPS_REV * 5.0f / TWO_PI_F,  3.0f, 6.0f, -1.5708f,   1.5708f},   // J4 wrist    (assumed 5:1)
};

static AccelStepper axis[NUM_JOINTS] = {
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[0], DIR_PINS[0]),
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[1], DIR_PINS[1]),
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[2], DIR_PINS[2]),
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[3], DIR_PINS[3]),
};

StepperDriver steppers;

void StepperDriver::begin() {
  pinMode(MOTOR_ENABLE_PIN, OUTPUT);
  digitalWrite(MOTOR_ENABLE_PIN, HIGH);  // boot with drivers disabled
  for (int i = 0; i < NUM_JOINTS; i++) {
    axis[i].setMaxSpeed(JOINT_CFG[i].max_speed_rad_s * JOINT_CFG[i].steps_per_rad);
    axis[i].setAcceleration(JOINT_CFG[i].max_accel_rad_s2 * JOINT_CFG[i].steps_per_rad);
    axis[i].setCurrentPosition(0);
  }
}

void StepperDriver::setTargetRad(const float q[NUM_JOINTS]) {
  for (int i = 0; i < NUM_JOINTS; i++) {
    float c = fminf(fmaxf(q[i], JOINT_CFG[i].limit_lo_rad), JOINT_CFG[i].limit_hi_rad);
    axis[i].moveTo((long)lroundf(c * JOINT_CFG[i].steps_per_rad));
  }
}

void StepperDriver::update() {
  if (!enabled_) return;               // no stepping (and no position drift in software) while disabled
  for (int i = 0; i < NUM_JOINTS; i++) axis[i].run();
}

void StepperDriver::enableMotors() { enabled_ = true; digitalWrite(MOTOR_ENABLE_PIN, LOW); }

void StepperDriver::disableMotors() {
  enabled_ = false;
  digitalWrite(MOTOR_ENABLE_PIN, HIGH);
  // Re-anchor targets on the current positions so re-enabling does not lurch toward a stale target.
  for (int i = 0; i < NUM_JOINTS; i++) axis[i].moveTo(axis[i].currentPosition());
}

float StepperDriver::positionRad(int j) const { return axis[j].currentPosition() / JOINT_CFG[j].steps_per_rad; }

void StepperDriver::setHomeHere() { for (int i = 0; i < NUM_JOINTS; i++) axis[i].setCurrentPosition(0); }

// ---- gripper ----
static const int GRIP_CH = 0;
static const int GRIP_FREQ = 50, GRIP_BITS = 14;
static const int GRIP_US_CLOSED = 1000, GRIP_US_OPEN = 2000;

void gripperBegin() {
  ledcSetup(GRIP_CH, GRIP_FREQ, GRIP_BITS);
  ledcAttachPin(GRIPPER_PIN, GRIP_CH);
  gripperSet(1.0f);
}

void gripperSet(float cmd) {
  cmd = fminf(fmaxf(cmd, -1.0f), 1.0f);
  float us = GRIP_US_CLOSED + (cmd + 1.0f) * 0.5f * (GRIP_US_OPEN - GRIP_US_CLOSED);
  ledcWrite(GRIP_CH, (uint32_t)(us / 20000.0f * ((1 << GRIP_BITS) - 1)));
}
