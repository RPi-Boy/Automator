#pragma once
// ESP32-WROOM-32 pin map. Avoided: GPIO0/2/5/12/15 (boot straps), 6-11 (flash), 1/3 (UART0 = micro-ROS).
// GPIO2 is only the on-board LED, which is safe as an output.
#define NUM_JOINTS 4

static const int STEP_PINS[NUM_JOINTS] = {18, 19, 21, 22};
static const int DIR_PINS[NUM_JOINTS]  = {23, 26, 27, 32};

#define MOTOR_ENABLE_PIN 25   // all drivers' EN, active LOW: HIGH = motors disabled
#define ESTOP_LED_PIN     2
#define ESTOP_IN_PIN     33   // hardware e-stop input, active LOW (Jetson GPIO or a button to GND)
#define GRIPPER_PIN       4   // hobby servo on an LEDC channel
