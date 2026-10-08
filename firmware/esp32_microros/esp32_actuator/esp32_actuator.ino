/*
 * ESP32 micro-ROS actuator controller for the 4-DOF arm.
 *
 * Subscribes : /joint_commands (Float32MultiArray [q1..q4 rad, gripper -1..1])  from joint_trajectory_node
 *              /arm_target     (Float32MultiArray [x,y,z,class,uncertainty])    safety gate: u > 0.75 -> halt
 *              /estop          (Bool)    true -> halt
 *              /estop_reset    (Bool)    true -> clear the latch (refused while the hardware line is asserted)
 *
 * Safety layers (all end in haltMotors()):
 *   1. hardware e-stop input GPIO33 (active LOW) via a real GPIO interrupt
 *   2. uncertainty > 0.75 on /arm_target or /estop = true
 *   3. watchdog: no /joint_commands for 200 ms -> drivers disabled (re-enabled by the next command)
 *
 * Threading: micro-ROS runs in a FreeRTOS task on core 0; loop() (core 1) only generates steps.
 * micro-ROS owns Serial as its transport: do NOT print to Serial here.
 *
 * Arduino IDE setup: ESP32 board package 2.0.x (the LEDC calls in stepper_driver.cpp are the 2.x API),
 * libraries micro_ros_arduino (humble) and AccelStepper (Mike McCauley). Open this folder as a sketch.
 * Jetson side: ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
 */
#include <Arduino.h>
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/bool.h>
#include <rmw_microros/rmw_microros.h>
#include <soc/gpio_struct.h>
#include "pins.h"
#include "stepper_driver.h"

static const float    UNCERTAINTY_LIMIT = 0.75f;
static const uint32_t WATCHDOG_MS = 200;

// ---- shared state ----
static volatile bool     estop_latched = false;
static volatile uint32_t last_cmd_ms = 0;
static volatile bool     have_cmd = false;
static portMUX_TYPE      mux = portMUX_INITIALIZER_UNLOCKED;
static float             target_q[NUM_JOINTS] = {0, 0, 0, 0};
static volatile float    target_grip = 1.0f;
static volatile bool     target_dirty = false;

// Single halt routine shared by the ISR and the software paths: direct register writes only
// (no Arduino calls, no printing), and safe from any context.
static void IRAM_ATTR haltMotors() {
  GPIO.out_w1ts = (1UL << MOTOR_ENABLE_PIN);  // EN high = drivers off
  GPIO.out_w1ts = (1UL << ESTOP_LED_PIN);
  estop_latched = true;
}

static void IRAM_ATTR estopIsr() { haltMotors(); }

// ---- micro-ROS objects ----
static rcl_subscription_t sub_cmd, sub_target, sub_estop, sub_reset;
static std_msgs__msg__Float32MultiArray msg_cmd, msg_target;
static std_msgs__msg__Bool msg_estop, msg_reset;
static rclc_executor_t executor;
static rclc_support_t support;
static rcl_allocator_t allocator;
static rcl_node_t node;

static void cmdCb(const void* m) {
  const auto* msg = (const std_msgs__msg__Float32MultiArray*)m;
  if (msg->data.size < 5) return;
  for (int i = 0; i < 4; i++) if (isnan(msg->data.data[i])) return;
  portENTER_CRITICAL(&mux);
  for (int i = 0; i < NUM_JOINTS; i++) target_q[i] = msg->data.data[i];
  target_grip = msg->data.data[4];
  target_dirty = true;
  last_cmd_ms = millis();
  have_cmd = true;
  portEXIT_CRITICAL(&mux);
}

static void targetCb(const void* m) {
  const auto* msg = (const std_msgs__msg__Float32MultiArray*)m;
  if (msg->data.size >= 5 && msg->data.data[4] > UNCERTAINTY_LIMIT) haltMotors();
}

static void estopCb(const void* m) {
  if (((const std_msgs__msg__Bool*)m)->data) haltMotors();
}

static void resetCb(const void* m) {
  if (!((const std_msgs__msg__Bool*)m)->data) return;
  if (digitalRead(ESTOP_IN_PIN) == LOW) return;   // hardware line still asserted: refuse
  portENTER_CRITICAL(&mux);
  estop_latched = false;
  have_cmd = false;                               // require a fresh command before moving again
  portEXIT_CRITICAL(&mux);
  digitalWrite(ESTOP_LED_PIN, LOW);
}

// Subscription messages need their sequences preallocated, otherwise incoming data is dropped silently.
static float cmd_buf[8], target_buf[8];

static void initFloatArray(std_msgs__msg__Float32MultiArray* msg, float* buf) {
  msg->data.data = buf;
  msg->data.size = 0;
  msg->data.capacity = 8;
  msg->layout.dim.data = NULL;
  msg->layout.dim.size = 0;
  msg->layout.dim.capacity = 0;
  msg->layout.data_offset = 0;
}

static bool microRosInit() {
  allocator = rcl_get_default_allocator();
  if (rclc_support_init(&support, 0, NULL, &allocator) != RCL_RET_OK) return false;
  if (rclc_node_init_default(&node, "esp32_actuator", "", &support) != RCL_RET_OK) return false;

  initFloatArray(&msg_cmd, cmd_buf);
  initFloatArray(&msg_target, target_buf);

  rclc_subscription_init_default(&sub_cmd, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray), "/joint_commands");
  rclc_subscription_init_default(&sub_target, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray), "/arm_target");
  rclc_subscription_init_default(&sub_estop, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool), "/estop");
  rclc_subscription_init_default(&sub_reset, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool), "/estop_reset");

  rclc_executor_init(&executor, &support.context, 4, &allocator);
  rclc_executor_add_subscription(&executor, &sub_cmd, &msg_cmd, &cmdCb, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &sub_target, &msg_target, &targetCb, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &sub_estop, &msg_estop, &estopCb, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &sub_reset, &msg_reset, &resetCb, ON_NEW_DATA);
  return true;
}

static void microRosTask(void*) {
  for (;;) {
    // Wait for the agent: `micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200` on the Jetson.
    while (rmw_uros_ping_agent(100, 1) != RMW_RET_OK) delay(100);
    if (!microRosInit()) { delay(500); continue; }

    uint32_t last_ping = millis();
    for (;;) {
      rclc_executor_spin_some(&executor, RCL_MS_TO_NS(5));
      if (millis() - last_ping > 1000) {          // agent vanished -> tear down and reconnect
        last_ping = millis();
        if (rmw_uros_ping_agent(50, 1) != RMW_RET_OK) break;
      }
    }
    // Commands stop while the agent is gone, so the watchdog in loop() disables the drivers.
    rclc_executor_fini(&executor);
    rcl_subscription_fini(&sub_cmd, &node);
    rcl_subscription_fini(&sub_target, &node);
    rcl_subscription_fini(&sub_estop, &node);
    rcl_subscription_fini(&sub_reset, &node);
    rcl_node_fini(&node);
    rclc_support_fini(&support);
  }
}

void setup() {
  // Order matters: drivers must be disabled before anything else comes up.
  steppers.begin();
  pinMode(ESTOP_LED_PIN, OUTPUT);
  digitalWrite(ESTOP_LED_PIN, LOW);
  gripperBegin();

  pinMode(ESTOP_IN_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ESTOP_IN_PIN), estopIsr, FALLING);
  if (digitalRead(ESTOP_IN_PIN) == LOW) haltMotors();   // asserted at boot

  Serial.begin(115200);
  set_microros_transports();   // UART0 @ 115200, matches `micro_ros_agent serial -b 115200`
  xTaskCreatePinnedToCore(microRosTask, "uros", 16384, NULL, 5, NULL, 0);
}

void loop() {
  if (estop_latched) {
    steppers.disableMotors();
    digitalWrite(ESTOP_LED_PIN, HIGH);
    return;
  }

  bool have, fresh;
  float q[NUM_JOINTS], grip;
  portENTER_CRITICAL(&mux);
  have = have_cmd;
  // millis() is read inside the lock, and the difference is signed: a command stamped just after a
  // separate earlier read would otherwise underflow the unsigned subtraction and trip the watchdog.
  fresh = (int32_t)(millis() - last_cmd_ms) <= (int32_t)WATCHDOG_MS;
  for (int i = 0; i < NUM_JOINTS; i++) q[i] = target_q[i];
  grip = target_grip;
  bool dirty = target_dirty;
  target_dirty = false;
  portEXIT_CRITICAL(&mux);

  if (!have || !fresh) {            // never commanded, or watchdog timeout
    if (steppers.motorsEnabled()) steppers.disableMotors();
    return;
  }

  if (!steppers.motorsEnabled()) steppers.enableMotors();
  if (dirty) {
    steppers.setTargetRad(q);
    gripperSet(grip);
  }
  steppers.update();
}
