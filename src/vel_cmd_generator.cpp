#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>
#include <string>

using namespace std::literals::chrono_literals;
using std::placeholders::_1;

#define PI 3.141592653589793

enum ControlState {
  IDLE,
  MOVING,
  STOPPING
};


class VelCmdGenNode : public rclcpp::Node {
  public:
    VelCmdGenNode() : Node("vel_cmd_gen_node") {
      control_state = IDLE;
      joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&VelCmdGenNode::joint_states_callback, this, _1));
      velocity_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/forward_velocity_controller/commands", rclcpp::SystemDefaultsQoS());
      start_stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/start_stop", 10, std::bind(&VelCmdGenNode::start_stop_callback, this, _1));
      DH_a = {0.0, -0.425, -0.3922, 0.0, 0.0, 0.0};
      DH_d = {0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996};
      DH_alpha = {PI/2, 0.0, 0.0, PI/2, -PI/2, 0.0};
      timer_ = this->create_wall_timer(20ms, std::bind(&VelCmdGenNode::timer_callback, this));
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr velocity_cmd_pub_;
    std::array<double, 6> joint_angles;
    std::array<double, 6> DH_a;
    std::array<double, 6> DH_d;
    std::array<double, 6> DH_alpha;
    ControlState control_state;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_stop_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
      // RCLCPP_INFO(this->get_logger(), "Received joint states: %s %f", msg->name[0].c_str(), msg->position[0]);
      joint_angles[0] = msg->position[0];
      joint_angles[1] = msg->position[1];
      joint_angles[2] = msg->position[2];
      joint_angles[3] = msg->position[3];
      joint_angles[4] = msg->position[4];
      joint_angles[5] = msg->position[5];
    }

    void start_stop_callback(const std_msgs::msg::Bool::SharedPtr msg) {
      if (msg->data) {
        control_state = MOVING;
        RCLCPP_INFO(this->get_logger(), "Received start command, moving.");
      } else {
        control_state = STOPPING;
        RCLCPP_INFO(this->get_logger(), "Received stop command, stopping.");
      }
    }

    void timer_callback() {
      auto cmd_msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
      switch (control_state) {
        case MOVING:
          {
          std::array<double, 6> vel_cmds = compute_vel_cmds(joint_angles, {0.01, 0.0, 0.0, 0.0, 0.0, 0.0}); // m/s and rad/s
          cmd_msg->data.resize(vel_cmds.size());
          for (size_t i = 0; i < vel_cmds.size(); i++){
            cmd_msg->data[i] = vel_cmds[i];
          }
          velocity_cmd_pub_->publish(std::move(cmd_msg));
          }
          break;
        case STOPPING:
          cmd_msg->data.resize(6);
          for (size_t i = 0; i < 6; i++){
            cmd_msg->data[i] = 0.0;
          }
          velocity_cmd_pub_->publish(std::move(cmd_msg));
          control_state = IDLE;
          break;
        case IDLE:
          break;
      }
    }

    // computes the homogenious transformation matrix
    Eigen::MatrixXd HTM(double theta, double alpha, double a, double d) {
      Eigen::MatrixXd H {
        {std::cos(theta), -1*std::sin(theta)*std::cos(alpha), std::sin(theta)*std::sin(alpha), a*std::cos(theta)},
        {std::sin(theta), std::cos(theta)*std::cos(alpha), -1*std::cos(theta)*std::sin(alpha), a*std::sin(theta)},
        {0, std::sin(alpha), std::cos(alpha), d},
        {0, 0, 0, 1}
      };

      return H;
    }

    std::array<double, 6> compute_vel_cmds(const std::array<double, 6>& joint_positions, const std::vector<double>& desired_ee_vel) {
      std::array<double, 6> vel_cmds;
      
      // compute jacobian, then nedded joint velocities based on desired end-effector velocity
      std::array<Eigen::MatrixXd, 6> HTMs;
      for (int i=0; i<6; i++) {
        HTMs[i] = HTM(joint_positions[i], DH_alpha[i], DH_a[i], DH_d[i]);
        std::cout << "HTM " << i << i+1 << ":\n" << HTMs[i] << "\n";
      }

      Eigen::MatrixXd H06 = HTMs[0] * HTMs[1] * HTMs[2] * HTMs[3] * HTMs[4] * HTMs[5];
      std::cout << "H06:\n" << H06 << "\n";

      Eigen::Vector3d d06 = H06.block<3,1>(0,3);
      std::cout << "d06:\n" << d06 << "\n";

      Eigen::MatrixXd J(6, 6);
      
      J.block<3, 1>(0, 0) = Eigen::Vector3d(0, 0, 1).cross(d06 - Eigen::Vector3d(0, 0, 0));
      J.block<3, 1>(3, 0) = Eigen::Vector3d(0, 0, 1);
      
      Eigen::MatrixXd H01 = HTMs[0];
      J.block<3, 1>(0, 1) = H01.block<3, 1>(0, 2).cross(d06 - H01.block<3, 1>(0, 3));
      J.block<3, 1>(3, 1) = H01.block<3, 1>(0, 2);

      Eigen::MatrixXd H02 = HTMs[0] * HTMs[1];
      J.block<3, 1>(0, 2) = H02.block<3, 1>(0, 2).cross(d06 - H02.block<3, 1>(0, 3));
      J.block<3, 1>(3, 2) = H02.block<3, 1>(0, 2);

      Eigen::MatrixXd H03 = HTMs[0] * HTMs[1] * HTMs[2];
      J.block<3, 1>(0, 3) = H03.block<3, 1>(0, 2).cross(d06 - H03.block<3, 1>(0, 3));
      J.block<3, 1>(3, 3) = H03.block<3, 1>(0, 2);

      Eigen::MatrixXd H04 = HTMs[0] * HTMs[1] * HTMs[2] * HTMs[3];
      J.block<3, 1>(0, 4) = H04.block<3, 1>(0, 2).cross(d06 - H04.block<3, 1>(0, 3));
      J.block<3, 1>(3, 4) = H04.block<3, 1>(0, 2);

      Eigen::MatrixXd H05 = HTMs[0] * HTMs[1] * HTMs[2] * HTMs[3] * HTMs[4];
      J.block<3, 1>(0, 5) = H05.block<3, 1>(0, 2).cross(d06 - H05.block<3, 1>(0, 3));
      J.block<3, 1>(3, 5) = H05.block<3, 1>(0, 2);

      std::cout << "Jacobian:\n" << J << "\n";

      Eigen::MatrixXd J_inv = J.inverse();
      std::cout << "Jacobian inverse:\n" << J_inv << "\n";

      Eigen::VectorXd desired_ee_vel_vec(6);
      desired_ee_vel_vec <<
        desired_ee_vel[0],
        desired_ee_vel[1],
        desired_ee_vel[2],
        desired_ee_vel[3],
        desired_ee_vel[4],
        desired_ee_vel[5];

      Eigen::VectorXd joint_velocities = J_inv * desired_ee_vel_vec;
      std::cout << "Joint velocities:\n" << joint_velocities << "\n";
      vel_cmds[0] = joint_velocities(0);
      vel_cmds[1] = joint_velocities(1);
      vel_cmds[2] = joint_velocities(2);
      vel_cmds[3] = joint_velocities(3);
      vel_cmds[4] = joint_velocities(4);
      vel_cmds[5] = joint_velocities(5);

      return vel_cmds;
    }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VelCmdGenNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
