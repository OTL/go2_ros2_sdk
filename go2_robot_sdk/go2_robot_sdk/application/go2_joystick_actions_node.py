# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""Joystick-driven WebRTC command publisher for the Go2 robot."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy

from go2_interfaces.msg import WebRtcReq
from ..domain.constants import ROBOT_CMD, RTC_TOPIC


@dataclass(frozen=True)
class CommandStep:
    """Represents a single WebRTC command to send."""
    name: str
    api_id: int
    parameter: str
    topic: str
    priority: int


class Go2JoystickActionsNode(Node):
    """Translate joystick button presses into WebRTC commands."""

    def __init__(self) -> None:
        super().__init__(
            'go2_joystick_actions_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self._command_interval = float(
            self.declare_parameter('command_interval', 0.25).value
        )
        if self._command_interval <= 0.0:
            self.get_logger().warn(
                'command_interval <= 0.0, defaulting to 0.1 s'
            )
            self._command_interval = 0.1

        self._output_topic = self.declare_parameter(
            'output_topic', '/webrtc_req'
        ).value
        self._default_topic = self.declare_parameter(
            'default_topic', RTC_TOPIC['SPORT_MOD']
        ).value
        self._default_priority = int(
            self.declare_parameter('default_priority', 0).value
        )

        self._publisher = self.create_publisher(WebRtcReq, self._output_topic, 10)
        self.create_subscription(Joy, 'joy', self._on_joy, 10)

        self._button_actions = self._load_button_actions()
        if not self._button_actions:
            self.get_logger().warn('No joystick actions configured; node is idle.')
        else:
            for button, steps in self._button_actions.items():
                command_names = ', '.join(step.name for step in steps)
                self.get_logger().info(
                    f'Button {button} mapped to commands: {command_names}'
                )

        self._previous_buttons: List[int] = []
        self._command_queue: Deque[CommandStep] = deque()
        self.create_timer(self._command_interval, self._process_queue)

    def _load_button_actions(self) -> Dict[int, List[CommandStep]]:
        """Reconstruct button mappings from ROS parameters."""
        actions: Dict[int, List[CommandStep]] = {}

        params = self.get_parameters_by_prefix('joystick_actions.buttons')
        button_ids = {
            int(key.split('.', 1)[0])
            for key in params.keys()
            if key and key.split('.', 1)[0].isdigit()
        }

        for button_idx in sorted(button_ids):
            steps = self._parse_button_config(button_idx)
            if steps:
                actions[button_idx] = steps

        return actions

    def _parse_button_config(self, button_idx: int) -> List[CommandStep]:
        prefix = f'joystick_actions.buttons.{button_idx}'
        params = self.get_parameters_by_prefix(prefix)

        topic = self._resolve_topic(params)
        priority = self._resolve_priority(params)
        command_names = self._resolve_command_names(params, button_idx)
        if not command_names:
            return []

        command_parameters, parameter_override = self._resolve_parameter_overrides(params)

        steps: List[CommandStep] = []
        for name in command_names:
            api_id = ROBOT_CMD.get(name)
            if api_id is None:
                self.get_logger().error(
                    f"Unknown command '{name}' configured for button {button_idx}."
                )
                continue

            parameter = command_parameters.get(name)
            if parameter is None:
                if parameter_override is not None and len(command_names) == 1:
                    parameter = parameter_override
                else:
                    parameter = str(api_id)

            steps.append(
                CommandStep(
                    name=name,
                    api_id=api_id,
                    parameter=parameter,
                    topic=topic,
                    priority=priority,
                )
            )

        return steps

    def _resolve_topic(self, params: Dict[str, Parameter]) -> str:
        topic_param = params.get('topic')
        topic = str(topic_param.value) if topic_param else self._default_topic
        return topic or self._default_topic

    def _resolve_priority(self, params: Dict[str, Parameter]) -> int:
        priority_param = params.get('priority')
        priority = (
            int(priority_param.value)
            if priority_param is not None
            else self._default_priority
        )
        return max(0, min(priority, 255))

    def _resolve_command_names(
        self,
        params: Dict[str, Parameter],
        button_idx: int,
    ) -> List[str]:
        sequence_param = params.get('sequence')
        sequence: List[str] = []
        if sequence_param is not None:
            sequence = [str(entry) for entry in list(sequence_param.value)]

        command_param = params.get('command')
        command_name = str(command_param.value) if command_param else ''

        if not sequence and not command_name:
            self.get_logger().warn(
                f'Button {button_idx} has no command or sequence configured.'
            )
            return []

        return sequence if sequence else [command_name]

    def _resolve_parameter_overrides(
        self, params: Dict[str, Parameter]
    ) -> Tuple[Dict[str, str], Optional[str]]:
        command_parameters: Dict[str, str] = {}
        parameter_override = None
        for key, parameter in params.items():
            if key.startswith('parameters.'):
                cmd = key.split('.', 1)[1]
                command_parameters[cmd] = str(parameter.value)
            elif key == 'parameter':
                parameter_override = str(parameter.value)

        return command_parameters, parameter_override

    def _on_joy(self, msg: Joy) -> None:
        if not self._button_actions:
            return

        buttons = list(msg.buttons)
        if not self._previous_buttons:
            self._previous_buttons = [0] * len(buttons)

        if len(buttons) > len(self._previous_buttons):
            self._previous_buttons.extend([0] * (len(buttons) - len(self._previous_buttons)))

        for idx, value in enumerate(buttons):
            prev = self._previous_buttons[idx] if idx < len(self._previous_buttons) else 0
            if value and not prev:
                self._handle_button_press(idx)

        self._previous_buttons = buttons

    def _handle_button_press(self, button_idx: int) -> None:
        steps = self._button_actions.get(button_idx)
        if not steps:
            self.get_logger().debug(f'Button {button_idx} pressed with no mapping.')
            return

        command_names = ', '.join(step.name for step in steps)
        self.get_logger().info(
            f'Enqueuing commands for button {button_idx}: {command_names}'
        )
        self._command_queue.extend(steps)

    def _process_queue(self) -> None:
        if not self._command_queue:
            return

        step = self._command_queue.popleft()
        msg = WebRtcReq()
        msg.id = 0
        msg.api_id = step.api_id
        msg.parameter = step.parameter
        msg.topic = step.topic
        msg.priority = step.priority
        self._publisher.publish(msg)
        self.get_logger().info(
            f"Published command {step.name} (api_id={step.api_id}) to {step.topic}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = Go2JoystickActionsNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
