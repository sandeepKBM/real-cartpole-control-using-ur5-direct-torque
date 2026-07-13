#!/usr/bin/env python3
"""Dashboard helpers for URSim / real-UR bring-up."""

from __future__ import annotations

import socket


def dashboard_command(robot_ip: str, command: str, *, port: int = 29999, timeout_s: float = 8.0) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    sock.connect((robot_ip, port))
    sock.recv(4096)
    sock.sendall((command + "\n").encode())
    response = sock.recv(8192).decode(errors="replace").strip()
    sock.close()
    return response


def query_remote_control(robot_ip: str) -> bool:
    response = dashboard_command(robot_ip, "is in remote control").lower()
    return "true" in response


def clear_operational_mode(robot_ip: str) -> str:
    """Return operational mode control to PolyScope (undo dashboard lock)."""
    return dashboard_command(robot_ip, "clear operational mode")


def power_on_and_release(robot_ip: str) -> dict[str, str]:
    commands = ["robotmode", "power on", "brake release", "robotmode", "is in remote control"]
    return {cmd: dashboard_command(robot_ip, cmd) for cmd in commands}
