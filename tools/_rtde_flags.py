#!/usr/bin/env python3
from rtde_control import RTDEControlInterface
import inspect

attrs = [a for a in dir(RTDEControlInterface) if "FLAG" in a]
print("flags:")
for a in sorted(attrs):
    print(f"  {a} = {getattr(RTDEControlInterface, a)}")
print("init:", inspect.signature(RTDEControlInterface.__init__))
