"""Failsafe Module

Engineering safety and disaster recovery mechanisms.
"""
from src.failsafe.fallback_selector import FallbackSelector
from src.failsafe.veto_switch import VetoSwitch

__all__ = ["FallbackSelector", "VetoSwitch"]
