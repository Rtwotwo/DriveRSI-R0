"""Historical compatibility for retired PPO. Use native model adapters for new work."""
import sys
from driversi.training.legacy import ppo as _implementation
sys.modules[__name__] = _implementation
