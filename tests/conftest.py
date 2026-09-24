"""Shared test configuration and fixtures."""

import os

from hypothesis import settings

# CI runs a fixed set of generated examples so that a failure reproduces on the next run. Local
# runs keep Hypothesis's default random exploration, which finds new cases over time.
settings.register_profile("ci", derandomize=True, database=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
