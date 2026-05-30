"""Pytest configuration for Nexus tests.

Note: Test-specific fixtures are handled via setUp() and tearDown() methods
in unittest.TestCase subclasses. This file is reserved for pytest plugins
and global configuration if needed in the future.
"""

import matplotlib

matplotlib.use("Agg")
