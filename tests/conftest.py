"""pytest session configuration.

Sets the matplotlib backend to Agg before any test module imports pyplot, so
that tests relying on figure creation (e.g., on_validation_epoch_end) work in
headless / CI environments without a display.
"""
import matplotlib

matplotlib.use("Agg")
