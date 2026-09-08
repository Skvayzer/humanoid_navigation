"""ROS 2 Foxy executor compatibility used only by the G1 rosbridge runtime.

rosbridge creates and destroys service clients as WebSocket sessions reconnect.
Foxy's mutually-exclusive callback group asserts if the executor observes an
entity in the tiny interval after it was removed from the group.  Newer ROS 2
releases tolerate this lifecycle.  Returning False for a removed entity keeps
the old executor alive until its node removes that entity from the wait set.
"""

import weakref

try:
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
except ImportError:
    MutuallyExclusiveCallbackGroup = None


if MutuallyExclusiveCallbackGroup is not None:
    def _safe_can_execute(self, entity):
        with self._lock:
            if weakref.ref(entity) not in self.entities:
                return False
            return self._active_entity is None


    def _safe_beginning_execution(self, entity):
        with self._lock:
            if weakref.ref(entity) not in self.entities:
                return False
            if self._active_entity is None:
                self._active_entity = entity
                return True
        return False


    def _safe_ending_execution(self, entity):
        with self._lock:
            if self._active_entity == entity:
                self._active_entity = None


    MutuallyExclusiveCallbackGroup.can_execute = _safe_can_execute
    MutuallyExclusiveCallbackGroup.beginning_execution = _safe_beginning_execution
    MutuallyExclusiveCallbackGroup.ending_execution = _safe_ending_execution
