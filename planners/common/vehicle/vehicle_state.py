from common.geometry.math_utils import limitWithinRange

class VehicleState(object):
    def __init__(self, x, y, yaw, speed):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = speed
        
class ActuatorState(object):
    def __init__(self, max_accel, max_decel, max_angle):
        self.accel = 0
        self.brake = 0
        self.angle = 0
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_angle = max_angle

    def setAccel(self,a):
        self.accel = min(max(0.0,a), self.max_accel)
        self.brake = min(abs(min(0.0,a)), self.max_decel)

    def setAngle(self, delta):
        self.angle = limitWithinRange(delta, -self.max_angle, self.max_angle)

# test
# ac = ActuatorState(5,5,10)
# ac.setAccel(-6)
# print(f"ac.accel={ac.accel}")
# print(f"ac.brake={ac.brake}")
# ac.setAccel(6)
# print(f"ac.accel={ac.accel}")
# print(f"ac.brake={ac.brake}")
# ac.setAngle(11)
# print(f"angle={ac.angle}")