

class PID(object):
    def __init__(self, dt, max, min, Kp, Kd, Ki):
        self.dt_ = dt
        self.max_ = max
        self.min_ = min
        self.Kp_ = Kp
        self.Kd_ = Kd
        self.Ki_ = Ki
        self.pre_error_ = 0
        self.integral_ = 0

    def setSampleTime(self, dt_):
        self.dt_ = dt_

    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.pre_error_ = 0.0
        self.output = 0.0

    def calculate(self, setpoint, pv):
        # Calculate error
        self.error = setpoint - pv

        # Proportional term
        self.P_term = self.Kp_ * self.error

        # Integral term
        self.integral_ += self.error * self.dt
        self.I_term = self.Ki_ * self.integral_

        # Derivative term
        self.derivative = (self.error - self.pre_error_) / self.dt_
        self.D_term = self.Kd_ * self.derivative
       
        # Calculate total output
        self.output = self.P_term + self.I_term + self.D_term
       
        # Restrict to max/min
        self.output = min(self.output, self.max_)
        self.output = max(self.output, self.min_)

        # save error to previous error
        self.pre_error_ = self.error

        return self.output