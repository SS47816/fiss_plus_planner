import math
import state

def update_next_kinematic_state(state, a, delta, lf, lr, dt):
    x = state.x
    y = state.y
    psi = state.psi
    v = state.v

    beta = math.atan((lr / (lr + lf)) * math.tan(delta))

    next_state = state.State(0,0,0,0)
    next_state.x = x + v * math.cos(psi * beta) * dt
    next_state.y = y + v * math.sin(psi * beta) * dt
    next_state.psi = psi + (v / lf) * math.sin(beta) * dt
    next_state.v = v + a * dt

    return next_state