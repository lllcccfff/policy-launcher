from rl_framework.ilqr.lqr_solver import ILQRSolverParameters, ILQRWarmStartParameters, ILQRSolver
import numpy as np

solver_params = ILQRSolverParameters(
    discretization_time=0.1,
    state_cost_diagonal_entries=[1.0, 8.0, 15.0, 2.0, 1.0],
    input_cost_diagonal_entries=[1.5, 5.0],
    state_trust_region_entries=[1.0, 1.0, 1.0, 1.0, 1.0],
    input_trust_region_entries=[1.0, 1.0],
    max_ilqr_iterations=100,
    convergence_threshold=1e-6,
    max_solve_time=0.1,
    max_acceleration=3.2,
    max_steering_angle=40 * np.pi / 180.0,
    max_steering_angle_rate=0.4,
    min_velocity_linearization=0.01,
    wheelbase=2.469
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.1,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

lqr = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)

def plan2control(plan_traj, init_state):
    current_state = init_state
    solutions = lqr.solve(current_state, plan_traj)
    optimal_inputs = solutions[-1].input_trajectory
    
    accel_cmd = optimal_inputs[0, 0]
    steering_rate_cmd = optimal_inputs[0, 1]

    current_steering_angle = init_state[4]
    dt = solver_params.discretization_time
    new_steering_angle = current_steering_angle + steering_rate_cmd * dt  # rad

    max_steer_rad = solver_params.max_steering_angle
    new_steer_rad = np.clip(new_steering_angle, -max_steer_rad, max_steer_rad)
    steer_action = new_steer_rad / max_steer_rad

    max_accel = solver_params.max_acceleration
    max_decel = solver_params.max_acceleration

    if accel_cmd >= 0.0:
        throttle = accel_cmd / max_accel      
        throttle_brake = float(np.clip(throttle, 0.0, 1.0))
    else:
        brake = -accel_cmd / max_decel 
        throttle_brake = float(-np.clip(brake, 0.0, 1.0)) 
    
    return throttle_brake, steer_action

if __name__ == '__main__':
    # plan_traj = np.zeros((6,5))
    # plan_traj[:, 0] = 1
    # plan_traj[:, 1] = np.ones(6)
    # plan_traj = np.cumsum(plan_traj, axis=0)
    # print(plan_traj)
    plan_traj = np.array([[-0.18724936,  2.29100776,  0.,          0.,          0.,        ],
                        [-0.29260731,  2.2971828 ,  0.,          0.,          0.        ],
                        [-0.46831554,  2.55596018,  0.,          0.,          0.        ],
                        [-0.5859955 ,  2.73183298,  0.,          0.,          0.        ],
                        [-0.62684   ,  2.84659386,  0.,          0.,          0.        ],
                        [-0.67761713,  2.80647802,  0.,          0.,          0.        ]])
    plan_traj = plan_traj[:, [1,0,2,3,4]]
    init_state = np.array([0.00000000e+00, 3.46944695e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    print(plan_traj.shape, init_state.shape)
    acc, steer = plan2control(plan_traj, init_state)
    print(acc, steer)
