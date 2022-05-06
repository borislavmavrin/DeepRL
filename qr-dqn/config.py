class Config():

    # DQN paper hyperparamters
    max_number_of_steps = 5000000
    batch_size = 32
    buffer_size = 1000000
    target_network_update_frequency = 10000
    gamma = 0.99
    sgd_update_frequency = 4
    action_repeat = 1  # skip frames
    agent_history_length = 4

    # Adam optimizer parameters (taken from Rainbow)
    lr_start = 0.0000625
    eps = 0.00015

    epsilon_start = 1.
    epsilon_end = 0.02
    eps_nsteps = 1000000
    epsilon_step_size = (epsilon_end - epsilon_start) / float(eps_nsteps - 1)

    replay_start_size = 5e+4
    clip_by_norm = 1.

    # environment specific parameters
    game = "PongNoFrameskip-v4"
    model = "QR-DQN"
    number_of_runs = 3
    max_skip_buffer_shape = (210, 160, 3, 2)
    device = 'cuda'
    width = 84
    height = 84
    state_shape = (height, width, agent_history_length)
    grad_clip = True
    max_pool = False
    experience_queue_size = 1000

    # misc paramters
    result_print_freq = 1000
    local_buffer_size = batch_size + 1
    base_path = "/tmp/tf_logs/" + game + "/" + model + "/"
    log_path = None
    num_quantiles = 200
