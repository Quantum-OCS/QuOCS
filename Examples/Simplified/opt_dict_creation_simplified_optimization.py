# give the optimization a name
optimization_client_name = "some_name"

# create the optimization_dictionary and add the name to it
optimization_dictionary = {"optimization_client_name": optimization_client_name}

# Total number of dCRAB superiteration
super_iteration_number = 10

# Maximum number of iteration per super-iteration
# (in max_eval_total: maximale anzahl von evaluation steps gesamt)
maximum_function_evaluations_number = 300

# algorithm settings
alg_settings_dict = {"algorithm_name": "dCRAB",
                     "super_iteration_number": super_iteration_number,
                     # "max_eval_total": super_iteration_number * maximum_function_evaluations_number,
                     "FoM_goal": 0.00001,
                     "optimization_direction": "minimization",
                     "total_time_lim": 7 * 60,
                     "compensate_drift": {
                         "compensate_after_SI": True,
                         "compensate_after_minutes": 15
                     },
                     "random_number_generator": {
                         "seed_number": 1337
                     },
                     # "re_evaluation": {
                     #     "re_evaluation_steps": [0.3, 0.5, 0.51]
                     # }
                     }

# add alg_settings to opt_dict
optimization_dictionary['algorithm_settings'] = alg_settings_dict

# direct search method (dsm) settings
dsm_settings_dict = {'general_settings':
                         {"dsm_algorithm_name": "NelderMead",
                          "is_adaptive": True
                          },
                     'stopping_criteria': {
                         # "max_eval": 100,
                         # "time_lim": 10000
                         # "xatol": 1e-10,
                         "change_based_stop": {
                             "cbs_funct_evals": 100,
                             "cbs_change": 0.02
                         }
                     }
                     }

# and add it to the opt_dict
optimization_dictionary['algorithm_settings']["dsm_settings"] = dsm_settings_dict

# We can define here how many times we need to use in the optimization
time_pi = {'time_name': 'time_pi', 'initial_value': 61e-9}
time_pi2 = {'time_name': 'time_pi2', 'initial_value': 31e-9}

# and add them to the opt_dict
optimization_dictionary["times"] = [time_pi, time_pi2]

# parameters
optimization_dictionary['parameters'] = []


######################################################################################################
# Pulse parameters
######################################################################################################

# number of basis vectors per pulse
vec_number = 2

# maximum number of oscillations
upper_freq_lim = 10

# G1_x
pulse_G1x = {'pulse_name': 'G1x',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi2',
             'amplitude_variation': 0.0318 / 100 * 200,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.0318 + 0.0*t'}
             }

# G1_y
pulse_G1y = {'pulse_name': 'G1y',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi2',
             'amplitude_variation': 0.0318 / 100 * 200,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.0*t'}
             }

# G2_x
pulse_G2x = {'pulse_name': 'G2x',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi2',
             'amplitude_variation': 0.0318 / 100 * 200,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.0*t'}
             }

# G2_y
pulse_G2y = {'pulse_name': 'G2y',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi2',
             'amplitude_variation': 0.0318 / 100 * 200,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.0318 + 0.0*t'}
             }

# G3_x
pulse_G3x = {'pulse_name': 'G3x',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi',
             'amplitude_variation': 0.3,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.11 + 0.0*t'}
             }

# G3_y
pulse_G3y = {'pulse_name': 'G3y',
             'upper_limit': 0.5,
             'lower_limit': -0.5,
             'bins_number': 1001,
             'time_name': 'time_pi',
             'amplitude_variation': 0.3,  # added to guess amp, to create 3 points for start simplex
             'basis': {'basis_name': 'Fourier',
                       'basis_class': 'Fourier',
                       'basis_module': 'quocslib.pulses.basis.Fourier',
                       'basis_vector_number': vec_number,  # number of frequencies within the below defined range
                       'random_super_parameter_distribution':
                           {'distribution_name': 'Uniform', 'distribution_class': 'Uniform',
                            'distribution_module': 'quocslib.pulses.superparameter.Uniform',
                            'lower_limit': 0.0, 'upper_limit': upper_freq_lim}  # number of oscillations within pulse
                       },
             'scaling_function': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 1.0 + 0.0*t'},
             'initial_guess': {'function_type': 'lambda_function', 'lambda_function': 'lambda t: 0.0*t'}
             }

# create a list containing the pulses
optimization_dictionary['pulses'] = [pulse_G3x, pulse_G3y]
