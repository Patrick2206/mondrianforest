#!/usr/bin/env python

# This is the interface for the Mondrian Forest algorithm.
# Here the Model is trained and predictions are performed.

import numpy as np
import pickle
import time as t_ime
import resource
import json
import pprint as pp     # pretty printing module
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal, get_filename_mf, get_config
from mondrianforest import process_command_line, MondrianForest

# Start overall time tracking
time_0 = t_ime.clock()


def pre_processing():

    """
    Processing settings and data
    :return: required data for execution
    """

    # Process and print settings
    settings = process_command_line()
    print 'Current settings:'
    pp.pprint(vars(settings))

    # Resetting random seed
    reset_random_seed(settings)

    # Loading data
    data = load_data(settings)
    param, cache = precompute_minimal(data, settings)

    return settings, data, param, cache


def execute_mf():

    """
    Executing the algorithm. Train and predict stepwise.
    Tracking execution time for training and predicting.
    """

    # Get required data
    settings, data, param, cache = pre_processing()

    # Track time data for execution
    time_method_sans_init = 0.
    time_prediction = 0.

    # Get Mondrian Forest
    mf = MondrianForest(settings, data)

    print '\nminibatch\tmetric_test\tnum_leaves'

    start_pos = 0
    number_batches = settings.n_minibatches
    accuracy = []

    if settings.store_every:

            log_prob_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
            log_prob_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
            metric_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
            metric_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
            time_method_minibatch = np.inf * np.ones(settings.n_minibatches)
            forest_numleaves_minibatch = np.zeros(settings.n_minibatches)

    # Bunch of information for later analysis
    pred_prob_overall_test = []

    # Algorithm execution
    for idx_minibatch in range(settings.n_minibatches):

        time_method_init = t_ime.clock()
        train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]

        # Train the model always for the initial data
        if idx_minibatch == 0:
            # Batch training for first minibatch
            mf.fit(data, train_ids_current_minibatch, settings, param, cache)

        # Train the model if the size is below the limit
        else:

            if model_space_below_limit():
                # Online update
                mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)

        time_method_sans_init += t_ime.clock() - time_method_init

        # Make predictions
        time_predictions_init = t_ime.clock()
        weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
        train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]

        pred_forest_train, metrics_train = \
            mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
            data['y_train'][train_ids_cumulative], \
            settings, param, weights_prediction, False)

        # Predict for the next n data points in time
        pred_forest_test, metrics_test = \
            mf.evaluate_predictions(data, data['x_test'][start_pos:start_pos + (len(data['x_test'])/number_batches)],
            data['y_test'][start_pos:start_pos + (len(data['y_test'])/number_batches)], \
            settings, param, weights_prediction, False)

        # Collect information about prediction
        for prediction in pred_forest_test['pred_prob']:
            pred_prob_overall_test.append(prediction)

        name_metric = settings.name_metric     # acc or mse
        log_prob_train = metrics_train['log_prob']
        log_prob_test = metrics_test['log_prob']
        metric_train = metrics_train[name_metric]
        metric_test = metrics_test[name_metric]
        tree_numleaves = np.zeros(settings.n_mondrians)

        if settings.store_every:

                    log_prob_train_minibatch[idx_minibatch] = metrics_train['log_prob']
                    log_prob_test_minibatch[idx_minibatch] = metrics_test['log_prob']
                    metric_train_minibatch[idx_minibatch] = metrics_train[name_metric]
                    metric_test_minibatch[idx_minibatch] = metrics_test[name_metric]
                    time_method_minibatch[idx_minibatch] = 0 #FIXME
                    tree_numleaves = np.zeros(settings.n_mondrians)

                    for i_p, p in enumerate(mf):
                        tree_numleaves[i_p] = len(p.leaf_nodes)
                    forest_numleaves_minibatch[idx_minibatch] = np.mean(tree_numleaves)

        tree_leafes_total = 0

        for i_t, tree in enumerate(mf.forest):
            tree_numleaves[i_t] = len(tree.leaf_nodes)
            tree_leafes_total += len(tree.leaf_nodes)

        # Print results
        forest_numleaves = np.mean(tree_numleaves)
        print '%9d\t\t%.3f\t\t%.3f' % (idx_minibatch, metric_test, forest_numleaves)

        # Additional space information for analysis
        print_space_inf = print_space_stats()

        if print_space_inf:

            print "Current total tree leaf nodes : " + str(tree_leafes_total)
            cur_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 # convert to MB
            print "Current total memory usage in MB: " + str(cur_mem_usage)
            print


        time_prediction += t_ime.clock() - time_predictions_init
        accuracy.append(metric_test)
        start_pos += (len(data['x_test'])/number_batches)

    # Total time w/o saving results
    time_total = t_ime.clock() - time_0
    time = [time_method_sans_init, time_prediction, time_total]

    # Process and dump statistics to file if desired
    if settings.save == 1:

        test = []
        train = []
        time_method_mb = 0
        forest_numleaves_mb = 0

        if settings.store_every:

            test = [log_prob_test_minibatch, metric_test_minibatch]
            train = [log_prob_train_minibatch, metric_train_minibatch]
            time_method_mb = time_method_minibatch
            forest_numleaves_mb = forest_numleaves_minibatch

        time = [time_method_sans_init, time_prediction, time_method_mb, time_total]
        metrics = [metric_test, metric_train]

        process_statistics(settings, data, pred_prob_overall_test, log_prob_train, metrics, test, train, time,
                           forest_numleaves_mb)

    # Print statistics to command line
    print_statistics(settings, accuracy, data, mf, time)


def process_statistics(settings, data, pred_prob_overall_test, log_prob_train, metrics, test, train, time,
                       forest_numleaves_minibatch):

    """
    Process various statistics, such as concrete predictions (pred_prob_overall_test).
    Dump them to pickle file for later use, such as for building a confusion matrix.
    """

    # File to dump statistics
    filename = get_filename_mf(settings)

    results = {'log_prob_test': pred_prob_overall_test, 'log_prob_train': log_prob_train, \
                'metric_test': metrics[0], 'metric_train': metrics[1], \
            'time_total': time[3], 'time_method': 0, \
            'time_init': time_0, 'time_method_sans_init': time[0],\
            'time_prediction': time[1]}

    if settings.store_every:

        results['log_prob_test_minibatch'] = test[0]
        results['log_prob_train_minibatch'] = train[0]
        results['metric_test_minibatch'] = test[1]
        results['metric_train_minibatch'] = train[1]
        results['time_method_minibatch'] = time[2]
        results['forest_numleaves_minibatch'] = forest_numleaves_minibatch

    results['settings'] = settings
    results['tree_stats'] = tree_stats = np.zeros((settings.n_mondrians, 2))
    pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Store final predictions as well; recreate new "results" dict
    results = {'pred_prob_overall_train': None, \
                'pred_prob_overall_test': pred_prob_overall_test}

    # Dump the result into a pickle file
    filename2 = filename[:-2] + '.tree_predictions.p'
    pickle.dump(results, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print
    print 'stats_filename = ' + filename2


def print_statistics(settings, accuracy, data, mf, time):

    """
    Print important information, such as time figures.
    """

    print '\nFinal forest stats:'
    print "\nAverage Accuracy = " + str(np.mean(accuracy))
    tree_stats = np.zeros((settings.n_mondrians, 2))
    tree_average_depth = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_stats[i_t, -2:] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
        tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
    print 'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f' \
            % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
    print 'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' \
            % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))

    print
    print 'Time for executing mondrianforest.py (seconds) = %f' % (time[0])
    print 'Time for prediction/evaluation (seconds) = %f' % (time[1])

    if settings.save == 1:
    	
	print 'Total time (Loading data/ initializing / running / predictions) (seconds) = %f\n' % (time[3])

    else: 
	
	print 'Total time (Loading data/ initializing / running / predictions) (seconds) = %f\n' % (time[0] + time[1])


def model_space_below_limit():

    """
    Return True if the current model size is below the specified limit.
    Otherwise return False to stop training the model from now on.
    """

    cur_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 # convert to MB

    cfg = get_config()
    space_limit = float(json.loads(cfg.get('classification', 'model_space_limit')))

    if cur_mem_usage <  space_limit:

        return True

    else: return False


def print_space_stats():

    """
    Return if additional space stats should be printed.
    """

    cfg = get_config()
    print_stats = json.loads(cfg.get('classification', 'print_space_stats'))

    if print_stats == "True":

        return True

    else: return False


# Run Mondrian Forest
execute_mf()

