import tensorflow as tf
import hpo
import bird_classification_models as model_configurations
import hpo.strategies.genetic_algorithm
import hpo.strategies.bayesian_method
import hpo.strategies.random_search
import bird_classification_data
import os
import hpo_experiment_runner


data_dir = "/home/rob/dissertation/data/bird_classification"
cache_path = os.path.join(os.getcwd(), ".cache")
if not os.path.exists(cache_path):
    os.mkdir(cache_path)


def construct_chromosome(remote_model_type):
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration, remote_model_type)


def construct_bird_data():
    return bird_classification_data.BirdData(data_dir, cache_path, 30, 30, 30)


def model_exception_handler(e):
    print("Exception occured while training the model.", e)


model_configuration = hpo.DefaulDLModelConfiguration(optimiser=model_configurations.optimiser, layers=model_configurations.cnn, loss_function="categorical_crossentropy", number_of_epochs=20)

#####################################
# Bayesian Selection - Random Forest
#####################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 50, hpo.strategies.bayesian_method.RandomForestSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run_with_live_graphs(hpo_instance, os.path.join(os.getcwd(), "bird_classification_hpo_bayesian_random_forest.results"))

#####################################
# Random Search
#####################################
strategy = hpo.strategies.random_search.RandomSearch(model_configuration, 50)
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "bird_classification_hpo_random_search.results"))

#########################################
# Genetic Algorithm - Roulette Selection
##########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=10, max_iterations=5, chromosome_type=construct_chromosome,
                                                            survivour_selection_stratergy="roulette")
strategy.mutation_strategy().mutation_probability(0.1)
strategy.survivour_selection_strategy().survivour_percentage(0.8)
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "bird_classification_hpo_genetic_algorithm_roulette.results"))
