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


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)


def construct_bird_data():
    return bird_classification_data.BirdData(data_dir, cache_path, 100, 100, 100)


def model_exception_handler(e):
    print("Exception occured while training the model.", e)


model_configuration = hpo.ModelConfiguration(optimiser=model_configurations.optimiser, layers=model_configurations.cats_and_dogs_cnn, loss_function="categorical_crossentropy", number_of_epochs=10)
print(model_configuration.number_of_hyperparameters())
model_configuration.hyperparameter_summary(True)

#####################################
# Random Search
#####################################
strategy = hpo.strategies.random_search.RandomSearch(model_configuration, 100)
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_bayesian_random_forest.results"))

#####################################
# Bayesian Selection - Random Forest
#####################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 100, hpo.strategies.bayesian_method.RandomForestSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_bayesian_random_forest.results"))

#########################################
# Genetic Algorithm - Roulette Selection
##########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=10, max_iterations=10, chromosome_type=construct_chromosome,
                                                            survivour_selection_stratergy="roulette")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().survivour_percentage(0.7)
hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_genetic_algorithm_roulette.results"))