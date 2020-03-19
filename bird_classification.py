import ray
import tensorflow as tf
import hpo
import bird_classification_models as model_configurations
import hpo.strategies.genetic_algorithm as hpo_strategy_ga
import bird_classification_data
import os


ray.init()

data_dir = "/home/rob/dissertation/data/bird_classification"
cache_path = os.path.join(os.getcwd(), ".cache")
if not os.path.exists(cache_path):
    os.mkdir(cache_path)


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)


def construct_bird_data():
    return bird_classification_data.BirdData(data_dir, cache_path, 100, 100, 100)


###############################################################################
# TODO: Remove this code, example only...
# results = hpo.Results.load(os.path.join(os.getcwd(), ".tmp/hpo.results.tmp"))
#
# model = hpo.Hpo.build_model(results.history()[0].model_configuration())
# model.train(construct_bird_data)
# exit()
###############################################################################

optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.001,
                  value_range=[1 * (10 ** n) for n in range(0, -7, -1)])
])

model_configuration = hpo.ModelConfiguration(optimiser=optimiser, layers=model_configurations.CATS_AND_DOGS, loss_function="categorical_crossentropy", number_of_epochs=10)
print(model_configuration.number_of_hyperparameters())
model_configuration.hyperparameter_summary(True)

strategy = hpo_strategy_ga.GeneticAlgorithm(population_size=30, max_iterations=10, chromosome_type=construct_chromosome,
                                            survivour_selection_stratergy="roulette")

strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().survivour_percentage(0.7)

hpo_instance = hpo.Hpo(model_configuration, construct_bird_data, strategy)

hpo_instance.execute()
