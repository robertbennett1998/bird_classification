import hpo
import os

#results = hpo.Results.load(os.path.join(os.getcwd(), "bird_classification_hpo_genetic_algorithm_threshold.results")) 
results = hpo.Results.load(os.path.join(os.getcwd(), ".tmp/hpo.results.tmp"))
print(results.history()[-1].meta_data())
best_result = results.best_result()
print("Best Result:", best_result.training_history()["val_accuracy"][-1], "-", best_result.meta_data())
#results.plot_average_score_over_optimisation_period()