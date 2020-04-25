import hpo
import os

results = hpo.Results.load(os.path.join(os.getcwd(), ".tmp/hpo.results.tmp"))
print(results.history()[-1].meta_data())
results.plot_average_score_over_optimisation_period()
