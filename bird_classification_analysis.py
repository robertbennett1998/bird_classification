from hpo import Results, Result
import os
from pathlib import Path
import matplotlib.pyplot as plt

all_results = dict()
result_file_paths = Path.glob(Path(os.getcwd()), "*.results")
for result_file_path in result_file_paths:
    key = result_file_path.stem.replace('_', ' ').replace("_hpo", "")
    print(key)
    all_results[key] = Results.load(result_file_path)

legend = list()
plt.subplots(1)

for key, results in all_results.items():
    ys = [result.training_history()["val_accuracy"][-1] for result in results.history()]
    ys_loss = [result.training_history()["val_loss"][-1] for result in results.history()]
    print(key, results.best_result().score(), results.best_result().meta_data(), sum(ys) / len(ys))
    plt.plot(ys)
    plt.title(key)
    plt.show()
    plt.plot(ys_loss)
    plt.title(key + "loss")
    plt.show()

