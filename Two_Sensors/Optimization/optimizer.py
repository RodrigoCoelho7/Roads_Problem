import optuna
import numpy as np
import matplotlib.pyplot as plt
from utils import INTERSECTION

hour = 3600
TRIALS = 100
TIMEOUT = hour*1

def objective(trial):

    global BEST_RESULT
    global HISTORY
    global BEST_MODEL

    model = INTERSECTION(trial.suggest_int("Tlow", 15, 60),  trial.suggest_int("Thigh", 30, 120))
    model.time_evolution(3600)
    mean_cars = np.array([np.mean(road.actual_cars)+np.max(road.actual_cars) for road in model.roads])
    result = np.sqrt(np.sum(mean_cars**2))

    HISTORY.append(result)

    if result < BEST_RESULT:
        BEST_RESULT = result
        BEST_MODEL = model

    return result

BEST_RESULT = np.inf
HISTORY = []
BEST_MODEL = None

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)

study.optimize(objective, TRIALS, TIMEOUT, show_progress_bar=True)

print('\n\nEND STUDY:\n')
print(f'\n  Best result: {BEST_RESULT} TIMES: {BEST_MODEL.TIMES}')

bins = np.arange(0,3600+180,180)
bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
fig,ax = plt.subplots()
for i,road in enumerate(BEST_MODEL.roads):
    total_cars = [np.sum(road.total_cars[j*180:(j+1)*180]) for j in range(20)]
    ax.step(bincentres,total_cars,label=f'Road{i+1}')  
ax.set_title('Total Cars')
ax.legend()

fig2,ax2 = plt.subplots()
for i,road in enumerate(BEST_MODEL.roads):
    ax2.plot(road.actual_cars,label=f'Road{i+1}')  
ax2.set_title('Actual Cars')
ax2.legend()

fig3,ax3 = plt.subplots()
ax3.plot(HISTORY)
plt.show()