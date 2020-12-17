
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from dataset_builder_loader.data_loader import *
from utilities.utils import *
from sleep_stage_config import Config

cfg = Config()
data_loader = DataLoader(cfg, "all", 2, 20)
data_loader.load_ml_data()
x_train = data_loader.x_train[: int(0.8*len(data_loader.x_train))]
y_train = data_loader.y_train[: int(0.8*len(data_loader.y_train))]
x_val = data_loader.x_train[int(0.8*len(data_loader.x_train)):]
y_val = data_loader.y_train[int(0.8*len(data_loader.y_train)):]

trained_model_path = r"D:\mesa_ml_trained\2_stages_30s_ml_all.pkl"
with open(trained_model_path, "rb") as store:
    trained_models = pickle.load(store)
lr_model = trained_models['models'][1][1]

y_prob_val = lr_model.predict_proba(x_val)
y_prob_val = y_prob_val[:, 1]
plot_roc_curve(y_val, y_prob_val, r"C:\tmp\sleep")
print("loading liner regression models finished")
