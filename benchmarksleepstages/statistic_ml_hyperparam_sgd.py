from sklearn import metrics
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
import sys, argparse
from sleep_stage_config import Config
from dataset_builder_loader.data_loader import *


def main(args):
    cfg = Config()
    data_loader = DataLoader(cfg, args.modality, args.num_classes, 20)
    print("...Loading dataset into memory...")
    df_train, df_test, featnames = data_loader.load_df_dataset()
    print("...Done...")

    # for linear SVM or hinge loss, C_svc * n_samples = 1 / alpha_sgd
    # 1. / C_svr ~ 1. / C_svc ~ 1. / C_logistic ~ alpha_elastic_net * n_samples ~ alpha_lasso * n_samples
    # ~ alpha_sgd * n_samples ~ alpha_ridge
    sgd_params = [
        {'classifier__loss': ['hinge', 'log', 'perceptron'],
         'classifier__penalty': ['l1', 'l2', 'elasticnet'],
          'classifier__fit_intercept': [True, False],
          'classifier__max_iter': [5, 10, 20],
         'classifier__alpha': 10.0 ** -np.arange(1, 7)
         }
    ]
    classifier = SGDClassifier(random_state=42, n_jobs=1)
    pipe = Pipeline([
        ('classifier', classifier)
    ])
    print("...optimizing parameters for %s..." % (pipe.get_params()["classifier"]))
    print("..Scoring function is %s..." % "accuracy")

    gs = optimize_parameters(df_train, featnames, pipe, sgd_params, "accuracy", "sgd", num_classes=args.num_classes,
                             output_path=cfg.HP_CV_OUTPUT_FOLDER,)
    gsres = display_parameters(gs)

    with open(os.path.join(cfg.HP_CV_OUTPUT_FOLDER, "%d_stages_gs_sgd_%ds_hrv_len_%s.pkl" %
                                          (args.num_classes, args.hrv_win_len, args.modality)), "wb") as f:
        pickle.dump(gs, f)
    predictions = gs.predict(df_test[featnames])

    print(" - Acuracy: %.4f" % (metrics.accuracy_score(df_test["stages"], predictions)))
    print(" - F1: %.4f" % (metrics.f1_score(df_test["stages"], predictions, average=args.eval_method)))
    print(" - Best parameter: %s" % gs.best_estimator_)
    gsres.to_csv(os.path.join(cfg.HP_CV_OUTPUT_FOLDER,  datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"))
    print("...Generated file '%s'..." % cfg.HP_CV_OUTPUT_FOLDER)
    print("...All done...")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all', help='the raw modality to use')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')
    parser.add_argument('--eval_method', type=str, default='macro',
                        help='The weighting method for classifier level metrics')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
