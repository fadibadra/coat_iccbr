import os
import re
import time
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import coat as ct
from coat.similarity import ES
from data.dataset import Overlap
from data.transform import random_transformation
from ml.coat import energy, upper_bound, predict
from ml.evaluation import collect_all_instances, pearson, split, MultiClassConfusionMatrix, nfolds, shuffle
from data.dataset import *
from data.transform import TRANSFORMATIONS_DIR

RESULTS_DIR = os.path.dirname(
    os.path.abspath(__file__))+'/../resources/results/'

"""
E1
--
Prediction Time experiment
Optimized vs non optimized CoAT
--
"""


def outfile_result_E1(d):
    return RESULTS_DIR+d.name+'_prediction_time'


def outfile_train_E1(d):
    return RESULTS_DIR+d.name+'_train'


def outfile_test_E1(d):
    return RESULTS_DIR+d.name+'_test'


def write_header_results_E1(out_file):
    with open(out_file, 'w') as f:
        f.write('N;non_optim;optim\n')


def write_results_E1(d, N, t_no_optim, t_optim):
    with open(outfile_result_E1(d), 'a') as f:
        f.write(str(N)+';'+str(t_no_optim)+';'+str(t_optim)+'\n')


def already_saved_E1(d, N):
    """ Checks if the result for N was already computed or not for this dataset. """
    found = False
    out_file = outfile_result_E1(d)
    if os.path.exists(out_file):
        f = open(out_file, 'r')
        for line in f:
            if re.match(str(N)+';', line) and len(line.strip()) > 0:
                found = True
    return found


def get_train_and_test(d):
    train_file = outfile_train_E1(d)
    test_file = outfile_test_E1(d)
    if not os.path.exists(train_file):
        df = shuffle(d.train)
        train = df[:-10]
        test = df[-10:]
        train.to_csv(train_file)
        test.to_csv(test_file)
    return (pd.read_csv(train_file), pd.read_csv(test_file))


def predict_optim(d, s, o, s_test):
    """ Optimized CoAT prediction. """
    start = time.time()
    v_optim = ct.predict(s, o, s_test, d.outcomes)
    end = time.time()
    return (v_optim, end-start)


def predict_no_optim(d, s, o, s_test):
    """ Non optimized CoAT prediction. """
    start = time.time()
    s_new = s.add(s_test)
    energies = []
    for r in d.outcomes:
        o_new = o.add([r])
        e = ct.energy(s_new, o_new)
        energies.append(e)
    v_no_optim = d.outcomes[energies.index(min(energies))]
    end = time.time()
    return(v_no_optim, end-start)


def fit_polynomial(N_values, av_time, degree, color, label):
    X_plot = np.linspace(0, max(N_values)+30, len(N_values)+1)[:, np.newaxis]
    X_train = np.array(N_values)[:, np.newaxis]
    y_train = np.array(av_time)
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X_train, y_train)
    # plt.plot(X_plot,polyreg.predict(X_plot),'-',color="firebrick")
    rmse = np.sqrt(mean_squared_error(y_train, polyreg.predict(X_train)))
    print(f'RMSE={rmse:.8f}')
    coefs = tuple(polyreg.steps[1][1].coef_)
    print(coefs)

    def p(x):
        return sum((coefs[i]*x**i for i in range(len(coefs))))
    plt.plot(X_plot, p(X_plot), '-', color=color, label=label)


def draw_E1(d):
    """ Draws results for E1 experiment. """
    (N_values, av_time_no_optim, av_time_optim) = ([], [], [])
    with open(outfile_result_E1(d), 'r') as f:
        for line in f:
            if not re.match('N', line) and len(line.strip()) > 0:
                (N, t_no_optim, t_optim) = line.split(';')
                N_values.append(int(N))
                av_time_no_optim.append(float(t_no_optim))
                av_time_optim.append(float(t_optim))

    plt.rc('axes', labelsize=16)
    plt.rc('font', **{'family': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, max(N_values)+30)
    plt.ylim(0, max(av_time_no_optim)+1)

    fit_polynomial(N_values, av_time_no_optim, 3, 'firebrick',
                   label=r'$f(n) = 1.25\:10^{-7}\:n^3 + 1.62\:10^{-5}\:n^2 + 5.84\:10^{-4}\:n$')
    fit_polynomial(N_values, av_time_optim, 2, '#55557fff',
                   label=r'$f(n) = 6.17\:10^{-6}\:n^2 + 4.42\:10^{-4}\:n$')

    plt.scatter(N_values, av_time_no_optim, c='firebrick',
                marker='^', label=r'Non optimized')
    plt.scatter(N_values, av_time_optim, c='#55557fff',
                marker='o', label=r'Optimized')
    plt.xlabel(r'size $n$ of the case base')
    plt.ylabel(r'prediction time (s)')
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 0, 3, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.tight_layout()
    plt.show()


def run_E1(d):
    """ Main procedure for the E1 experiment. """
    results_file = outfile_result_E1(d)
    if not os.path.exists(results_file):
        write_header_results_E1(results_file)
    (train, test) = get_train_and_test(d)
    test_data = test[d.features].values
    n = len(test_data)
    av_time_no_optim = []
    av_time_optim = []
    N_values = list(range(10, len(train)-len(train) % 10+1, 10))
    N_values_still_to_compute = [
        N for N in N_values if not already_saved_E1(d, N)]
    for N in N_values_still_to_compute:
        t_no_optim_N = 0.
        t_optim_N = 0.
        s = ct.simmatrix(train[:N][d.features].values, ES())
        o = ct.simmatrix(train[:N][d.tgt_att].values, d.outcome_scale)
        for k in range(n):
            s_test = test_data[k]

            # non optimized version
            (v_no_optim, t_no_optim) = predict_no_optim(d, s, o, s_test)

            # optimized version
            (v_optim, t_optim) = predict_optim(d, s, o, s_test)

            if v_optim != v_no_optim:
                print(f'different value predicted!')

            t_no_optim_N += t_no_optim
            t_optim_N += t_optim
        t_no_optim_N /= n
        t_optim_N /= n
        av_time_no_optim.append(t_no_optim_N)
        av_time_optim.append(t_optim_N)
        print(f'N={N} \t non optim {t_no_optim_N:.8f} s \toptim {t_optim_N:.8f}')
        write_results_E1(d, N, t_no_optim_N, t_optim_N)


"""
E2
--
Relation between Complexity and Performance
--
"""

NB_INSTANCES = 200


def outfile_E2(d): return RESULTS_DIR + \
    d.name+'_loss_acc_pairs'


def outfile_E2_random(
    d): return outfile_E2(d)+'_random'


def outfile_E2_euclidian(
    d): return outfile_E2(d)+"_euclidian"


def get_nb_saved(out_file):
    """ Retrieves the number of transformations already saved in the file. """
    n = 0
    if os.path.exists(out_file):
        f = open(out_file, 'r')
        for line in f:
            if not re.match('energy', line) and len(line.strip()) > 0:
                n += 1
    return n


def write_header_E2(out_file):
    with open(out_file, 'w') as f:
        f.write(
            'energy;coat_acc;coat_std;nn_acc;nn_std\n')


def write_pair_to_file(out_file, energy, average_coat_accuracy, std_coat, average_nn_accuracy, std_nn):
    if not os.path.exists(out_file):
        write_header_E2(out_file)
    with open(out_file, 'a') as f:
        f.write(str(energy)+';'+str(average_coat_accuracy) +
                ';'+str(std_coat)+';'+str(average_nn_accuracy) +
                ';'+str(std_nn)+'\n')


def generate_loss_accuracy_pair(d, folds, L, out_file):
    fold_nb = 1
    total_start = time.time()
    (coat_accuracies, nn_accuracies) = ([], [])
    energies = []
    for train, test in split(folds):
        # print(f'fold {fold_nb} ({len(train)}/{len(test)})')
        Lx = np.matmul(train[d.features].values, L)
        s = ct.simmatrix(Lx, ES())
        train_targets = train[d.tgt_att].values
        o = ct.simmatrix(train_targets, d.outcome_scale)
        energies.append(ct.energy(s, o))
        test_data = np.matmul(test[d.features].values, L)
        test_targets = test[d.tgt_att].values
        m_coat = MultiClassConfusionMatrix(d.outcomes)
        m_nn = MultiClassConfusionMatrix(d.outcomes)
        prediction_time = 0.0
        nb_tests = len(test_data)
        for i in range(nb_tests):
            s_test = test_data[i]
            real_o = test_targets[i]
            # CoAT prediction
            start = time.time()
            v = ct.predict(s, o, s_test, d.outcomes)
            end = time.time()
            prediction_time += (end-start)
            m_coat.add_prediction(v, real_o)
            # NN prediction
            k = 5
            s1 = s.add(s_test)
            nn_outcomes = np.array(train_targets)[
                np.argsort(s1[len(s1)-1])[-k-1:-1]]
            (outcomes, counts) = np.unique(nn_outcomes, return_counts=True)
            nn_pred = outcomes[list(counts).index(max(counts))]
            m_nn.add_prediction(nn_pred, real_o)
        # print(m.m)
        # print(f'average prediction time: {(prediction_time)/n:.5f}s')
        coat_accuracy = m_coat.accuracy()
        coat_accuracies.append(coat_accuracy)
        nn_accuracy = m_nn.accuracy()
        nn_accuracies.append(nn_accuracy)
        fold_nb += 1
    E = sum(energies)/len(energies)
    average_coat_accuracy = sum(coat_accuracies)/len(coat_accuracies)
    std_coat = np.std(np.array(coat_accuracies))
    average_nn_accuracy = sum(nn_accuracies)/len(nn_accuracies)
    std_nn = np.std(np.array(nn_accuracies))
    print(
        f'E = {E:.2f}, CoAT {average_coat_accuracy:.2f} \u00B1 {std_coat:.5f}, 5NN {average_nn_accuracy:.2f} \u00B1 {std_nn:.5f}')
    total_time = (time.time()-total_start)
    print(f'Total time = {total_time:.5f}')

    write_pair_to_file(out_file, E, average_coat_accuracy,
                       std_coat, average_nn_accuracy, std_nn)


def generate_random_loss_accuracy_pairs(d, folds):
    """ Generate loss / accuracy pairs for random transformation matrices. """
    out_file = outfile_E2_random(d)
    Lis = load_transformations(d)
    # how many already computed?
    i = get_nb_saved(out_file)
    while i < len(Lis):
        L = Lis[i]
        print(f'L{i + 1:d}')
        print(L)
        generate_loss_accuracy_pair(
            d, folds, L, out_file)
        i += 1


def generate_euclidian_loss_accuracy_pair(d, folds):
    """ Generate a loss/accuracy pair for the identity L transformation. """
    out_file = outfile_E2_euclidian(d)
    if not os.path.exists(out_file):
        write_header_E2(out_file)
        L = np.identity(len(d.features))
        print(f'Id')
        print(L)
        generate_loss_accuracy_pair(d, folds, L, out_file)


def generate_loss_accuracy_pairs(d):
    df = collect_all_instances(d).sample(
        frac=1).reset_index(drop=True).iloc[:NB_INSTANCES]
    folds = nfolds(df, d)
    generate_euclidian_loss_accuracy_pair(d, folds)
    generate_random_loss_accuracy_pairs(d, folds)


def print_results(d):
    rand_df = pd.read_csv(outfile_E2_random(d), sep=';')
    n = len(rand_df)
    idx_max = rand_df.idxmax()['coat_acc']
    t = rand_df.iloc[idx_max]
    up = upper_bound(NB_INSTANCES)
    print(f'L* {t["coat_acc"]*100:.2f}% \u00B1 {t["coat_std"]} E/Emax = {t["energy"]/up:.5f} (best of {n} L transformations)')
    eucl_df = pd.read_csv(outfile_E2_euclidian(d), sep=';')
    t = eucl_df.iloc[0]
    print(
        f'Id {t["coat_acc"]*100:.2f}% \u00B1 {t["coat_std"]} E/Emax = {t["energy"]/up:.5f}')


def draw_E2(d):
    """ Draws energy/accuracy correlation diagram. """
    df = pd.read_csv(outfile_E2_random(d), sep=';')
    df_euclidian = pd.read_csv(outfile_E2_euclidian(d), sep=';')
    energy_values = list(df['energy'].values)
    plt.figure(figsize=(10, 8))
    plt.rc('axes', labelsize=22)
    plt.rc('font', **{'family': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    algo = 'coat'
    acc_values = list(df[algo+'_acc'].values)
    std_values = list(df[algo+'_std'].values)
    linreg = LinearRegression()
    X = np.array(energy_values)[:, np.newaxis]
    y = acc_values
    linreg.fit(X, y)
    plt.plot([v*10e-6 for v in energy_values],
             linreg.predict(X), color='green')
    plt.xlabel(r'Dataset complexity $(\times 10^6)$', fontsize=22)
    plt.ylabel(r'Accuracy (CoAT)', fontsize=22)
    plt.errorbar([v*10e-6 for v in energy_values], acc_values,
                 yerr=std_values, color='#55557fff', fmt='o')
    plt.errorbar([v*10e-6 for v in list(df_euclidian['energy'].values)],
                 list(df_euclidian[algo+'_acc'].values),
                 yerr=list(df_euclidian[algo+'_std'].values), color='firebrick', fmt='o')
    title = d.name
    if title == "balance":
        title = "Balance Scale"
    if title == "iris":
        title = "Iris"
    plt.title(rf'{title}',fontsize=26)
    p = pearson(energy_values, acc_values)
    print(f'Pearson coefficient p={p} for the {d.name} dataset.')
    plt.tight_layout()
    plt.show()


# LOAD / SAVE transformations Li
def generate_random_transformations(d, N):
    """ Generate and save N random transformations for dataset d. """
    Li_arrays = []
    i = 0
    while i < N:
        Li_arrays.append(random_transformation(d))
        i += 1
    np.save(TRANSFORMATIONS_DIR+d.name +
            '_transformations.npy', np.array(Li_arrays))


def load_transformations(d):
    """ Loads the Li transformations from file.
    @return an array of transformations. """
    return np.load(TRANSFORMATIONS_DIR+d.name+'_transformations.npy')


"""
E3
--
Relation between Complexity and Difficulty (class separability)
--
"""
def outfile_E3(): return RESULTS_DIR + 'E3_results'


def save_E3_results(scale, energy, energy_max):
    if os.path.exists(outfile_E3()):
        df = pd.read_csv(outfile_E3(), sep=';')
    else:
        df = pd.DataFrame(columns=['scale', 'energy', 'energy_max'])
    df = pd.concat([df, pd.DataFrame({'scale': [scale], 'energy':[
                   energy], 'energy_max':[energy_max]})], ignore_index=True)
    df.sort_values(by=['scale'], inplace=True)
    df.to_csv(outfile_E3(), sep=';', index=False)


def already_saved_E3(scale):
    """ Checks if the result for a scale value was already computed or not. """
    found = False
    out_file = outfile_E3()
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, sep=';')
        if scale in df['scale'].values:
            found = True
    return found


def run_E3():
    for scale in range(501):
        if already_saved_E3(scale):
            print(f'{scale} already saved')
        else:
            d = Overlap(100, scale)
            e = energy(d)[0]
            Emax = upper_bound(len(d.train))
            print(
                f'scale={scale:.2f} E={int(e)} E/Emax={e/Emax*100:.5f}% of {Emax}')
            save_E3_results(scale, e, Emax)
    # d.show()


def draw_E3():
    df = pd.read_csv(outfile_E3(), sep=';')
    fig, axs = plt.subplots(4, 2, sharey=True, figsize=(10, 20))
    scales = np.array([[1, 3], [5, 10], [25, 50], [100, 200]])
    for (i, j) in product(range(scales.shape[0]), range(scales.shape[1])):
        scale = scales[i, j]
        e = df[df['scale'] == scale]['energy'].values[0]
        Emax = df[df['scale'] == scale]['energy_max'].values[0]
        (xA, yA, xB, yB) = Overlap(100, scale).load_distribution()
        a = axs[i, j]
        a.scatter(xA, yA, marker='.', color='royalblue', s=[30]*len(xA))
        a.scatter(xB, yB, marker='.', color='coral', s=[30]*len(xB))
        a.axis('equal')
        a.axis([-50, 50, -50, 50])
        a.set_title(f'scale={scale} \t' +
                    r'$\Gamma / \Gamma_{max} = $'+f'{e/Emax*100:.2f}%')

    plt.subplots_adjust(top=0.945,
                        bottom=0.06,
                        left=0.17,
                        right=0.825,
                        hspace=0.38,
                        wspace=0.61)
    # left  = 0.125  # the left side of the subplots of the figure
    # right = 0.9    # the right side of the subplots of the figure
    # bottom = 0.1   # the bottom of the subplots of the figure
    # top = 0.9      # the top of the subplots of the figure
    # wspace = 0.2   # the amount of width reserved for blank space between subplots
    # hspace = 0.2   # the amount of height reserved for white space between subplots
    plt.show()


def plot_E3():
    df = pd.read_csv(outfile_E3(), sep=';')
    x = list(df['scale'].values)
    y = list(df['energy']/df['energy_max'])
    plt.rc('axes', labelsize=16)
    plt.rc('font', **{'family': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(x, y, '-', color='#55557fff')
    plt.xlabel(r'overlapping degree (scale)')
    plt.ylabel(r'$\Gamma / \Gamma_{max}$')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1., decimals=0))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    """
    Usage :
        python iccbr.py complexity [Balance|Iris|Pima|Monks1|Monks2|Monks3|User|Voting|Wine]
        python iccbr.py coat [Balance|Iris|Pima|Monks1|Monks2|Monks3|User|Voting|Wine]
        python iccbr.py Exp1 Balance [run|draw]
        python iccbr.py Exp2 Balance [generate|results|draw]
        python iccbr.py Exp3 [run|draw|plot]
    """
    import sys
    method = sys.argv[1]
    if method in ('Exp1', 'Exp2','complexity', 'coat'):
        name = sys.argv[2]
        d = getattr(__import__(
            'data.dataset', fromlist=[name]), name)()
        if method == 'complexity':
            Emax = upper_bound(len(d.train))
            (E,prediction_time) = energy(d)
            print(f'Complexity={int(E):d} ({E/Emax*100:.4f}% of {Emax})')
            print(f'Total time = {prediction_time:.5f}s')
        if method == 'coat':
            (average_accuracy,std,average_prediction_time) = predict(d)
            print(f'Average prediction time = {average_prediction_time:.5f} s')
            print(f'Average accuracy = {average_accuracy:.5f}')
            print(f'Std deviation = {std:.5f}')
        if method == 'Exp1':
            op = sys.argv[3]
            if op == 'run':
                run_E1(d)
            if op == 'draw':
                draw_E1(d)
        if method == 'Exp2':
            op = sys.argv[3]
            if op == 'generate':
                print(f'{d.name}')
                generate_loss_accuracy_pairs(d)
            if op == 'draw':
                draw_E2(d)
            if op == 'results':
                print_results(d)
            if op == 'init_Li':
                generate_random_transformations(d, 100)
    if method == 'Exp3':
        op = sys.argv[2]
        if op == 'run':
            run_E3()
        if op == 'draw':
            draw_E3()
        if op == 'plot':
            plot_E3()
