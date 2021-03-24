import numpy as np
import csv


def read_tr_file(name, internal_test):
    """
    Metodo per la lettura dei dati di training dal rispettivo file csv
    """
    try:
        tr_set = []
        with open(name) as datafile:
            filereader = csv.reader(datafile, delimiter=",")
            for row in filereader:
                if row[0][0] == "#":
                    continue

                pattern = np.array([float(value) for value in row[1:11]])
                expected_values = np.array([float(value) for value in row[11:13]])
                tr_set.append((pattern, expected_values))

        tr_set = np.array(tr_set)
        np.random.shuffle(tr_set)  # per ora commentiamolo
        n = int((1 - internal_test) * len(tr_set))  # indice di sperazione fra training e validation
        internal_test_set = tr_set[n:]  # estraggo i pattern che andranno a formare il set di validation
        tr_set = tr_set[:n]  # estraggo i pattern che andranno a formare il set di training
        return tr_set, internal_test_set

    except IOError:
        print("it was impossible to retrieve data from " + name)


def read_ts_file(name):
    """
    Metodo per la lettura dei dati di test dal rispettivo file csv
    """
    try:
        ts_set = []  # "matrice" contenente i pattern ed i valori a loro relativi
        with open(name) as datafile:
            filereader = csv.reader(datafile, delimiter=",")
            for row in filereader:
                if row[0][0] == "#":
                    continue

                ts_set.append([float(value) for value in row[1:11]])

        return np.array(ts_set)

    except IOError:
        print("it was impossible to retrieve data from " + name)


def read_monk(name: str, val: float, encoding: bool):
    """
    Metodo per la lettura dei dati dei file monk, training o test
    """
    try:
        tr_set = []
        with open(name) as datafile:
            monk_reader = csv.reader(datafile, delimiter=" ")
            for row in monk_reader:
                pattern = np.zeros(17)  # patterns
                pattern[int(row[2])-1] = 1  # codifico a1
                pattern[int(row[3])+2] = 1  # codifico a2
                pattern[int(row[4])+5] = 1  # codifico a3
                pattern[int(row[5])+7] = 1  # codifico a4
                pattern[int(row[6])+10] = 1  # codifico a5
                pattern[int(row[7])+14] = 1  # codifico a6
                value: int = int(row[1])
                if not encoding:
                    expected_values = np.array([value])
                else:
                    expected_values = np.array((1-value, value))

                tr_set.append((pattern, expected_values))

        tr_set = np.array(tr_set)
        if ".test" not in name and val > 0:  # train_monk
            n = int((1-val)) * len(tr_set)  # indice di separazione fra training set e validation set
            val_set = tr_set[n:]  # estraggo i pattern che andranno a formare il set di validation
            tr_set = tr_set[:n]  # estraggo i pattern che andranno a formare il set di training
            return tr_set, val_set
        else:  # test_monk oppure niente validation
            return tr_set

    except IOError:
        print("it was impossible to retrieve data from " + name)


def write_report(name: str, values):
    try:
        with open(name, "w", newline="") as datafile:
            report_writer = csv.writer(datafile, delimiter=",")
            cup = [["# Niko Dalla Noce, Alessandro Ristori"], ["# SushiPizza"], ["# ML-CUP20 v1"], ["# 24/01/2021"]]
            report_writer.writerows(cup)
            for out, i in zip(values, np.arange(len(values))):
                report_writer.writerow([i+1, out[0], out[1]])

    except IOError:
        print("it was impossbile to write data on " + name)
