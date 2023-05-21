import pandas as pd
from sklearn.utils import shuffle
from PreProcessing import *
from backpropagationalgo import BackPropagationAlgo
import warnings

warnings.simplefilter("ignore")

back_propagation = BackPropagationAlgo()


def run(NumHiddenLayers, NumNeuronsInHiddenLayers, ActivationFun, eta, epochs, BiasCheck):
    data = pd.read_csv(r'dataset\penguins.csv')

    data = FillingData(data)
    data = Feature_Encoder(data)

    # Min-Max Normalization
    df = data.drop('species', axis=1)
    df_norm = (df - df.min()) / (df.max() - df.min())
    df_norm.insert(0, 'species', data.species)
    data = df_norm

    class_ = data['species']
    rows = []

    c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for i in range(len(data)):
        row = [data['bill_length_mm'][i], data['bill_depth_mm'][i], data['flipper_length_mm'][i], data['gender'][i],
               data['body_mass_g'][i]]
        if class_[i] == 'Adelie':
            row.append(c[0])
        elif class_[i] == 'Gentoo':
            row.append(c[1])
        elif class_[i] == 'Chinstrap':
            row.append(c[2])
        rows.append(row)

    data = pd.DataFrame(rows, columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g',
                                       'species'])

    NumNeuronInHiddenLayersLIST = []
    for i in range(0, NumHiddenLayers):
        NumNeuronInHiddenLayersLIST.append(int(NumNeuronsInHiddenLayers[i]))

    C1 = data.iloc[0:50]
    C2 = data.iloc[50:100]
    C3 = data.iloc[100:150]

    frames = [C1[0:30], C2[0:30], C3[0:30]]
    train = pd.concat(frames)
    train = shuffle(train)
    L_train = train.to_numpy().reshape(90, 6)

    frames = [C1[30:50], C2[30:50], C3[30:50]]
    test = pd.concat(frames)
    test = shuffle(test)
    L_test = test.to_numpy().reshape(60, 6)

    if BiasCheck:
        L_train = np.c_[np.ones((90, 1)), L_train]
        L_test = np.c_[np.ones((60, 1)), L_test]

    # training function
    weights = back_propagation.StartBackpropagationAlgo(L_train, NumHiddenLayers, NumNeuronInHiddenLayersLIST,
                                                        ActivationFun, eta,
                                                        epochs,
                                                        BiasCheck)
    #print("TESTING")
    # testing function
    test_acc = Test_Fun(L_test, weights, ActivationFun, BiasCheck)
    train_acc = Test_Fun(L_train, weights, ActivationFun, BiasCheck)

    return test_acc, train_acc


def Test_Fun(L_test, weights, active_func, bias_check):
    y_pred = []
    for i in range(len(L_test)):  # output activation function for output layer only
        y_pred.append(back_propagation.Feedforward(L_test[i][0:-1], weights, active_func, bias_check)[-1])
    y_pred = np.array(y_pred)

    for i in range(len(y_pred)):  # يحولها ل 0 1 0  ويحط ال 1 مكان اكبر رقم ف الصف
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == np.max(y_pred[i]):
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    count = 0
    c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    C_M = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    for i in range(len(y_pred)):
        # print("actual= [ "+ str(L_test[i][6][0]) +", "+ str(L_test[i][6][1]) +", "+ str(L_test[i][6][2]) +" ]" +
        #      "  " + "pred= [ "+ str(y_pred[i][0]) +", "+ str(y_pred[i][1]) +", "+ str(y_pred[i][2]) +" ]" +
        #         "   "+"i= "+str(i))
        #   [1, 0, 0], [0, 1, 0], [0, 0, 1]
        class1 = -1
        class2 = -1

        if (L_test[i][-1][0] == y_pred[i][0]) and (L_test[i][-1][1] == y_pred[i][1]) and (
                L_test[i][-1][2] == y_pred[i][2]):
            count += 1

        if L_test[i][-1] == c[0]:
            class1 = 0
        elif L_test[i][-1] == c[1]:
            class1 = 1
        elif L_test[i][-1] == c[2]:
            class1 = 2

        if (c[0][0] == y_pred[i][0]) and (c[0][1] == y_pred[i][1]) and (c[0][2] == y_pred[i][2]):
            class2 = 0
        elif (c[1][0] == y_pred[i][0]) and (c[1][1] == y_pred[i][1]) and (c[1][2] == y_pred[i][2]):
            class2 = 1
        elif (c[2][0] == y_pred[i][0]) and (c[2][1] == y_pred[i][1]) and (c[2][2] == y_pred[i][2]):
            class2 = 2
        C_M[class2][class1] = C_M[class2][class1] + 1
    print(" ")
    print("\t Confusion Matrix \t")
    print("\t C1 \t C2 \t C3")
    print("-------------------------------")
    num = 1
    for i in range(3):
        print("C" + str(num) + "\t", end=' ')
        for j in range(3):
            print(str(C_M[j][i]) + "\t\t", end=' ')
        print()
        num += 1
    print("-------------------------------")

    print("Count: " + str(count))
    accuracy = count / len(y_pred) * 100
    print("accuracy: " + str(accuracy))
    print(" ")

    return accuracy
