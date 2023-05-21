import numpy as np

class BackPropagationAlgo:

    def Sigmoid(self, net):
        F = 1 / (1 + np.exp(-net))
        return F

    def TanH(self, net):
        F = (1 - np.exp(-net)) / (1 + np.exp(-net))
        return F

    def StartBackpropagationAlgo(self, TrainData, NumHiddenLayers, neuron_in_hidden_layers, ActivationFun, eta, epochs, BiasCheck):
        # create_random weights
        weights = []
        input_class = 5
        for i in range(NumHiddenLayers):
            rand_matrix = np.random.uniform(-1, 1, size=(neuron_in_hidden_layers[i], input_class + int(BiasCheck)))
            input_class = neuron_in_hidden_layers[i]
            weights.append(rand_matrix)

        weights.append(np.random.rand(3, input_class + int(BiasCheck)))   # output weights

        for epoch in range(epochs):
            for i in range(len(TrainData)):
                x = TrainData[i][0:-1]
                y = TrainData[i][-1]
                NeuronsOutputs = self.Feedforward(x, weights, ActivationFun, BiasCheck)
                weights = self.BackPropagate(x, y, weights, NeuronsOutputs, ActivationFun, eta, BiasCheck, NumHiddenLayers)

        return weights

    def Feedforward(self, x, weights, active_func, bias_check):
        neurons_outputs = []  # list of neurons outputs from activation function
        for i in range(len(weights)):
            AllNetValues = np.dot(weights[i], x)   # list
            Y_Activaton = []
            if bias_check and (i != len(weights) - 1):
                Y_Activaton.append(1)
            for n in AllNetValues:
                if active_func == 1:
                    output = self.Sigmoid(n)
                    Y_Activaton.append(output)
                else:
                    output = self.TanH(n)
                    Y_Activaton.append(output)

            x = Y_Activaton
            neurons_outputs.append(Y_Activaton)

        return neurons_outputs

    def OutputLayerError(self, y, revesoutput, ActivationFun):
        error = []
        for i in range(len(y)):
            y_acual = y[i]  # y_acual
            y_predc = revesoutput[i]  # y_predc

            if ActivationFun:
                e = (y_acual - y_predc) * y_predc * (1 - y_predc)
                error.append(e)
            else:
                e = (y_acual - y_predc) * (1 - (y_predc*y_predc))
                error.append(e)
        return error

    def HiddenLayerError(self, NumHiddenLayer, Nexterror, NodeWeights, revesoutput, ActivationFun, bias):
        error = []
        for i in range(NumHiddenLayer):
            if bias == 1:
                NodeWeights[i] = NodeWeights[i][:, 1:]

            shape = NodeWeights[i].shape
            w = np.reshape(NodeWeights[i], (shape[1], shape[0]))
            e = np.dot(w, Nexterror)

            Y_Act = []
            if ActivationFun:
                for c in range(len(revesoutput[i+1])):
                   tmp = revesoutput[i+1][c] * (1-revesoutput[i+1][c])
                   Y_Act.append(tmp)

                if bias == 1:
                    Y_Act.pop(0)

                Final_E = e*Y_Act
                error.append(Final_E)
            else:
                for c in range(len(revesoutput[i+1])):
                   tmp = 1 - (revesoutput[i + 1][c] * revesoutput[i + 1][c])
                   Y_Act.append(tmp)

                if bias == 1:
                    Y_Act.pop(0)

                Final_E = e * Y_Act
                error.append(Final_E)
            Nexterror = Final_E
        return error

    def BackPropagate(self, x, y, weights, outputs, activate_func, eta, bias_check, NumHiddenLayer):
        output_layer = outputs[-1]
        finalf = []
        for value in output_layer:
            e = np.max(outputs[-1])
            if value == e:
                finalf.append(1)
            else:
                finalf.append(0)

        check = 0
        for i, j in zip(y, finalf):
            if i == j:
                check += 1

        if len(y) != check:
            output2 = outputs.copy()  # output reversed
            output2.reverse()

            # here start calc error in output layer
            OutputNodeError = self.OutputLayerError(y, output2[0], activate_func)

            weight2 = weights.copy()  # weight reversed
            weight2.reverse()
            NodeWeights = weight2.copy()

            output2.append(x)

            # here start calc error in output layer
            Nodes_error = self.HiddenLayerError(NumHiddenLayer, OutputNodeError, NodeWeights, output2, activate_func, bias_check)
            count=0
            for Layer in range(len(output2) - 1):  # start from hidden layer which before output layer
                if Layer == 0:
                    for j in range(len(output2[Layer + 1])):
                        XInput = output2[Layer + 1][j]
                        for k in range(len(output2[Layer])):
                            weight2[Layer][k, j] = weight2[Layer][k, j] + eta * OutputNodeError[k] * XInput
                else:
                    LayerError = []
                    LayerErrorTmp = Nodes_error[count]
                    for g in range(len(LayerErrorTmp)):
                        LayerError.append(LayerErrorTmp[g])

                    PrvLayer = len(output2[Layer])

                    if bias_check == 1:
                        LayerError.append(1)
                        PrvLayer = PrvLayer - 1

                    for j in range(len(output2[Layer + 1])):
                        XInput = output2[Layer + 1][j]
                        for k in range(PrvLayer):
                            weight2[Layer][k, j] = weight2[Layer][k, j] + eta * LayerError[k] * XInput
                    count += 1
                #print("انتش واجري ")

            weights = weight2.copy()
            weights.reverse()

        return weights