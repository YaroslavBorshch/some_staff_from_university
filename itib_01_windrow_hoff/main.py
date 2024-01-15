from neuro import NeuronWithUnitStepFunction, NeuroneWithSigmoidActivationFunction

if __name__ == '__main__':
    print("Обучение с пороговой функцией активации")
    n = NeuronWithUnitStepFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    n.train()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Обучение с сигмоидальной функцией активации")
    n2 = NeuroneWithSigmoidActivationFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    n2.train()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Обучение с сигмоидальной функцией активации и неполной выборкой")
    n3 = NeuroneWithSigmoidActivationFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    n3.train_partly()
