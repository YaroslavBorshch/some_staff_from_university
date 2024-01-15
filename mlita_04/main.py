import convolution
import criteria
import pareto
import saaty

mgts = (5, 5, 5, 3)
rustelecom = (6, 3, 3, 4)
akado = (2, 2, 4, 2)
qwerty = (3, 4, 4, 5)
choices = (mgts, rustelecom, akado, qwerty)
labels = ("МГТС", "Ростелеком", "Акадо", "QWERTY")
dirs = (-1, 1, 1, 1)

limits = (0.5, 0.2, 0.5)
main_criteria_choice = criteria.main_criteria(choices, dirs, 0, limits,
                                                   labels)
print("Main criteria choice:", labels[main_criteria_choice])
print()

pareto_choice = pareto.pareto(choices, dirs, 1, 3, labels)
print("Pareto choice:", labels[pareto_choice])
print()

weights = (-4, 5, 3, 6)
convolution_choice = convolution.convolution(choices, weights)
print("Convolution choice:", labels[convolution_choice])
print()

saaty_choice = saaty.saaty(choices, weights, labels)
print("Saaty choice:", labels[saaty_choice])