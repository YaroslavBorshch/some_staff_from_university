import neuro

if __name__ == '__main__':
    n = neuro.Network([2], [-0.3], 1, 2, 1, 0.3)
    n.learning()
