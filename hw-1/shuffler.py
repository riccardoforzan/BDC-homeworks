import random

if __name__ == "__main__":

    lines = open('datasets/facebook_large.txt').readlines()
    random.shuffle(lines)
    open('shuffle_facebook_large.txt', 'w').writelines(lines)
