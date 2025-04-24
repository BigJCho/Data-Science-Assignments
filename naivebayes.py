# Manually compute class pr iors, and mean and variance per class
# Sum logs of likelihoods + log prior. Higher value is the class
import numpy as np
import pandas as pd

# We calculate the guassian distribution for both categories for every feature

def big_gauss(x, mean, var):
    coeff = -0.5 * np.log(2.0 * np.pi * var)
    exponent = -((x - mean) ** 2) / (2.0 * var)
    return coeff + exponent

file_path = 'spambase/spambase.data'
df = pd.read_csv(file_path, header=None)

# Calculating mean, variance, and priors
spam = df[df[57] == 1]
not_spam = df[df[57] == 0]
prior_spam  = len(spam) / len(df)
prior_not_spam = 1 - prior_spam

s_means = []
s_vars = []
ns_means = []
ns_vars = []

for i in range(57):
    s_means.append(spam[i].mean())
    s_vars.append(spam[i].var())
    ns_means.append(not_spam[i].mean())
    ns_vars.append(not_spam[i].var())
s_means = np.array(s_means)
s_vars = np.array(s_vars)
ns_means = np.array(ns_means)
ns_vars = np.array(ns_vars)

# Classifying the dataset using our calculated values
email = df.iloc[:, :-1].values
label = df.iloc[:, -1].values

correct = 0

for i in range(len(email)):
    x = email[i]
    prob_spam = np.sum(big_gauss(x, s_means, s_vars)) + np.log(prior_spam)
    prob_not_spam = np.sum(big_gauss(x, ns_means, ns_vars)) + np.log(prior_not_spam)

    prediction = 1 if prob_spam > prob_not_spam else 0
    if prediction == label[i]:
        correct += 1

accuracy = correct / len(email)
print(f'Accuracy: {accuracy}')