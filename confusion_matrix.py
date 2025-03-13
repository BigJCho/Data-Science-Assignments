import pandas as pd

#Manually defined weights and bias
#Further development of this can allow for input from csv file on command line
weights = [24, -15, -38, -7, -41, 35, 0, -2, 19, 33, -3, 7, 3, -47, 26, 10, 40, -1, 3, 0]
bias = 500
results = {'true_positive': 0 , 'false_positive' : 0, 'true_negative' : 0, 'false_negative' : 0}
pos_buckets = { 0: 0, 100 : 0, 200 : 0, 300 : 0, 400 : 0, 'rest' : 0}
neg_buckets = { 0: 0, 100 : 0, 200 : 0, 300 : 0, 400 : 0, 'rest' : 0}
#Read in file, change as necessary
#Further development can allow for taking filename as a command line argument
src = pd.read_csv('hw3.data1.csv')

for index, row in src.iterrows():
    #Zip will stop at 20 columns due to weights defined above
    y = sum(v * w for v, w in zip(row, weights)) + bias
    #Store the label from the model
    label = row.iloc[20]
    #I loved ternary operators when learning Java, saw that Python had it too when working on another assignment
    calculated_label = 1 if y > 0 else -1
    if calculated_label == label:
        if label == 1:
            results['true_positive'] += 1
        else:
            results['true_negative'] += 1
    else:
        if label == 1:
            results['false_negative'] += 1
        else:
            results['false_positive'] += 1
    if y > 0:
        match y:
            case n if n < 100:
                pos_buckets[0] += 1
            case n if n < 200:
                pos_buckets[100] += 1
            case n if n < 300:
                pos_buckets[200] += 1
            case n if n < 400:
                pos_buckets[300] += 1
            case n if n < 500:
                pos_buckets[400] += 1
            case _:
                pos_buckets['rest'] += 1
    else:
        match y:
            case n if n > -100:
                neg_buckets[0] += 1
            case n if n > -200:
                neg_buckets[100] += 1
            case n if n > -300:
                neg_buckets[200] += 1
            case n if n > -400:
                neg_buckets[300] += 1
            case n if n > -500:
                neg_buckets[400] += 1
            case _:
                neg_buckets['rest'] += 1
        
    
accuracy = (results['true_positive'] + results['true_negative']) / (results['true_negative'] + results['true_positive'] + results['false_negative'] + results['false_positive'])
cost_fp = 100
cost_fn = 1000
economic_gain = (results['false_negative'] * -cost_fn) + (results['false_positive'] * -cost_fp)
print(results)
print(f"Accuracy: {accuracy}")
print(f"Economic gain: {economic_gain}")
print(f"Negative buckets: {neg_buckets}")
print(f"Positive buckets: {pos_buckets}")
    