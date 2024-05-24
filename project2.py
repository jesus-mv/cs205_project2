import numpy as np 
import random
import copy

# make it so that a user can input a file 
base_file_path = 'data/'

data = np.loadtxt(base_file_path + 'CS205_small_Data__1.txt')
data_subset = data[:, :5]

num_of_classes = data_subset.shape[1]

# might need to turn these into arrays but keep set for now
local_best_feature_set = set()

global_best_feature_set = set()
global_best_accuracy = 0

#todo...
def k_fold_cross_validation(data_subset, best_features, test_feature):
    return random.randint(0, 100)

# search done, code evaluation function
print("Beginning Search.") #indexing is weird 
for i in range(0, num_of_classes - 1):
    local_best_feature = None
    local_best_accuracy = 0
    for curr_feature in range(1, num_of_classes): # features 1, 2, 3, ...
        if curr_feature not in local_best_feature_set:
            curr_accuracy = k_fold_cross_validation(data_subset, local_best_feature_set, curr_feature) #? 
            temp = copy.deepcopy(local_best_feature_set)
            temp.add(curr_feature)
            print("Using feature(s)", temp, "accuracy is", str(curr_accuracy) + "%")

            if (curr_accuracy > local_best_accuracy):
                local_best_accuracy = curr_accuracy
                local_best_feature = curr_feature

            if (curr_accuracy > global_best_accuracy):
                global_best_accuracy = curr_accuracy
                global_best_feature_set = temp
                
    local_best_feature_set.add(local_best_feature)

    if (i != num_of_classes - 2):
        print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy) + "%")

print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy) + "%")
 
