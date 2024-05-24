import numpy as np 
import copy

#general issues: really, really slow on large datasets 
#                small data 19 picks features [6, 9] with
#                acc of 95% and not [6, 9, 11] with acc 0.946

# todo: make it so that a user can input a file
base_file_path = 'data/'

data = np.loadtxt(base_file_path + 'CS205_small_Data__19.txt')
num_of_classes = data.shape[1] - 1

local_best_feature_set = []

global_best_feature_set = []
global_best_accuracy = 0

# todo... (?)
def k_fold_cross_validation(data, best_features, test_feature):
    
    num_correct = 0

    temp_best_features = copy.deepcopy(best_features)
    temp_best_features.append(test_feature)
    temp_best_features.insert(0, 0) 
    temp_best_features.sort()

    # makes a new dataset with only best_features + test_features 
    new_data = data[:, temp_best_features]
    

    for i in range(new_data.shape[0]):
        data_to_classify = new_data[i, 1:]
        label_of_data = new_data[i, 0]

        nn_distance = np.inf
        for j in range(new_data.shape[0]):
            if i != j:
                distance = np.sqrt(np.sum((data_to_classify - new_data[j, 1:]) ** 2))
                if (distance < nn_distance):
                    nn_distance = distance
                    nn_label = new_data[j, 0]

        if label_of_data == nn_label:
            num_correct += 1

    accuracy = num_correct / new_data.shape[0]
    return accuracy
    
# todo: backward elimination
print("Beginning Search.") 
for i in range(0, num_of_classes): #for each class... 
    local_best_feature = None
    local_best_accuracy = 0
    for curr_feature in range(1, num_of_classes + 1): # features 1, 2, 3, ... 
        if curr_feature not in local_best_feature_set:
            curr_accuracy = k_fold_cross_validation(data, local_best_feature_set, curr_feature) #? 
            temp = copy.deepcopy(local_best_feature_set)
            temp.append(curr_feature)
            temp.sort()
            print("Using feature(s)", temp, "accuracy is", str(curr_accuracy))

            if (curr_accuracy > local_best_accuracy):
                local_best_accuracy = curr_accuracy
                local_best_feature = curr_feature

            if (curr_accuracy > global_best_accuracy):
                global_best_accuracy = curr_accuracy
                global_best_feature_set = temp
                
    local_best_feature_set.append(local_best_feature)
    local_best_feature_set.sort()

    if (i != num_of_classes - 1):
        print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy))

print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))
