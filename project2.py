import numpy as np 
import copy
import random

#general issues: 
# small data 19 picks features [6, 9] with
# acc of 95% and not [6, 9, 11] with acc 0.946

# large data 6 picks features [1, 29] with acc
#  of 0.978 and not [1, 4, 29] with acc 0.97 

# both scenarios do give the same accuracies as given 

def main():
    data, num_of_features, feature_list, input_mode = input_sequence()
    
    if (input_mode == 1):
        forward_selection_search(data, num_of_features)
    elif (input_mode == 2):
        backward_elimination_search(data, num_of_features, feature_list)
    else:
        print("Invalid input!")
    
    

def input_sequence():
    print("Welcome to Jesus Martinez Vega's Feature Selection Algorithm.")
    raw_input_file = input("Type in the name of the file to test: ")
    input_file = str(raw_input_file)
    raw_input_mode = input("Type the number of the algorithm you want to run.\n    1) Foward Selection\n    2) Backward Elminination\n")
    input_mode = int(raw_input_mode)

    base_file_path = 'data/'
    data = np.loadtxt(base_file_path + raw_input_file)
    num_of_features = data.shape[1] - 1
    num_of_instances = data.shape[0]

    print("This dataset has " + str(num_of_features) + " features (not including the class attribute), with " + str(num_of_instances) + " instances")

    feature_list = []
    for i in range(1, num_of_features + 1):
        feature_list.append(i)

    accuracy = k_fold_cross_validation(data, feature_list, None, 0) 

    print("new: Running nearest neighbor with all " + str(num_of_features) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(accuracy))       

    return data, num_of_features, feature_list, input_mode

def forward_selection_search(data, num_of_features):
    local_best_feature_set = []

    global_best_feature_set = []
    global_best_accuracy = 0

    print("Beginning Search.\n") 
    for i in range(0, num_of_features): #for each feature... 
        local_best_feature = None
        local_best_accuracy = 0
        for curr_feature in range(1, num_of_features + 1): # features 1, 2, 3, ... 
            if curr_feature not in local_best_feature_set:
                curr_accuracy = k_fold_cross_validation(data, local_best_feature_set, curr_feature, 0) 
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

        if (i != num_of_features - 1):
            print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy))

    print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))

#todo
def backward_elimination_search(data, num_of_features, feature_list):
    local_best_feature_set = feature_list

    global_best_feature_set = feature_list

    print("Beginning Search.\n") 

    # evaluate initial feature set 
    global_best_accuracy = k_fold_cross_validation(data, local_best_feature_set, None, 0) #4th parameter doesnt matter
    print("Using feature(s)", local_best_feature_set, "accuracy is", str(global_best_accuracy))
    print("feature set", local_best_feature_set, "was best, accuracy is", str(global_best_accuracy))

    for i in range(0, num_of_features): #for each feature... 
        local_best_feature = None
        local_best_accuracy = 0
        for curr_feature in range(1, num_of_features + 1): # features 1, 2, 3, ... 
            if curr_feature in local_best_feature_set:
                curr_accuracy = k_fold_cross_validation(data, local_best_feature_set, curr_feature, 1) 
                temp = copy.deepcopy(local_best_feature_set)
                temp.remove(curr_feature)
                temp.sort()
                print("Using feature(s)", temp, "accuracy is", str(curr_accuracy))

                if (curr_accuracy > local_best_accuracy):
                    local_best_accuracy = curr_accuracy
                    local_best_feature = curr_feature

                if (curr_accuracy > global_best_accuracy):
                    global_best_accuracy = curr_accuracy
                    global_best_feature_set = temp
                
        local_best_feature_set.remove(local_best_feature)
        local_best_feature_set.sort()

        if (len(local_best_feature_set) == 1):
            break

        if (i != num_of_features - 1):
            print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy))

    print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))


# k fold cross validation with k = 1 using nearest neighbor 
# mode == 0, forwards selection, mode == 1, backwards elimination
def k_fold_cross_validation(data, test_features, test_feature, mode):
    
    num_correct = 0

    temp_test_features = copy.deepcopy(test_features)
    if (test_feature != None): # just want to test features as is, no need to remove/add feature.
        if (mode == 0):
            temp_test_features.append(test_feature)
        elif (mode == 1):
            temp_test_features.remove(test_feature)
        else:
            return -1 # error
    temp_test_features.insert(0, 0) 
    temp_test_features.sort()

    # makes a new dataset with only test_features + test_feature
    new_data = data[:, temp_test_features]
    indicies = np.arange(new_data.shape[0])

    for i in range(new_data.shape[0]):
        data_to_classify = new_data[i, 1:] 
        label_of_data = new_data[i, 0]     

        # a bit strange, the data point that is supposed to be removed is actually kept in 
        # so the second nearest neighbor is taken
        distances = np.linalg.norm(new_data[:, 1:] - data_to_classify, axis=1)
        nearest_nneighbor_idx = distances.argsort()[1]

        if (label_of_data == new_data[nearest_nneighbor_idx, 0]):
            num_correct += 1

    accuracy = num_correct / new_data.shape[0]
    return accuracy
    
if __name__ == "__main__":
    main()