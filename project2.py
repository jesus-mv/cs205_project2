import numpy as np 
import copy

# example datasets:
# CS205_small_Data__19.txt: features [6, 9] with accuracy of 0.95
# CS205_large_Data__6.txt : features [1, 29] with accuracy of 0.97

# my datasets:
# CS205_small_Data__23.txt: features [2, 6] with accuracy of 0.954
# CS205_large_Data__2.txt : features [1, 8] with accuracy of 0.9736

def main():
    data, num_of_features, feature_list, input_mode = input_sequence()
    
    if (input_mode == 1):
        forward_selection_search(data, num_of_features)
    elif (input_mode == 2):
        backward_elimination_search(data, num_of_features, feature_list)
    else:
        print("Invalid input!")
        return -1 # error

    return 0

# get file input, read data, and determine if user would like to 
# do forward selection or backward elimination
def input_sequence():
    print("Welcome to Jesus Martinez Vega's Feature Selection Algorithm.")
    raw_input_file = input("Type in the name of the file to test: ")
    input_file = str(raw_input_file)
    raw_input_mode = input("\nType the number of the algorithm you want to run.\n    1) Forward Selection\n    2) Backward Elimination\n")
    input_mode = int(raw_input_mode)

    base_file_path = 'data/'
    data = np.loadtxt(base_file_path + input_file)
    num_of_features = data.shape[1] - 1
    num_of_instances = data.shape[0]

    print("\nThis dataset has " + str(num_of_features) + " features (not including the class attribute), with " + str(num_of_instances) + " instances\n")

    feature_list = []
    for i in range(1, num_of_features + 1):
        feature_list.append(i)

    accuracy = k_fold_cross_validation(data, feature_list, None, 0) 

    print("Running nearest neighbor with all " + str(num_of_features) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(accuracy) + "\n")       

    return data, num_of_features, feature_list, input_mode

# forward selection search 
def forward_selection_search(data, num_of_features):
    accuracy_decrease = False
    local_best_feature_set = []

    global_best_feature_set = []
    global_best_accuracy = 0

    print("Beginning Search.\n") 
    for i in range(0, num_of_features): # for each feature...
        local_best_feature = None
        local_best_accuracy = 0
        for curr_feature in range(1, num_of_features + 1): # curr_features 1, 2, 3, ... 
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

        print("")
        if (local_best_accuracy < global_best_accuracy and not accuracy_decrease):
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            accuracy_decrease = True
        elif (local_best_accuracy > global_best_accuracy and accuracy_decrease):
            print("(Accuracy has increased! Continuing search Continuing search in case of local maxima)")
            accuracy_decrease = False

        if (i != num_of_features - 1):
            print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy), "\n")

    print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))

# backward elimination search
def backward_elimination_search(data, num_of_features, feature_list):
    accuracy_decrease = False
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
        for curr_feature in range(1, num_of_features + 1): # curr_features 1, 2, 3, ... 
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

        # so that an empty feature set isnt evaluated
        if (len(local_best_feature_set) == 1):
            print("")
            break

        print("")
        if (local_best_accuracy < global_best_accuracy and not accuracy_decrease):
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            accuracy_decrease = True
        elif (local_best_accuracy > global_best_accuracy and accuracy_decrease):
            print("(Accuracy has increased! Continuing search Continuing search in case of local maxima)")
            accuracy_decrease = False

        if (i != num_of_features - 1):
            print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy), "\n")

    print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))


# k fold cross validation with k = 1 using nearest neighbor 
# mode == 0 means forwards selection, mode == 1 means backwards elimination
def k_fold_cross_validation(data, test_features, test_feature, mode):
    
    num_correct = 0

    temp_test_features = copy.deepcopy(test_features)
    if (test_feature != None): # if none, then just want to test features as is, no need to remove/add feature.
        if (mode == 0):
            temp_test_features.append(test_feature) # give the dataset a try with test_feature
        elif (mode == 1):
            temp_test_features.remove(test_feature) # give the dataset a try without test_feature
        else:
            return -1 # error
        
    # temp_test_features needs to be ordered from smallest to largest (and needs to include 0)
    # to properly slice the dataset
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
        # done this way so that sorted indexes arent messed up
        distances = np.linalg.norm(new_data[:, 1:] - data_to_classify, axis=1)

        # sort distances without modifying array, returns index of closest neighbour (not including self)
        nearest_neighbor_idx = distances.argsort()[1]

        if (label_of_data == new_data[nearest_neighbor_idx, 0]):
            num_correct += 1

    accuracy = num_correct / new_data.shape[0]
    return accuracy
    
if __name__ == "__main__":
    main()