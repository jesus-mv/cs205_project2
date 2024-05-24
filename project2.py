import numpy as np 
import copy

#general issues: really, really slow on large datasets 
#                small data 19 picks features [6, 9] with
#                acc of 95% and not [6, 9, 11] with acc 0.946

def main():
    data, num_of_features, feature_list, input_mode = input_sequence()

    if (input_mode == 1):
        forward_selection_search(data, num_of_features)
    elif (input_mode == 2):
        backward_elimination_search(data, num_of_features, feature_list) #todo
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

    accuracy = k_fold_cross_validation(data, feature_list, None)

    print("Running nearest neighbor with all " + str(num_of_features) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(accuracy))    

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

        if (i != num_of_features - 1):
            print("feature set", local_best_feature_set, "was best, accuracy is", str(local_best_accuracy))

    print("Finished search!! The best feature subset is", global_best_feature_set, "which has an accuracy of", str(global_best_accuracy))

#todo 
def backward_elimination_search(data, num_of_features, feature_list):
    return 

# k fold cross validation with k = 1 using nearest neighbor 
def k_fold_cross_validation(data, best_features, test_feature):
    
    num_correct = 0

    temp_best_features = copy.deepcopy(best_features)
    if (test_feature != None):
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
    
if __name__ == "__main__":
    main()