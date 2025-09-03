# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:13:42 2025

@author: p-sik
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def dataset(features, valid=False):
    """
    Splits the input dataset into training, testing, and optionally validation 
    sets.
    
    Parameters:
    -----------        
    features : pandas.DataFrame
        A DataFrame containing the feature data. The DataFrame must include the 
        target column 'Class' and other feature columns.
        
    valid : bool, optional, default=False
        If True, splits the test set further into a validation set. If False, 
        only splits into training and testing sets.
    
    Returns:
    --------
    X_train : pandas.DataFrame
        The training features.
        
    X_test : pandas.DataFrame
        The testing features.
        
    y_train : pandas.Series
        The target variable (Class) for the training set.
        
    y_test : pandas.Series
        The target variable (Class) for the testing set.
        
    X_valid : pandas.DataFrame, optional
        The validation features, returned only if valid=True.
        
    y_valid : pandas.Series, optional
        The target variable (Class) for the validation set, returned only if 
        valid=True.
    """
    # Drop metadata columns safely
    X = features.drop(columns=['X', 'Y', 'Class', 'Note', 'imID'], errors='ignore')
    y = features['Class']

    # Only stratify if every class has >= 2 samples
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        print("Warning: some classes have <2 samples, skipping stratify.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=stratify, random_state=42
    )

    if valid:
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test, y_test, stratify=y_test if stratify is not None else None, random_state=42
        )
        return X_train, X_test, y_train, y_test, X_valid, y_valid
    else:
        return X_train, X_test, y_train, y_test


def get_optimal_rfc(X_train, y_train, param_dist=None):
    """
    Performs a random search to find the optimal hyperparameters for a Random 
    Forest Classifier.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data (features).
        
    y_train : pandas.Series or numpy.ndarray
        Target labels corresponding to the training data.
        
    param_dist : dict or None, optional, default=None
        A dictionary of parameter distributions for random search. If None, 
        default parameter distributions are used. The dictionary should specify 
        the hyperparameters and their possible values (or distributions). 
        For example:
            - n_estimators
            - max_depth
            - min_samples_split
            - min_samples_leaf
            - max_features
    
    Returns:
    --------
    estimator : RandomForestClassifier
        The best model found by random search.
    """

    # Initialize Random Forest Classifier with balanced class weights
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Default parameter distribution if none is provided
    if param_dist is None:
        from scipy.stats import randint
        
        param_dist = {
            # Number of trees
            'n_estimators': randint(10, 100),  
            # Max depth of trees
            'max_depth': randint(5, 50), 
            # Min samples to split an internal node
            'min_samples_split': randint(2, 20),  
            # Min samples at a leaf node
            'min_samples_leaf': randint(1, 20), 
            # Features to consider at each split
            'max_features': [4, 5, 6, 7, 8, 9, 10]  
        }

    # Set up RandomizedSearchCV with the specified parameters
    random_search = RandomizedSearchCV(
        # RandomForestClassifier as the estimator
        estimator=rf,  
        # Parameter distributions to search over
        param_distributions=param_dist,  
        # Number of random combinations to try
        n_iter=100, 
        # Use balanced accuracy as scoring metric
        scoring='balanced_accuracy',  
        # 5-fold cross-validation
        cv=5,  
        # Set random seed for reproducibility
        random_state=42,  
        # Use all available CPU cores
        n_jobs=-1,  
        # Print detailed progress messages
        verbose=1  
    )

    # Run the random search with the training data
    random_search.fit(X_train, y_train)

    # Get the best estimator (model) from the random search
    estimator = random_search.best_estimator_

    # Print the best parameters found and the accuracy on the training set
    print("Best parameters found:", random_search.best_params_)
    print("Train set accuracy:", estimator.score(X_train, y_train))
    
    return estimator


def select_features(X_train, y_train, num=5, estimator=None):
    """
    Selects the most important features using forward Sequential Feature 
    Selection (SFS) based on a given estimator (default: RandomForestClassifier).

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature data.

    y_train : pandas.Series or numpy.ndarray
        Target labels corresponding to the training data.
        
    num : integer, optional (default=5)
        Number of features to be selected.

    estimator : sklearn estimator, optional (default=None)
        The model used to evaluate feature importance during selection. If None,
        a RandomForestClassifier with class_weight='balanced' will be used.

    Returns:
    --------
    selected_features : pandas.Index
        Names of the selected features.
    """
    
    # Use RandomForestClassifier by default if no estimator is provided
    if estimator is None:
        estimator = RandomForestClassifier(random_state=42, 
                                           class_weight='balanced')

    # Create a Sequential Feature Selector (forward selection)
    sfs = SequentialFeatureSelector(
        # model to evaluate features
        estimator=estimator,  
        # number of features to select           
        n_features_to_select=num,  
        # start from no features and add one at a time
        direction='forward', 
        # metric to optimize            
        scoring='balanced_accuracy',  
        # 5-fold cross-validation
        cv=5, 
        # use all CPU cores                           
        n_jobs=-1                         
    )

    # Fit the selector on training data
    sfs.fit(X_train, y_train)

    # Retrieve the names of the selected features
    selected_features = X_train.columns[sfs.get_support()]
    
    print("Selected features:", list(selected_features))
    
    return selected_features
    

def fitting(X_train, y_train, estimator, reports=True, sfeatures=None):
    """
    Fits a classifier to the training data and reports performance metrics.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Feature matrix for training.
        
    y_train : array-like
        Target labels for training.
        
    estimator : sklearn-like classifier
        Any object with `.fit()` and `.predict()` methods
        
    reports : bool, default=True
        If True, prints training accuracy and classification report.
        
    sfeatures : list of str or list of int, optional
        Subset of features to use for training. If None, all features are used.
        
    Returns:
    --------
    estimator : object
        The trained classifier.
        
    y_pred : array-like
        Predicted labels for the training data.
    """
    
    # Select specific features if provided
    if sfeatures is not None:
        try:
            X_train = X_train[sfeatures]
        except Exception as e:
            raise ValueError(f"Invalid feature selection: {e}")
    
    # Fit the classifier to the training data
    estimator.fit(X_train, y_train)
    
    # Predict the labels on the training set
    y_pred = estimator.predict(X_train)
    
    if reports:
        # Print training accuracy
        print("Train accuracy with selected features:",
              accuracy_score(y_train, y_pred))
        
        # Print classification report
        print("\nClassification train report:\n")
        print(classification_report(y_train, y_pred))
    
        # Plot confusion matrix
        cm = confusion_matrix(y_train, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
    
    return estimator, y_pred


def predicting(X_test, estimator, sfeatures=None, y_test=None):
    """
    Uses a trained classifier to make predictions on test data and optionally 
    evaluates performance.
    
    Parameters:
    -----------
    X_test : pd.DataFrame or np.ndarray
        Feature matrix for testing.
        
    estimator : sklearn-like classifier
        Trained model with a `.predict()` method.
        
    sfeatures : list of str or list of int, optional 
        Subset of features to use for prediction. If None, all features are used.
        
    y_test : array-like, optional
        Ground truth labels. If provided, accuracy and a classification report 
        will be printed.
        
        
    Returns:
    --------
    y_pred : array-like
        Predicted labels for the test data.
    """
    
    # Select specific features if provided
    if sfeatures is not None:
        try:
            X_test = X_test[sfeatures]
        except Exception as e:
            raise ValueError(f"Invalid feature selection: {e}")
    
    # Predict the labels
    y_pred = estimator.predict(X_test)
    
    if y_test is not None:
        # Print test accuracy
        print("Test accuracy with selected features:",
              accuracy_score(y_test, y_pred))
        
        # Print classification report
        print("\nClassification test report:\n")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix if requested
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Greens')
    
    return y_pred
            
