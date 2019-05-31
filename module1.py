import pandas as pd
from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass
def download_data():
    frame = pd.read_csv("Stocks\WYNN.csv")
    del frame['Open']
    del frame['High']
    del frame['Low']
    del frame['Adj Close']
    del frame['Volume']
    frame['Date']=frame['Date'].str.replace("-","").astype(int)
    frame['Date']=(frame['Date']-(frame['Date']/10000).astype(int)*10000)
    frame['Moving']= frame['Close'].rolling(window=2).mean()
    frame['Close']
    return frame
def get_features_and_labels(frame):


    # Replace missing values with 0.0
    frame[frame.isnull()] = 0.0

    arr = np.array(frame, dtype=np.float)

    # Normalize the entire data set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    arr = MinMaxScaler().fit_transform(arr)


    X, y = arr[:, :-1], arr[:, -1]

    
    # Use 50% of the data for training, but we will test against the
    # entire set
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5)
    X_test, y_test = X, y
    
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test



def evaluate_learner(X_train, X_test, y_train, y_test):

    # Use a support vector machine for regression
    from sklearn.svm import SVR

    # Train using a radial basis function
    svr = SVR(kernel='rbf', gamma=0.1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'RBF Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a linear kernel
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Linear Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a polynomial kernel
    svr = SVR(kernel='poly', degree=2)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Polynomial Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred


# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, expected values, actual values)
    
    All the elements in results will be plotted.
    '''

    # Using subplots to display the results on the same X axis
    fig, plts = plt.subplots(nrows=len(results), figsize=(8, 8))
    fig.canvas.set_window_title('Predicting data from ')

    # Show each element in the plots returned from plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        # Configure each subplot to have no tick marks
        subplot.set_xticklabels(())
        subplot.set_yticklabels(())

        subplot.set_ylabel('WYNN Stock Price')

        subplot.set_title(title)

        subplot.plot(y, 'b', label='actual')
        subplot.plot(y_pred, 'r', label='predicted')
        
        # Shade the area between the predicted and the actual values
        subplot.fill_between(
            np.arange(0, len(y), 1),
            y,
            y_pred,
            color='r',
            alpha=0.2
        )

        subplot.axvline(len(y) // 2, linestyle='--', color='0', alpha=0.2)

        subplot.legend()

    fig.tight_layout()

    plt.show()

    plt.savefig('plot.png')

  


if __name__ == '__main__':
    frame = download_data()
     

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple regression learners on the data
    print("Evaluating regression learners")
    results = list(evaluate_learner(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)


