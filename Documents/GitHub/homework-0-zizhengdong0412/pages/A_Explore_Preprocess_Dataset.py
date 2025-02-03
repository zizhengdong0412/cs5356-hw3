import streamlit as st                  
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

feature_lookup = {
    'longitude':'**longitude** - longitudinal coordinate',
    'latitude':'**latitude** - latitudinal coordinate',
    'housing_median_age':'**housing_median_age** - median age of district',
    'total_rooms':'**total_rooms** - total number of rooms per district',
    'total_bedrooms':'**total_bedrooms** - total number of bedrooms per district',
    'population':'**population** - total population of district',
    'households':'**households** - total number of households per district',
    'median_income':'**median_income** - median income',
    'ocean_proximity':'**ocean_proximity** - distance from the ocean',
    'median_house_value':'**median_house_value**'
}

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

# Checkpoint 1
def load_dataset(filepath):
    """
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe storing the dataset
    """
    data=None

    # Write code here.
    data = pd.read_csv(filepath)
    return data

# Helper Function
def display_features(df,feature_lookup):
    """
    This function displays feature names and descriptions (from feature_lookup and dataset columns).
    
    Inputs:
    - df (pandas.DataFrame): The input DataFrame to with features to be displayed.
    - feature_lookup (dict): A dictionary containing the descriptions for the features.
    
    Outputs: None
    """
    # Write code here.
    for feature in df.columns:
       if feature in feature_lookup:
           print(f"{feature}: {feature_lookup[feature]}")
       else:
           print(f"{feature}: Not available.")

# Checkpoint 2
def sidebar_filter(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: chart type (options include: 'Scatterplots','Lineplots','Histogram','Boxplot')
        - x: features (Optional)
        - y: targets (Optional)
    Output: 
        - list of sidebar filters on features
    """
    side_bar_data = []

    # Write code here.
    st.sidebar.markdown("### Filter Data")
    x_f = st.sidebar.selectbox("X-Axis Feature", df.columns if x is None else [x])
    y_f = st.sidebar.selectbox("Y-Axis Feature", df.columns if y is None else [y])
    x_range = (df[x_f].min(), df[x_f].max())
    y_range = (df[y_f].min(), df[y_f].max())
    side_bar_data = [x_range, y_range]
    return side_bar_data

# Checkpoint 3
def summarize_missing_data(df, top_n=3):
    """
    This function summarizes missing values in the dataset

    Input: 
        - df: the pandas dataframe
        - top_n: (integer) top n features with missing values, default value is 3
    Output: 
        - out_dict: a dictionary containing the following keys and values: 
            - 'num_categories': counts the number of features that have missing values
            - 'average_per_category': counts the average number of missing values across features
            - 'total_missing_values': counts the total number of missing values in the dataframe
            - 'top_missing_categories': lists the top n features with missing values
    """
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': None}

    # Write code here.
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    num_cat = len(missing_values)
    total_mv = missing_values.sum()
    avg_per_cat = total_mv / len(df.columns)
    top_missing_cat = missing_values.nlargest(top_n).index.tolist()
    out_dict = {
        'num_categories': num_cat,
        'average_per_category': avg_per_cat,
        'total_missing_values': total_mv,
        'top_missing_categories': top_missing_cat
    }

    return out_dict

# Checkpoint 4
def remove_features(df,removed_features):
    """
    Remove the features in removed_features (list) from the input pandas dataframe df. 

    Input
    - df is dataset in pandas dataframe
    Output: 
    - df: pandas dataframe with features removed
    """
    X = None

    # Write code here.
    X = df.drop(columns=removed_features, errors='ignore')
    return X

# Checkpoint 5
def impute_dataset(df, impute_method):
    """
    Impute the dataset df with imputation method impute_method 
    including mean, median, zero values or drop Nan values in 
    the dataset (all numeric and string columns).

    Input
    - df is dataset in pandas dataframe
    - impute_method: string data type with imputation method. Options: 'Zero', 'Mean', 'Median','DropNans'
    Output
    - df: pandas dataframe with imputed feature
    """
    X = None

    # Write code here.
    X = df.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if impute_method == 'DropNans':
        X.dropna(inplace=True)

    elif impute_method == 'Zero':
        X[numeric_cols] = X[numeric_cols].fillna(0)

    elif impute_method == 'Mean':
        for col in numeric_cols:
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)

    elif impute_method == 'Median':
        for col in numeric_cols:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    return X

# Checkpoint 6
def one_hot_encode_feature(df, feature):
    """
    This function performs one-hot-encoding on the given features

    Input
        - df: the pandas dataframe
        - feature: the feature(s) to perform one-hot-encoding
    Output
        - df: dataframe with one-hot-encoded feature
    """
    
    # Write code here.
    if isinstance(feature, str):
       feature = [feature]
    df = pd.get_dummies(df, columns=feature, drop_first=False)

    return df

# Checkpoint 7
def integer_encode_feature(df, feature):
    """
    This function performs integer-encoding on the given features

    Input
        - df: the pandas dataframe
        - feature: the feature(s) to perform integer-encoding
    Output
        - df: dataframe with integer-encoded feature
    """
    
    # Write code here.
    df = df.copy()
    if isinstance(feature, str):
        feature = [feature]

    for col in feature:
        cat = df[col].dropna().unique().tolist()
        cat.sort()
        cat_to_int = {cat: i for i, cat in enumerate(cat)}
        df[col + "_int"] = df[col].map(cat_to_int)
    return df

# Checkpoint 8
def scale_features(df, features, scaling_method): 
    """
    Use the scaling_method to transform numerical features in the dataset dataframe. 

    Input
        - df: the pandas dataframe
        - features: list of features
        - scaling method is a string; Options include {'Standardization', 'Normalization', 'Log'}
    Output
        - Standarization: X_new = (X - mean)/STD
        - Normalization: X_new = (X - X_min)/(X_max - X_min)
        - Log: X_log = log(X)
    """
    X = None

    # Write code here.
    X = df.copy()

    for col in features:

        if scaling_method == 'Standardization':
            mean_val = X[col].mean()
            std_val = X[col].std()
            X[col + "_std"] = 0 if std_val == 0 else (X[col] - mean_val) / std_val

        elif scaling_method == 'Normalization':
            min_val = X[col].min()
            max_val = X[col].max()
            range_val = max_val - min_val
            X[col + "_norm"] = 0 if range_val == 0 else (X[col] - min_val) / range_val

        elif scaling_method == 'Log':
            X[col + "_log"] = np.log2(X[col])

    return X

# Checkpoint 9
def create_feature(df, math_select, math_feature_select, new_feature_name):
    """
    Create a new feature with name new_feature_name in dataset df with the 
    mathematical operation math_select (string) on features math_feature_select (list). 

    Input
        - df: the pandas dataframe
        - math_select: the math operation to perform on the selected features (string)
        - math_feature_select: list of up to two features to be performed mathematical operation on
        - new_feature_name: the name for the new feature (string)
    Output
        - df: the dataframe with a new feature
    """
    
    # Write code here.
    df = df.copy()
    if len(math_feature_select) == 1:
        col = math_feature_select[0]

        if math_select == 'log':
            df[new_feature_name] = np.log(df[col])

        elif math_select == 'square root':
            df[new_feature_name] = np.sqrt(df[col])

        elif math_select == 'ceil':
            df[new_feature_name] = np.ceil(df[col])

        elif math_select == 'floor':
            df[new_feature_name] = np.floor(df[col])

    elif len(math_feature_select) == 2:
        col1, col2 = math_feature_select

        if math_select == 'add':
            df[new_feature_name] = df[col1] + df[col2]

        elif math_select == 'subtract':
            df[new_feature_name] = df[col1] - df[col2]

        elif math_select == 'multiply':
            df[new_feature_name] = df[col1] * df[col2]

        elif math_select == 'divide':
            df[new_feature_name] = df[col1] / df[col2]
    else:
        raise ValueError("Redo")
    return df

# Checkpoint 10
def remove_outliers(df, feature, outlier_removal_method=None):
    """
    This function removes the outliers from feature(s) using interquartile (IRQ) and standard deviation (STD). 

    Input
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    dataset = None
    lower_bound = None
    upper_bound = None
    
    # Write code here.
    dataset = df.copy()
    if isinstance(feature, str):
        feature = [feature]

    for col in feature:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1
        upper_bound = Q3

        method = outlier_removal_method if outlier_removal_method else 'IQR'
        method = method.upper()

        if method == 'IQR':
            lower_cut = Q1 - 1.5 * IQR
            upper_cut = Q3 + 1.5 * IQR
            dataset = dataset[(dataset[col] >= lower_cut) & (dataset[col] <= upper_cut)]

        elif method == 'STD':
            mean_val = dataset[col].mean()
            std_val = dataset[col].std()
            lower_cut = mean_val - 3 * std_val
            upper_cut = mean_val + 3 * std_val
            dataset = dataset[(dataset[col] >= lower_cut) & (dataset[col] <= upper_cut)]

    return dataset, lower_bound, upper_bound

## Checkpoint 11
def compute_descriptive_stats(df, stats_feature_select, stats_select):
    """
    Compute descriptive statistics stats_select on a feature stats_feature_select 
    in df. Statistics stats_select include mean, median, max, and min. Return 
    the results in an output string out_str and dictionary out_dict (dictionary).

    Input: 
    - df: the pandas dataframe
    - stats_feature_select: list of features to computer statistics on
    - stats_select: list of mathematical operations
    Output: 
    - output_str: string used to display feature statistics
    - out_dict: dictionary of feature statistics
    """
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }

    # Write code here.
    col = stats_feature_select[0]
    stats_select_lower = [s.lower() for s in stats_select]
    if 'mean' in stats_select_lower:
        out_dict['mean'] = round(df[col].mean(), 2)
    if 'median' in stats_select_lower:
        out_dict['median'] = round(df[col].median(), 2)
    if 'max' in stats_select_lower:
        out_dict['max'] = round(df[col].max(), 2)
    if 'min' in stats_select_lower:
        out_dict['min'] = round(df[col].min(), 2)

    output_str = f"Stats for feature '{col}':\n"
    for stat in ['mean', 'median', 'max', 'min']:
        if out_dict[stat] is not None:
            output_str += f"  {stat} = {out_dict[stat]}\n"
    return output_str, out_dict

# Checkpoint 12
def compute_correlation(df, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings

    Input
        - df: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = None
    cor_summary_statements = []

    # Write code here.
    correlation = df[features].corr()
    strong_threshold = 0.7

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            corr_val = correlation.loc[f1, f2]
            if corr_val >= 0:
                sign_str = "positively"
            else:
                sign_str = "negatively"
            if abs(corr_val) >= strong_threshold:
                strength_str = "strongly"
            else:
                strength_str = "weakly"
            statement = (f"- Features {f1} and {f2} are "
                            f"{strength_str} {sign_str} correlated: {corr_val:.3f}")

            cor_summary_statements.append(statement)

    return correlation, cor_summary_statements

###################### FETCH DATASET #######################

df=None

filename = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    if(filename):
        df = load_dataset(filename)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('### 1. Explore Dataset Features')

    # Display feature names and descriptions (from feature_lookup)
    display_features(df,feature_lookup)
    
    # Display dataframe as table
    st.dataframe(df)

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Features')

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots','Lineplots','Histogram','Boxplot']
    )

    # Draw plots
    if chart_select == 'Scatterplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.scatter(data_frame=df,
                                x=x_values, y=y_values,
                                range_x=[side_bar_data[0][0],
                                        side_bar_data[0][1]],
                                range_y=[side_bar_data[1][0],
                                        side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,
                                range_x=[side_bar_data[0][0],
                                            side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.line(df,
                            x=x_values,
                            y=y_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]],
                            range_y=[side_bar_data[1][0],
                                    side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.box(df,
                            x=x_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)

    if st.sidebar.button('Clip feature from %.2f to %.2f' % (side_bar_data[0][0], side_bar_data[0][1])):
        df[x_values+'_clipped'] = df[x_values]
        df[df[x_values+'_clipped']>side_bar_data[0][1]] = 0
        df[df[x_values+'_clipped']<side_bar_data[0][0]] = 0
        st.sidebar.write(x_values + ' cliped from '+str(side_bar_data[0][0])+' to '+str(side_bar_data[0][1]))
        if(chart_select == 'Scatterplots' or chart_select == 'Lineplots'):
            df[y_values+'_clipped'] = df[y_values]
            df[df[y_values+'_clipped']>side_bar_data[1][1]] = 0
            df[df[y_values+'_clipped']<side_bar_data[1][0]] = 0
            st.sidebar.write(y_values + ' cliped from '+str(side_bar_data[1][0])+' to '+str(side_bar_data[1][1]))

    # Display original dataframe
    st.markdown('## 3. View initial data with missing values or invalid inputs')
    st.dataframe(df)

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Show summary of missing values including 
    missing_data_summary = summarize_missing_data(df)

    # Remove param
    st.markdown('### 4. Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        df.columns,
    )
    df = remove_features(df, removed_features)

    ########
    # Display updated dataframe
    st.dataframe(df)

    # Impute features
    st.markdown('### 5. Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    impute_method = st.selectbox(
        'Select imputation method',
        ('Zero', 'Mean', 'Median','DropNans')
    )

    # Call impute_dataset function to resolve data handling/cleaning problems
    df = impute_dataset(df, impute_method)
    
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.dataframe(df)

    ############################################# PREPROCESS DATA #############################################

    # Handling Text and Categorical Attributes
    st.markdown('### 6. Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_feature_select_int = st.selectbox(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            df = integer_encode_feature(df, text_feature_select_int)
    
    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            df = one_hot_encode_feature(df, text_feature_select_onehot)

    # Show updated dataset
    st.write(df)

    # Sacling features
    st.markdown('### 7. Feature Scaling')
    st.markdown('Use standardization or normalization to scale features')

    # Use selectbox to provide impute options {'Standardization', 'Normalization', 'Log'}
    scaling_method = st.selectbox(
        'Select feature scaling method',
        ('Standardization', 'Normalization', 'Log')
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    scale_features_select = st.multiselect(
        'Select features to scale',
        numeric_columns,
    )

    # Call scale_features function to scale features
    df = scale_features(df, scale_features_select, scaling_method)
    #########

    # Display updated dataframe
    st.dataframe(df)

    # Create New Features
    st.markdown('## 8. Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        df = create_feature(
                            df, math_select, math_feature_select, new_feature_name)
                        st.write(df)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    df = create_feature(df, math_select, [
                                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(df)

    st.markdown('### 9. Inspect Features for outliers')
    outlier_feature_select = None
    numeric_columns = list(df.select_dtypes(include='number').columns)

    outlier_method_select = st.selectbox(
        'Select statistics to display',
        ['IQR', 'STD']
    )

    outlier_feature_select = st.selectbox(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df, lower_bound, upper_bound = remove_outliers(
            df, outlier_feature_select, outlier_method_select)
        st.write('Outliers for feature %s are lower than %.2f and higher than %.2f' % (
            outlier_feature_select, lower_bound, upper_bound))
        st.write(df)

    # Descriptive Statistics 
    st.markdown('### 10. Summary of Descriptive Statistics')

    stats_numeric_columns = list(df.select_dtypes(['float','int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )

    stats_select = st.multiselect(
        'Select statistics to display',
        ['Mean', 'Median','Max','Min']
    )
            
    # Compute Descriptive Statistics including mean, median, min, max
    display_stats, _ = compute_descriptive_stats(df, stats_feature_select, stats_select)

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### 11. Correlation Analysis")

    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)