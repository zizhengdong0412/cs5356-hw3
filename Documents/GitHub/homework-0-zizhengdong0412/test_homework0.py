from pages import A_Explore_Preprocess_Dataset
import pandas as pd
import numpy as np

student_filepath="datasets/housing_dataset.csv"
test_filepath= "test_dataframe_file/inital_housing.csv"
s_dataframe = pd.read_csv(student_filepath)
e_dataframe = pd.read_csv(test_filepath)
e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
numeric_columns = list(e_X.select_dtypes(['float', 'int']).columns)
nan_colns = e_X.columns[e_X.isna().any()].tolist()

######## CheckPoint 1 ##############
def test_load_dataset():
    '''Test case 1'''
    s_dataframe = A_Explore_Preprocess_Dataset.load_dataset(student_filepath)
    e_dataframe =pd.read_csv(test_filepath)
    pd.testing.assert_frame_equal(s_dataframe,e_dataframe)

###### CheckPoint 2 ##########
def test_sidebar_filter():
    '''Test case 2'''
    dummy_data = pd.DataFrame({
        'A': [-1, 2.0, 3.12, 4.5, -5.11],
        'B': [90, 10, 22, 33.02, 111.82]
    })
    s_=A_Explore_Preprocess_Dataset.sidebar_filter(dummy_data,'Scatterplots','A', 'B')
    assert s_==[(-5.11, 4.5),(10.0,111.82)]


########### Checkpoint 3 ##########
def test_summarize_missing_data_num_categories():
    '''Test case 3'''
    dummy_data = {'Feature1': [1, 2, np.nan, 4],
                    'Feature2': [np.nan, 2, 3, 4],
                    'Feature3': [1, np.nan, np.nan, 4],
                    'Feature4': [1, 2, 3, 4]}  
    df = pd.DataFrame(dummy_data)
    expected = {
        'num_categories': 3.0,  
        'average_per_category': 1.0,  
        'total_missing_values': 4.0, 
        'top_missing_categories': ['Feature3', 'Feature2', 'Feature1']
    }
    s_ = A_Explore_Preprocess_Dataset.summarize_missing_data(df, top_n=3)
    assert s_['num_categories'] == expected['num_categories']

def test_summarize_missing_data_average_per_category():
    '''Test case 4'''
    dummy_data = {'Feature1': [1, 2, np.nan, 4],
                    'Feature2': [np.nan, 2, 3, 4],
                    'Feature3': [1, np.nan, np.nan, 4],
                    'Feature4': [1, 2, 3, 4]}  
    df = pd.DataFrame(dummy_data)
    expected = {
        'num_categories': 3.0,  
        'average_per_category': 1.0,  
        'total_missing_values': 4.0, 
        'top_missing_categories': ['Feature3', 'Feature2', 'Feature1']
    }
    s_ = A_Explore_Preprocess_Dataset.summarize_missing_data(df, top_n=3)
    assert s_['average_per_category'] == expected['average_per_category']

def test_summarize_missing_data_total_missing_values():
    '''Test case 5'''
    dummy_data = {'Feature1': [1, 2, np.nan, 4],
                    'Feature2': [np.nan, 2, 3, 4],
                    'Feature3': [1, np.nan, np.nan, 4],
                    'Feature4': [1, 2, 3, 4]}  
    df = pd.DataFrame(dummy_data)
    expected = {
        'num_categories': 3.0,  
        'average_per_category': 1.0,  
        'total_missing_values': 4.0, 
        'top_missing_categories': ['Feature3', 'Feature2', 'Feature1']
    }
    s_ = A_Explore_Preprocess_Dataset.summarize_missing_data(df, top_n=3)
    assert s_['total_missing_values'] == expected['total_missing_values']

def test_summarize_missing_data_top_missing_categories():
    '''Test case 6'''
    dummy_data = {'Feature1': [1, 2, np.nan, 4],
                    'Feature2': [np.nan, 2, 3, 4],
                    'Feature3': [1, np.nan, np.nan, 4],
                    'Feature4': [1, 2, 3, 4]}  
    df = pd.DataFrame(dummy_data)
    expected = {
        'num_categories': 3.0,  
        'average_per_category': 1.0,  
        'total_missing_values': 4.0, 
        'top_missing_categories': ['Feature3', 'Feature2', 'Feature1']
    }
    s_ = A_Explore_Preprocess_Dataset.summarize_missing_data(df, top_n=3)
    assert all([a == b for a, b in zip(sorted(s_['top_missing_categories']), sorted(expected['top_missing_categories']))])

######## CheckPoint 4 ##############

def test_remove_features():
    '''Test case 7'''
    e_remove= pd.read_csv("./test_dataframe_file/remove.csv")
    e_X= e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_remove = A_Explore_Preprocess_Dataset.remove_features(e_X, ['latitude', 'longitude'])
    pd.testing.assert_frame_equal(s_remove,e_remove)

######## CheckPoint 5 ##############

def test_impute_dataset_zero():
    '''Test case 8'''
    e_zero_df = pd.read_csv("test_dataframe_file/Zero.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_zero_df = A_Explore_Preprocess_Dataset.impute_dataset(e_X, 'Zero')
    pd.testing.assert_frame_equal(e_zero_df,s_zero_df)

def test_impute_dataset_median():
    '''Test case 9'''
    e_median_df = pd.read_csv("test_dataframe_file/Median.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_median_df = A_Explore_Preprocess_Dataset.impute_dataset(e_X, 'Median')
    pd.testing.assert_frame_equal(e_median_df,s_median_df)

def test_impute_dataset_mean():
    '''Test case 10'''
    e_mean_df = pd.read_csv("test_dataframe_file/Mean.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_mean_df = A_Explore_Preprocess_Dataset.impute_dataset(e_X, 'Mean')
    pd.testing.assert_frame_equal(e_mean_df,s_mean_df)

# Checkpoint 6 ################### 
def test_one_hot_encode_feature():
    '''Test case 11'''
    student_one_hot_encode_feature = A_Explore_Preprocess_Dataset.one_hot_encode_feature(
        s_dataframe, 'ocean_proximity')
    assert 'ocean_proximity_<1H OCEAN' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_INLAND' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_ISLAND' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_NEAR BAY' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_NEAR OCEAN' in student_one_hot_encode_feature.columns

# Checkpoint 7 ################### 
def test_integer_encode_feature():
    '''Test case 12'''
    student_integer_encode_feature = A_Explore_Preprocess_Dataset.integer_encode_feature(
        s_dataframe, 'ocean_proximity')
    assert student_integer_encode_feature['ocean_proximity_int'].sum() == 24009.0   

######## CheckPoint 8 ##############
def test_scale_features_std():
    '''Test case 13'''
    # Dummy Data
    dummy_data ={
        'longitude': [16, 32, 64],
        'latitude': [2, 4, 8]
    }
    df = pd.DataFrame(dummy_data)
    expected = {'longitude_log': [4., 5., 6.], 
                'latitude_log': [1., 2., 3.]}
    s_standardization = A_Explore_Preprocess_Dataset.scale_features(e_dataframe,['longitude','latitude'],'Standardization')
    assert np.isclose(0, s_standardization['longitude_std'].mean())
    assert np.isclose(1, s_standardization['longitude_std'].std())
    assert np.isclose(0, s_standardization['latitude_std'].mean())
    assert np.isclose(1, s_standardization['latitude_std'].std())

def test_scale_features_norm():
    '''Test case 14'''
    # Dummy Data
    dummy_data ={
        'longitude': [16, 32, 64],
        'latitude': [2, 4, 8]
    }
    df = pd.DataFrame(dummy_data)
    expected = {'longitude_log': [4., 5., 6.], 
                'latitude_log': [1., 2., 3.]}
    s_Normalization = A_Explore_Preprocess_Dataset.scale_features(e_dataframe,['longitude','latitude'],'Normalization')
    assert np.isclose(0, s_Normalization['longitude_norm'].min())
    assert np.isclose(1, s_Normalization['longitude_norm'].max())
    assert np.isclose(0, s_Normalization['latitude_norm'].min())
    assert np.isclose(1, s_Normalization['latitude_norm'].max())

def test_scale_features_log():
    '''Test case 15'''
    # Dummy Data
    dummy_data ={
        'longitude': [16, 32, 64],
        'latitude': [2, 4, 8]
    }
    df = pd.DataFrame(dummy_data)
    expected = {'longitude_log': [4., 5., 6.], 
                'latitude_log': [1., 2., 3.]}
    s_Log = A_Explore_Preprocess_Dataset.scale_features(df,['longitude','latitude'],'Log')
    print("wrong")
    print(pd.DataFrame(s_Log))
    print(pd.DataFrame(expected))
    assert np.all(np.isclose(expected['latitude_log'], s_Log['latitude_log']))
    assert np.all(np.isclose(expected['longitude_log'], s_Log['longitude_log']))

################ Checkpoint 9 ##########
def test_create_feature_sqrt():
    '''Test case 16'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)

    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'square root', ['feature1'], 'sqrt_feature1')
    assert np.allclose(df_updated['sqrt_feature1'], np.sqrt(df['feature1']))

def test_create_feature_ceil():
    '''Test case 17'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'ceil', ['feature2'], 'ceil_feature2')
    assert np.allclose(df_updated['ceil_feature2'], np.ceil(df['feature2']))

def test_create_feature_floor():
    '''Test case 18'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'floor', ['feature2'], 'floor_feature2')
    assert np.allclose(df_updated['floor_feature2'], np.floor(df['feature2']))

def test_create_feature_add():
    '''Test case 19'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'add', ['feature1', 'feature2'], 'sum_feature1 + 2')
    assert np.allclose(df_updated['sum_feature1 + 2'], df['feature1'] + df['feature2'])

def test_create_feature_subtract():
    '''Test case 20'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'subtract', ['feature1', 'feature2'], 'diff_feature1 - 2')
    assert np.allclose(df_updated['diff_feature1 - 2'], df['feature1'] - df['feature2'])

def test_create_feature_multiply():
    '''Test case 21'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'multiply', ['feature1', 'feature2'], 'prod_feature1 * 2')
    assert np.allclose(df_updated['prod_feature1 * 2'], df['feature1'] * df['feature2'])

def test_create_feature_divide():
    '''Test case 22'''
    dummy_data = {
        'feature1': [32, 19, 16],
        'feature2': [1, 2, 30]
    }
    df = pd.DataFrame(dummy_data)
    df_updated = A_Explore_Preprocess_Dataset.create_feature(df.copy(), 'divide', ['feature1', 'feature2'], 'div_feature1/2')
    assert np.allclose(df_updated['div_feature1/2'], df['feature1'] / df['feature2'])  
    
###### Checkpoint 10 ##############

def test_remove_outliers_iqr():
    '''Test case 23'''
    dummy_data = {'feature': [-1,-99, 2, 3, 4, 5,5,8,9,100,88]}
    df = pd.DataFrame(dummy_data)
    cleaned_df, lower_bound, upper_bound = A_Explore_Preprocess_Dataset.remove_outliers(df, 'feature', 'IQR')
    print(cleaned_df, lower_bound, upper_bound)
    assert not np.any(np.isin([100, 88, -99], cleaned_df['feature'].values))
    assert lower_bound, upper_bound==(-6.5, 17.5)    

def test_remove_outliers_std():
    '''Test case 24'''
    dummy_data = {'feature': [-1,-99, 2, 3, 4, 5,5,8,9,100,88]}
    df = pd.DataFrame(dummy_data)
    cleaned_df, lower_bound, upper_bound = A_Explore_Preprocess_Dataset.remove_outliers(df, 'feature', 'STD')
    print(cleaned_df, lower_bound, upper_bound)
    assert lower_bound, upper_bound==(-142.78, 165.33) 

################# Checkpoint 11 #################

## You have to round to two decimal places
def test_compute_descriptive_stats_mean():
    '''Test case 25'''
    _, out_dict=A_Explore_Preprocess_Dataset.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.25,
        'max': 41.95,
        'min': 32.54,
    }
    assert out_dict['mean']==e_dict['mean']
    
def test_compute_descriptive_stats_median():
    '''Test case 26'''
    _, out_dict=A_Explore_Preprocess_Dataset.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.25,
        'max': 41.95,
        'min': 32.54,
    }
    assert out_dict['median']==e_dict['median']

def test_compute_descriptive_stats_max():
    '''Test case 27'''
    _, out_dict=A_Explore_Preprocess_Dataset.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.25,
        'max': 41.95,
        'min': 32.54,
    }
    assert out_dict['max']==e_dict['max']

def test_compute_descriptive_stats_min():
    '''Test case 28'''
    _, out_dict=A_Explore_Preprocess_Dataset.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.25,
        'max': 41.95,
        'min': 32.54,
    }
    assert out_dict['min']==e_dict['min']

######## CheckPoint 12 ##############

def test_compute_corr():
    '''Test case 29'''
    e_corr = np.array([[1,  -0.035676], [ -0.035676, 1]])
    test_corr = A_Explore_Preprocess_Dataset.compute_correlation(e_dataframe,['latitude','total_rooms'])[0].to_numpy()
    assert np.allclose(e_corr, test_corr)