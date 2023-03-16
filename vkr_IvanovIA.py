import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import datetime
import warnings
warnings.filterwarnings("ignore")

# загрузка данных из файлов
df_train = pd.read_csv('airline_train.csv')
df_test = pd.read_csv('airline_test.csv')

#проверка на пропуски и дубликаты
# print(df_train.head())
# print(df_train.isnull().sum())
# print(df_train.duplicated().sum())

# print(df_test.head())
# print(df_test.isnull().sum())
# print(df_test.duplicated().sum())


# корреляция

# corr_train = df_train['Departure Delay in Minutes'].corr(df_train['Arrival Delay in Minutes'])
# corr_test = df_test['Departure Delay in Minutes'].corr(df_test['Arrival Delay in Minutes'])

############ 

#####  Визуализация  ##############
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.pie(df_train.satisfaction.value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"], autopct = '%1.1f%%')
# plt.title('Обучающая выборка')
# plt.subplot(1, 2, 2)
# plt.pie(df_test.satisfaction.value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"], autopct = '%1.1f%%')
# plt.title('Тестовая выборка')
# plt.show()
####################################
# df_train_young = df_train[df_train['Age'] < 14]
# df_test_young = df_test[df_test['Age'] < 14]

# plt.figure(figsize=(15, 15))
# plt.subplot(1, 2, 1)
# plt.pie(df_train_young.satisfaction.value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"], autopct = '%1.1f%%')
# plt.title('Возраст < 14 лет: обучающая выборка')
# plt.subplot(1, 2, 2)
# plt.pie(df_test_young.satisfaction.value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"], autopct = '%1.1f%%')
# plt.title('Возраст < 14 лет: тестовая выборка')
# plt.show()
###########################################
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.pie(df_train.Gender.value_counts(), labels = ["Male", "Female"], autopct = '%1.1f%%')
# plt.title('Обучающая выборка')
# plt.subplot(1, 2, 2)
# plt.pie(df_test.Gender.value_counts(), labels = ["Male", "Female"], autopct = '%1.1f%%')
# plt.title('Тестовая выборка')

# plt.show()
####################################################################
# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# sns.countplot(x ="Gender", data = df_train, hue ="satisfaction", order = df_train['Gender'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Gender: обучающая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.subplot(1, 2, 2)
# sns.countplot(x ="Gender", data = df_test, hue ="satisfaction", order = df_train['Gender'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Gender: тестовая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.show()

###################################################################
# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# sns.countplot(x ="Customer Type", data = df_train, hue ="satisfaction", order = df_train['Customer Type'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Customer Type: обучающая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.subplot(1, 2, 2)
# sns.countplot(x ="Customer Type", data = df_test, hue ="satisfaction", order = df_train['Customer Type'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Customer Type: тестовая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.show()
# ###################################
# plt.figure(figsize=(20, 8))
# plt.subplot(1, 2, 1)
# sns.histplot( x= "Age", data = df_train, kde= True, bins = 75)
# plt.title('Распределение по возрасту: обучающая выборка')

# plt.subplot(1, 2, 2)
# sns.histplot( x= "Age", data = df_test, kde= True, bins = 75)
# plt.title('Распределение по возрасту: тестовая выборка')
# plt.show()
# ###################################

# ###################################

# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# sns.countplot(x ="Type of Travel", data = df_train, hue ="satisfaction", order = df_train['Type of Travel'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Type of Travel: обучающая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.subplot(1, 2, 2)
# sns.countplot(x ="Type of Travel", data = df_test, hue ="satisfaction", order = df_train['Type of Travel'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Type of Travel: тестовая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.show()
###################################

# ###################################
# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# sns.countplot(x ="Class", data = df_train, hue ="satisfaction", order = df_train['Class'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Class: обучающая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.subplot(1, 2, 2)
# sns.countplot(x ="Class", data = df_test, hue ="satisfaction", order = df_train['Class'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# })
# plt.title("Satisfaction vs Class: тестовая выборка")
# sns.despine(top = True, right = True, left = False, bottom = False)
# plt.legend(loc='upper center',  title = "satisfaction")
# plt.show()

# def classes (df):
#     plt.figure(figsize=(5, 5))
#     sns.countplot(x ="Class", data = df, hue ="satisfaction", order = df_train['Class'].value_counts().index, hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
#     'satisfied': 'tab:blue',
#     'neutral or dissatisfied': 'tab:orange'
# 
#     plt.title("Satisfaction results vs Class: обучающая выборка")
#     sns.despine(top = True, right = True, left = False, bottom = False)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")
#     plt.show()

# classes(df_train)
# classes(df_test)


# ###################################


# sns.histplot(x = "Flight Distance", data = df_train, hue ="satisfaction", hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
# 'satisfied': 'tab:blue',
# 'neutral or dissatisfied': 'tab:orange'})
# plt.title("Satisfaction results vs Flight Distance: обучающая выборка")
# plt.show()

# sns.histplot(x = "Flight Distance", data = df_test, hue ="satisfaction", hue_order = ['satisfied', 'neutral or dissatisfied'], palette = {
# 'satisfied': 'tab:blue',
# 'neutral or dissatisfied': 'tab:orange'})
# plt.title("Satisfaction results vs Flight Distance: тестовая выборка")
# plt.show()

########################################

# оценка субъективных параметров
# df_not_business = df_train[df_train['Class'] != 'Business'].loc[:, 'Inflight wifi service':'Cleanliness']
# df_business = df_train[df_train['Class'] == 'Business'].loc[:, 'Inflight wifi service':'Cleanliness']
# df_male = df_train[df_train['Gender'] == 'Male'].loc[:, 'Inflight wifi service':'Cleanliness']
# df_female = df_train[df_train['Gender'] == 'Female'].loc[:, 'Inflight wifi service':'Cleanliness']
# df_train_all = df_train.loc[:, 'Inflight wifi service':'Cleanliness']
# df_test_all = df_test.loc[:, 'Inflight wifi service':'Cleanliness']

# def figures(df):
#     plt.figure(figsize=(20, 20))
    
#     for i in range(df.shape[1]):
#         plt.subplot(4, 4, i+1)
#         labels = sorted(list(df.iloc[:, i].unique()))
#         plt.pie(df.iloc[:, i].value_counts(), labels = labels, counterclock = False, startangle = 90, autopct = '%1.1f%%')
#         plt.title(df.columns[i])
        
#     plt.show()
# print('По выборке в целом:')
# figures(df_train_all)
# print('По бизнес-классу:')
# figures(df_business)
# print('По остальным классам:')
# figures(df_not_business)
# print('По мужчинам:')
# figures(df_male)
# print('По женщинам:')
# figures(df_female)
# ######################


########################################### Расчеты ###########################
# проверка на полноту значений в столбцах с субъективной оценкой со стороны пассажира (должно быть от 0 до 5 во всех столбцах)
df_check_train = df_train.loc[:, 'Inflight wifi service':'Cleanliness'] # отбираем нужные столбцы
df_check_test = df_test.loc[:, 'Inflight wifi service':'Cleanliness']

for i in range(df_check_test.shape[1]): # перебор по столбцам
    if df_check_train.iloc[:,i].nunique() != df_check_test.iloc[:,i].nunique(): # если число уникальных значений в столбцах
    #обучающей и тестовой выборки отличается, то:
        for j in range(len(df_check_train.iloc[:,i].unique())): # перебор по каждому значению в столбце
            if j not in df_check_test.iloc[:,i].unique(): # если не находим:
                nr_add = (df_check_train[df_check_train.iloc[:, i] == j].index.values)[0] # добавляем отсутствующую строку в тестовую выборку
                df_add = pd.DataFrame(df_train.iloc[nr_add:nr_add+1])
                df_test = df_test.append(df_add)
                df_test.drop_duplicates(inplace = True)
                df_test = df_test.reset_index(inplace=False, drop = True)        

# очистка данных
def cleaning(df): 
    df = df[df['Age'] >= 14] # фильтрация по возрасту, расстоянию и задержке
    df = df[df['Flight Distance'] >= 250]
    df = df[df['Departure Delay in Minutes'] < (df['Departure Delay in Minutes'].mean() + 5*(df['Departure Delay in Minutes'].std()))]
    
    df['satisfaction']= pd.get_dummies(df['satisfaction'])[list(df['satisfaction'].unique())[0]] # переводим satisfaction в 0/1

    target = df['satisfaction'] # назначаем целевую переменную

    df = df.drop(df.iloc[:,[0, 1]], axis = 1) # удаляем первые 2 столбца 
    categorical_indexes = [0, 1, 3, 4] + list(range(6, 20)) # определяем, какие переменные будут категориальными
    df.iloc[:,categorical_indexes] = df.iloc[:,categorical_indexes].astype('category') # переводим их в категориальные
    
    df['Arrival Delay in Minutes'].fillna(df['Departure Delay in Minutes'], inplace = True) # ищем пропуски в задержках, заполняем их и оставляем один столбец 
    df['Departure Delay in Minutes'] = (df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'])/2
    df = df.drop(df.iloc[:,[-2]], axis = 1)
    df.rename(columns = {'Departure Delay in Minutes':'Delay in Minutes'}, inplace = True)
    
    numerical_columns = [c for c in df.columns if df[c].dtype.name != 'category'] # разбиваем столбцы на количественные и категориальные
    numerical_columns.remove('satisfaction')
    categorical_columns = [c for c in df.columns if df[c].dtype.name == 'category']
    df_describe = df.describe(include = ['category'])
    
    binary_columns = [c for c in categorical_columns if df_describe[c]['unique'] == 2] # разбиваем категориальные столбцы на бинарные и небинарные
    nonbinary_columns = [c for c in categorical_columns if df_describe[c]['unique'] > 2]
    
    df_binary = df[binary_columns].reset_index(inplace=False, drop = True)
    
    for col in binary_columns:
        df_binary[col]= pd.get_dummies(df_binary[col])[list(df_binary[col].unique())[0]] # обрабатываем бинарные
     
    df_nonbinary = pd.get_dummies(df[nonbinary_columns]).reset_index(inplace=False, drop = True)
    
    df_numerical = df[numerical_columns]
  
    scaler = MinMaxScaler() # нормализуем данные в количественных столбцах
    df_numerical = scaler.fit_transform(df_numerical)
    df_numerical = pd.DataFrame(df_numerical, columns = numerical_columns)
  
    df = pd.concat((df_numerical, df_nonbinary, df_binary), axis = 1) # итоговый датасет
    
    return df, target

# получение обучающей и тестовой выборок
X_train, y_train = cleaning(df_train)
X_test, y_test = cleaning(df_test)

# расчетные функции
def neighbor (x_train, x_test, y_train, n):
    classifier = KNeighborsClassifier(n_neighbors = n)
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    return y_pred 


def Bayes (x_train, x_test, y_train):
    classif = BernoulliNB()
    #Обучение модели Наивного Байеса
    classif.fit(x_train, y_train.values.ravel())
    #Проверка модели на тестовой выборке
    y_pred = classif.predict(x_test)
    classif = GaussianNB()
    classif.fit(x_train, y_train.values.ravel())
    y_pred3 = classif.predict(x_test)
    return  y_pred, y_pred3 
    

def Tree (x_train, x_test, y_train):
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', min_samples_split = 2, 
                                 min_samples_leaf = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred
     
def neural (x_train, x_test, y_train, n):
    clf = MLPClassifier(hidden_layer_sizes= (n), activation = 'logistic', solver = 'sgd', 
                            learning_rate_init = 0.01, max_iter = 2000, tol = 0.000001)
    clf.fit(x_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)
    return y_pred

def LogRegr (x_train, x_test, y_train):
    classifier = LogisticRegression(solver = 'sag', penalty = 'l2')
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    return y_pred

# метрики проверки
def verif (y_test, y_pred):
    print('Accuracy = ', accuracy_score(y_test, y_pred))
    print('Recall = ', recall_score(y_test, y_pred))
    print('Precision = ', precision_score(y_test, y_pred))
    print('f1 = ', f1_score(y_test, y_pred))

# вызов расчетных функций
dt_start = datetime.datetime.now() 
print('Нейронная сеть:')
verif (y_test, neural(X_train, X_test, y_train, 5))
dt_end = datetime.datetime.now()
print('Продолжительность вычислений, с: ', (dt_end - dt_start).total_seconds())

dt_start = datetime.datetime.now() 
print('Ближайший сосед:')
verif (y_test, neighbor(X_train, X_test, y_train, 10))
dt_end = datetime.datetime.now()
print('Продолжительность вычислений, с: ', (dt_end - dt_start).total_seconds())

dt_start = datetime.datetime.now()
print('Наивный Байес:')
y1, y2 = Bayes(X_train, X_test, y_train)
print('Бернулли:')
verif(y_test, y1)
print('Гаусс:')
verif(y_test, y2)
dt_end = datetime.datetime.now()
print('Продолжительность вычислений, с: ', (dt_end - dt_start).total_seconds())

dt_start = datetime.datetime.now() 
print('Дерево:')
verif(y_test, Tree (X_train, X_test, y_train))
dt_end = datetime.datetime.now()
print('Продолжительность вычислений, с: ', (dt_end - dt_start).total_seconds())

dt_start = datetime.datetime.now() 
print('Лог. регрессия:')
verif(y_test, LogRegr (X_train, X_test, y_train))
dt_end = datetime.datetime.now()
print('Продолжительность вычислений, с: ', (dt_end - dt_start).total_seconds())









