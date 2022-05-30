# стандартные
import numpy as np

# графика
import matplotlib.pyplot as plt

# препроцессинг и метрики
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

def ts_feature_preparation(ts, window, n_predictions):
    """ 
        Преобразует ряд значений в фичи и метки для нейронной сети.
        Предполагаем, что данные ряда нормализованы.
        Возвращает фичи и метки.
        Параментры: ts - ряд значений,
                    window - глубина анализа ряда значений, 
                    n_predictions - глубина предсказания.
    """
    # данные для тренировочного набора
    ts_training_scaled = ts[:-n_predictions]
    # выделяем фичи и метки
    features_set = []
    labels = []
    for i in range(window, ts_training_scaled.shape[0]):
        features_set.append(ts_training_scaled[i-window:i])
        labels.append(ts_training_scaled[i])
    # преобразуем фичи и метки в массивы numpy
    features_set, labels = np.array(features_set), np.array(labels)
    # преобразуем фичи в в формат для работы с нейронной сетью
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    # печатаем размерности массивов
    print(f'Данные для обучения: фичи - {features_set.shape}, метки - {labels.shape}')
    # возвращаем фичи и метки
    return features_set, labels

def lstm_model(units, input_shape, summary = False):
    """ 
        Собирает и возвращает модель.
        Параментры: units - количество нейронов входного слоя,
                    input_shape - размерность данных (features_set.shape[1], 1), 
                    summary - смотрим ли на модель.
    """
    # конфигурация нейронной сети
    model = Sequential()
    # LSTM
    model.add(LSTM(units = units, return_sequences = False, input_shape = input_shape))
    model.add(Dropout(0.2))
    # полносвязные слои
    #model.add(Dense(units = 1000))
    #model.add(Dropout(0.2))
    #model.add(Dense(units = 100))
    #model.add(Dropout(0.2))
    #model.add(Dense(units = 30))
    #model.add(Dropout(0.1))
    # выходной слой
    model.add(Dense(units = 1))
    # собираем модель
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # смотрим на модель
    if summary:
        print(model.summary())
    # возвращаем модель
    return model

def model_fit(model, features_set, labels, epochs, batch_size = 16, validation_split = .1, verbose = 2):
    """ 
        Тренирует модель.
        Возвращает модель и объект history.
        Параментры: model - модель,
                    features_set - фичи,
                    labels - метки,
                    epochs - количество эпох,
                    batch_size - размер батча,
                    validation_split - размер набора валидации,
                    verbose - степень детализации процесса обучения.
    """
    hist = model.fit(features_set, labels, epochs = epochs, batch_size = batch_size, \
                     validation_split = validation_split, verbose = verbose)
    # возвращаем
    return model, hist
           

def pict_loss_val_loss(history):
    """ 
        Рисуем графики loss и val_loss.
        Параметры: history - объект history.
    """
    plt.figure(figsize=(16, 4))
    plt.plot(history.history['loss'], color = 'blue')
    plt.plot(history.history['val_loss'], color = 'orange')
    plt.grid()
    plt.show()
    
def test_feature_preparation(ts, window, n_predictions):
    """ 
        Преобразует ряд значений в фичи для нейронной сети и метки для оценки качества.
        Возвращает фичи и метки.
        Параментры: ts - ряд значений,
                    window - глубина анализа ряда значений, 
                    n_predictions - глубина предсказания
    """
    ts_testing_processed = ts[len(ts) - n_predictions:]
    test_inputs = ts[len(ts) - len(ts_testing_processed) - window:]
    test_inputs = test_inputs.reshape(-1, 1)
    
    test_features = []
    test_labels = []
    for i in range(window, window + ts_testing_processed.shape[0]):
        test_features.append(test_inputs[i - window:i])
        test_labels.append(test_inputs[i])
    
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    
    print(f'Данные для тестирования: фичи - {test_features.shape}, метки - {len(test_labels)}')
    
    # возвращаем фичи и метки
    return test_features, test_labels

def test_predict(model, test_features):
    """ 
        Предсказывает.
        Возвращает предсказание.
        Параментры: model - обученная модель,
                    test_features - фичи для предсказания
    """
    # возвращаем прогноз
    return model.predict(test_features)

def pict_predict(predictions, test_labels, error, shift = None, name = None):
    """ 
        Печатает графики предсказанных и истинных значений, печатает ошибки.
        Параментры: predictions - предсказанные значения,
                    test_labels - истинные значения
    """
    # рисуем
    plt.figure(figsize = (16, 8))
    
    # сдвиг (тема не понятная, возможно косячу)
    if shift:
        plt.plot(test_labels[:-shift], color = 'blue', label = 'истинные значения')
        plt.plot(predictions[shift:], color = 'red', label = 'предсказанные значения')
        plt.plot(error[:-shift], color = 'green', label = 'ошибка')
    else:
        plt.plot(test_labels, color = 'blue', label = 'истинные значения')
        plt.plot(predictions, color = 'red', label = 'предсказанные значения')
        plt.plot(error, color = 'green', label = 'ошибка')
    
    # продолжаем рисовать
    plt.title(f"Прогноз компоненты '{name}'")
    plt.xlabel('Дата')
    plt.ylabel('Оборот')
    plt.grid()
    #plt.yticks([])
    plt.legend()
    plt.show()
    
    # печатаем ошибки
    if shift:
        print('R^2: %1.2f' % r2_score(test_labels[:-shift], predictions[shift:]))
        print(f'RMSE на норм. данных: {np.sqrt(mean_squared_error(test_labels[:-shift], predictions[shift:]))}')
    else:
        print('R^2: %1.2f' % r2_score(test_labels, predictions))
        print(f'RMSE на норм. данных: {np.sqrt(mean_squared_error(test_labels, predictions))}')