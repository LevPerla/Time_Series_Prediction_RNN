# Time_Series_Prediction_RNN

### Getting Started

Installation:  
<code>$ pip3 install -r requirements.txt</code>

### Example
You could find it on notebooks/Example.ipynb


Библиотека включает в себя два основных модуля:
* Data_processor – класс, который отвечает за преобразование данных;
* Model – класс, который представляет из себя модель рекуррентной нейронной сети с ее атрибутами.

### Data_processor
Класс data_processor имеет 2 атрибута: scaler_target и scaler_factors, в которых сохраняются настройки шкалировщиков (классы, которые приводят значения совокупности к диапазону от 0 до 1).

#### К классу можно применить 5 методов:
1.	scaler_fit_transform – обучение шкалировщика и приведение значений совокупности к диапазону от 0 до 1;
2.	scaler_transform - приведение значений совокупности к диапазону от 0 до 1 на основе уже обученного шкалировщика;
3.	scaler_inverse_transform – обратное преобразование прогнозных значений;
4.	train_test_split – разбиение совокупности на обучающую и тестовую выборки;
5.	split_sequence – преобразование временного ряда под формат ввода нейронной сети.
Класс data_processor включает в себя основные способы обработки данных, которые необходимо выполнять при прогнозировании с помощью рекуррентных нейронных сетей.

### Model
#### Данный класс имеет 6 атрибутов:
1.	model – модель нейронной сети типа Sequential (внутренний класс библиотеки Keras);
2.	id – Номер модели в формате uuid, необходим для сохранения логов экспериментов;
3.	n_step_in – длина входного вектора;
4.	n_step_out – длина выходного вектора;
5.	n_features – количество временных рядов на вход;
6.	params – описание гиперпараметров модели в формате словаря Python.

#### Класс model поддерживает следующие методы:
1.	load_model – загрузить веса нейронной сети из файла формата h5;
2.	build_model – собрать нейронную сеть согласно параметрам, указанным в файле configs.json;
3.	fit – обучить нейронную сеть;
4.	save – сохранение весов модели в формате h5 по указанному пути;
5.	predict – спрогнозировать с помощью нейронной сети, использует два метода:
a.	predict_point_by_point – циклический прогноз по одному значению вперед;
b.	predict_multi_step – прогноз вектором на прогнозных горизонт.
