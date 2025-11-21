# Door Location Prediction - MLOps Project

Автоматическое предсказание местоположения дверей в BIM-моделях с использованием CatBoost и полного MLOps-пайплайна.

## Описание проекта

Проект решает задачу регрессии для предсказания позиции дверей вдоль стен в архитектурных проектах. Используется метод **leave-one-project-out cross-validation**: модель обучается на N-1 проектах и тестируется на оставшемся проекте.

### Основные характеристики:
- **Данные:** 9 архитектурных проектов, 7916 дверей
- **Модель:** CatBoost Regressor
- **Фичи:** геометрические характеристики стен и смежных помещений
- **Метрики:** Median Error: 0.0149, Mean Error: 0.0365

## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd mlops_hw1_bim_door_prediction
```

### 2. Установка зависимостей
```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
.\venv\Scripts\Activate.ps1

# Активация (Linux/Mac)
source venv/bin/activate

# Установка пакетов
pip install -r requirements.txt
```

### 3. Получение данных
```bash
# Скачать данные из DVC
dvc pull
```

### 4. Запуск пайплайна
```bash
# Запустить весь пайплайн (prepare + train)
dvc repro
```

## Структура проекта

```
mlops_hw1_bim_door_prediction/
├── data/
│   ├── raw/                          # Исходные CSV файлы (nodes, edges)
│   └── processed/                    # Обработанные данные по проектам
│       ├── project_109.csv
│       ├── project_160.csv
│       └── ...
├── src/
│   ├── prepare.py                    # Подготовка данных и feature engineering
│   └── train.py                      # Обучение модели и логирование в MLflow
├── models/                           # Сохраненные модели
│   ├── catboost_model_test_109.pkl
│   ├── feature_importance.csv
│   └── predictions_project_109.csv
├── params.yaml                       # Конфигурация (гиперпараметры, пути)
├── dvc.yaml                          # Определение DVC пайплайна
├── dvc.lock                          # Версии данных и зависимостей
├── requirements.txt                  # Python зависимости
└── README.md                         # Документация
```

## MLOps Pipeline

Пайплайн состоит из двух стадий:

### Stage 1: Prepare (Подготовка данных)
```bash
python src/prepare.py
```

**Входы:**
- `data/raw/nodes_{project_id}.csv` - узлы графа (стены, двери, помещения)
- `data/raw/edges_{project_id}.csv` - рёбра графа (связи между элементами)

**Процесс:**
1. Загрузка данных для всех проектов
2. Извлечение дверей и их границ относительно стен
3. Расчёт геометрических признаков
4. Нормализация координат в диапазон [0, 1]
5. Сохранение обработанных данных по проектам

**Выходы:**
- `data/processed/project_{id}.csv` - данные для каждого проекта
- `data/processed/all_projects.csv` - объединённый датасет

### Stage 2: Train (Обучение модели)
```bash
python src/train.py
```

**Входы:**
- Обработанные данные из `data/processed/`
- Параметры из `params.yaml`

**Процесс:**
1. Разделение данных: train (N-1 проектов) / test (1 проект)
2. Обучение CatBoost модели
3. Оценка модели (fraction-based и spatial metrics)
4. Логирование в MLflow (параметры, метрики, артефакты)
5. Сохранение модели и предсказаний

**Выходы:**
- `models/catboost_model_test_{id}.pkl` - обученная модель
- `models/feature_importance.csv` - важность признаков
- `models/predictions_project_{id}.csv` - предсказания на тестовом проекте

## Признаки (Features)

Модель использует 7 геометрических признаков:

1. `wall_length` - длина стены
2. `room_1_start_fraction` - начало границы первого помещения (0-1)
3. `room_1_end_fraction` - конец границы первого помещения (0-1)
4. `room_1_wall_fraction` - доля стены, занимаемая первым помещением
5. `room_2_start_fraction` - начало границы второго помещения (0-1)
6. `room_2_end_fraction` - конец границы второго помещения (0-1)
7. `room_2_wall_fraction` - доля стены, занимаемая вторым помещением

**Target:** `door_position_fraction` - позиция двери вдоль стены (0-1)

## Метрики качества

### Fraction-based метрики (нормализованные координаты):
- **MAE:** 0.013692
- **RMSE:** 0.024487
- **R²:** 0.991772

### Spatial метрики (реальные координаты, метры):
- **Median Error:** 0.0149 м
- **Mean Error:** 0.0365 м
- **Max Error:** 0.3534 м
- **95th percentile:** 0.1142 м

## Конфигурация

Основные параметры в `params.yaml`:

```yaml
# Тестовый проект (можно менять для cross-validation)
train:
  test_project_id: '109'

# Гиперпараметры CatBoost
model:
  catboost:
    learning_rate: 0.015
    iterations: 1000
    depth: 6
    random_state: 42
```

## MLflow UI

Для просмотра экспериментов, метрик и артефактов:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Откройте в браузере: http://localhost:5000

## Воспроизводимость

### Полное воспроизведение с нуля:
```bash
# 1. Клонирование
git clone <repo-url>
cd mlops_hw1_bim_door_prediction

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Получение данных
dvc pull

# 4. Запуск пайплайна
dvc repro
```

### Изменение тестового проекта:
```bash
# Редактируем params.yaml: test_project_id: '160'
# Запускаем только стадию train
dvc repro train
```

### Изменение гиперпараметров:
```bash
# Редактируем params.yaml: model.catboost.learning_rate
# DVC автоматически обнаружит изменения и переобучит модель
dvc repro
```

## Управление данными с DVC

### Версионирование данных:
```bash
# Посмотреть статус данных
dvc status

# Добавить изменения
dvc add data/processed

# Отправить в удалённое хранилище (после настройки)
dvc push
```

### Откат к предыдущей версии:
```bash
git checkout <commit-hash> dvc.lock
dvc checkout
```

## Разработка

### Запуск отдельных скриптов:
```bash
# Только подготовка данных
python src/prepare.py

# Только обучение
python src/train.py
```

### Проверка пайплайна без выполнения:
```bash
dvc dag  # Показать граф зависимостей
```

## Требования

- Python >= 3.9
- 8GB RAM (для обработки всех проектов)
- ~500MB свободного места на диске

Основные зависимости:
- catboost >= 1.2
- mlflow >= 2.9.0
- dvc >= 3.0
- pandas >= 2.1.0
- numpy >= 1.26.0
- scikit-learn >= 1.3.0

## Автор

Проект выполнен в рамках курса MLOps, МФТИ

## Лицензия

MIT License