from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import requests


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'heart_disease',
    default_args=default_args,
    description='AI model training and evaluation DAG',
    schedule_interval=timedelta(days=1),
)

def getAndTrain(**kwargs):
    url = "https://storage.googleapis.com/kagglesdsdata/datasets/1226038/2047221/heart.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240609%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240609T205415Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4c1de7b500f36e92b6756a8f21c76f37485d9cd551ccf039ae4bf9aad15b606a662c55abff63ea600371157511ffbf11587b516bba67e123ce4dfc9cc684933e94c8ac5d0da66d6ffdc96c09d43b1f57b706001c2d3853afc1d2eb9d0797ac168cd0f0499e13d58434056e7df0619f8eb110036c361c367cb32b7e300330ca43c734af2e34c76b1245ab7f7d83483d8cd5e65845ebb3fd6b1377c48599b87a1e243fdae56abb50381edc402a8b578f58d3d1e9c5c4afd28efeabd8c5e4ceddf9aea3917e75b06e5b80b38d1681691594cfb5ba7c660374a6fe681d5bda40a74c6efbea8b30469e14c4ac4f2af45d90d72885973e1e1053c2d5037963ba3831f2"

    csv_filename = 'heart.csv'

    # Baixar o arquivo CSV
    response = requests.get(url)
    response.raise_for_status()  # Verifica se o download foi bem-sucedido

    # Salvar o arquivo CSV localmente
    with open(csv_filename, 'wb') as file:
        file.write(response.content)
        
    df = pd.read_csv(csv_filename)
    categorias = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    valores_numericos = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

    df = pd.get_dummies(df, columns=categorias, drop_first=True)
    X = df.drop(['output'], axis=1)
    y = df['output'].values

    rs = RobustScaler()
    X[valores_numericos] = rs.fit_transform(X[valores_numericos])

    print("Data preprocessing completed.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Model Evaluation - Accuracy: {accuracy}, F1-Score: {f1}")

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")

preprocess_task = PythonOperator(
    task_id='getAndTrain',
    python_callable=getAndTrain,
    dag=dag,
)

preprocess_task
