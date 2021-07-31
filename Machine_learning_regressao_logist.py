import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

data_set = pd.read_csv('Social_Network_Ads.csv')

x = data_set.iloc[:,[2,3] ].values
y = data_set.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_training,x_test,y_training,y_test = train_test_split(x,y,
 test_size=0.25,random_state=0 )


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_training =sc.fit_transform(x_training)
x_test =sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
 
classificador = LogisticRegression(random_state = 0)
classificador.fit(x_training,y_training) 


y_prev= classificador.predict(x_test)

from sklearn.metrics import confusion_matrix

matriz_conf = confusion_matrix(y_test,y_prev)


# codigo da impressão

from matplotlib.colors import ListedColormap
X_set, y_set = x_training, y_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classificador (Dados de Treino)')
plt.xlabel('Idade')
plt.ylabel('Salário Estimado')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set =x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classificador (Dados de Treino)')
plt.xlabel('Idade')
plt.ylabel('Salário Estimado')
plt.legend()
plt.show()