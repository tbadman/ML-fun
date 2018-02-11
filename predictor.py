import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

titanic_df = pd.read_csv('titanic_train.csv')
titanic_df_trimmed = titanic_df.drop(['PassengerId','Name','Ticket','Embarked'],axis=1)

titanic_df_trimmed['fsize'] = titanic_df_trimmed['Parch'] + titanic_df_trimmed['SibSp']
titanic_df_trimmed.groupby(['fsize','Sex'])['Survived'].mean().plot.bar()
plt.show()
