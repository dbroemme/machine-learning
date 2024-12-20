import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

# Load the data from the CSV file
#data = pd.read_csv('../datasets/StudentPerformanceFactors.csv')
#data = data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Gender', 'Exam_Score']]

#data = pd.read_csv('./data/spotify-2023.csv', encoding="ISO-8859-1")
#data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
# Drop or fill NaN values (if there are any), e.g., filling with 0
#data['streams'].fillna(0, inplace=True)  # or df.dropna(subset=['streams'], inplace=True)
# Convert to integer
#data['streams'] = data['streams'].astype(int)
#data.to_csv("./data/modified_spotify.csv", index=False, encoding="utf-8")


data = pd.read_csv('./data/modified_spotify.csv', encoding="ISO-8859-1")
#df_encoded = pd.get_dummies(data, columns=['key', 'mode'], drop_first=True)
#scaler = StandardScaler()
#data[['key', 'mode']] = scaler.fit_transform(data[['key', 'mode']])
categorical_columns = ['key', 'mode'] 
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[f'{column}'] = label_encoder.fit_transform(data[column])
print(data.info())
print(data.head())

#data = data[['bpm', 'key', 'mode', 'energy_%', 'streams']]
data = data[['bpm', 'acousticness_%', 'instrumentalness_%', 'energy_%', 'streams']]

sns.pairplot(data, hue='streams')
plt.show()


