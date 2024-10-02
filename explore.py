import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
data = pd.read_csv('../datasets/StudentPerformanceFactors.csv')
data = data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Gender', 'Exam_Score']]

sns.pairplot(data, hue='Gender')
plt.show()
