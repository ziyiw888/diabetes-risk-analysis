import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    nhanes = pd.read_sas('DEMO_J.XPT')
    brfss = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

    nhanes_filtered = nhanes[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMIN2']].copy()
    nhanes_filtered.columns = ['ID', 'Age', 'Sex', 'Income']

    brfss_filtered = brfss[['Diabetes_012', 'Sex', 'Age', 'Income', 'BMI']].copy()

    nhanes_filtered['Sex'] = nhanes_filtered['Sex'].replace({1: 'Male', 2: 'Female'})
    brfss_filtered['Sex'] = brfss_filtered['Sex'].replace({0: 'Female', 1: 'Male'})

    nhanes_filtered['Age'] = nhanes_filtered['Age'].astype(int)
    brfss_filtered['Age'] = brfss_filtered['Age'].astype(int)

    merged = pd.merge(brfss_filtered, nhanes_filtered, on=['Age', 'Sex'])

    print('Merged dataset shape:', merged.shape)

    merged_cleaned = merged.dropna()
    print('After cleaning:', merged_cleaned.shape)

    merged_cleaned.to_csv('merged_cleaned.csv', index=False)

    print('\nMissing values per column:')
    print(merged_cleaned.isnull().sum())

    print('\nVariables of interest:', merged_cleaned.columns.tolist())

    print('\nBMI Summary:')
    print(merged_cleaned['BMI'].describe())

    print('\nIncome_x value counts:')
    print(merged_cleaned['Income_x'].value_counts().sort_index())

    diabetes_by_income = merged_cleaned.groupby('Income_x')['Diabetes_012'].mean()
    diabetes_by_income.plot(kind='bar')
    plt.xlabel('Income Level')
    plt.ylabel('Avg Diabetes Score')
    plt.title('Diabetes Rate by Income')
    plt.tight_layout()
    plt.savefig('diabetes_by_income.png')

    sns.kdeplot(data=merged_cleaned, x='BMI', hue='Diabetes_012', fill=True)
    plt.title('BMI Distribution by Diabetes Status')
    plt.xlabel('BMI')
    plt.tight_layout()
    plt.savefig('bmi_distribution.png')


if __name__ == '__main__':
    main()