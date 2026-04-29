import pandas as pd

def main():
    df = pd.read_csv('merged_cleaned.csv')

    df = df[(df['BMI'] >= 10) & (df['BMI'] <= 70)]

    print("Running Test 1: Non-empty dataset")
    assert df.shape[0] > 0

    print("Running Test 2: Missing values")
    cols_to_check = ['Diabetes_012', 'Sex', 'Age', 'Income_x', 'BMI']
    for col in cols_to_check:
        missing = df[col].isnull().sum()
        assert missing == 0

    print("Running Test 3: BMI range")
    for value in df['BMI']:
        assert 10 <= value <= 70

    print("Running Test 4: Diabetes_012 values")
    for value in df['Diabetes_012']:
        assert value in [0, 1, 2]

    print("Running Test 5: Sex values")
    for value in df['Sex']:
        assert value in ['Male', 'Female']

    print("All tests passed!")

if __name__ == '__main__':
    main()