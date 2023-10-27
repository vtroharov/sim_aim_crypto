import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def analysis_1():
    df = pd.read_csv('ethereum_backup.csv')
    # print(df.head(5))
    print("Count:", df['Final_Value'].count())

    print("AIM Mean:", round(df['Final_Value'].mean(), 2))
    print("AIM Median:", df['Final_Value'].median())
    print("AIM Max:", df['Final_Value'].max())

    print("HODL Mean:", round(df['HODL'].mean(), 2))
    print("HODL Median:", df['HODL'].median())
    print("HODL Max:", df['HODL'].max())

    # print(df.loc[df['Final_Value'] > 150000])

    df.loc[df['Final_Value'] > 180000].to_csv('out_1.csv')


def analysis_2():
    df = pd.read_csv('out_1.csv')
    # print("Count:", df['Final_Value'].count())
    print("Count:", sum(df['Increment'] == 5))

analysis_2()