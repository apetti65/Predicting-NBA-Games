import pandas as pd

file = "games.csv"
df = pd.read_csv(file)
df2021 = pd.read_csv('2021.csv')
df2020 = pd.read_csv('2020.csv')
df2019 = pd.read_csv('2019.csv')
df2018 = pd.read_csv('2018.csv')
df2017 = pd.read_csv('2017.csv')
df2016 = pd.read_csv('2016.csv')
df2015 = pd.read_csv('2015.csv')

for index, row in df.iterrows():
    year = locals()["df"+str(row['SEASON'])]
    home = row['HOME_TEAM_ID']
    away = row['VISITOR_TEAM_ID']
    df.loc[index, 'HOME_ORTG'] = (year.loc[year['Team']==home, 'ORtg']).iloc[0]
    df.loc[index, 'AWAY_ORTG'] = (year.loc[year['Team']==away, 'ORtg']).iloc[0]
    df.loc[index, 'HOME_DRTG'] = (year.loc[year['Team']==home, 'DRtg']).iloc[0]
    df.loc[index, 'AWAY_DRTG'] = (year.loc[year['Team']==away, 'DRtg']).iloc[0]
    df.loc[index, 'HOME_SRS'] = (year.loc[year['Team']==home, 'SRS']).iloc[0]
    df.loc[index, 'AWAY_SRS'] = (year.loc[year['Team']==away, 'SRS']).iloc[0]
    df.loc[index, 'HOME_EFG'] = (year.loc[year['Team']==home, 'eFG%']).iloc[0]
    df.loc[index, 'AWAY_EFG'] = (year.loc[year['Team']==away, 'eFG%']).iloc[0]
    df.loc[index, 'HOME_TS'] = (year.loc[year['Team']==home, 'TS%']).iloc[0]
    df.loc[index, 'AWAY_TS'] = (year.loc[year['Team']==away, 'TS%']).iloc[0]
    df.loc[index, 'MOV'] = row['PTS_home'] - row['PTS_away']
df.to_csv('mov_games.csv', index=False)