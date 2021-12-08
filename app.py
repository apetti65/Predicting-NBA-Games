import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
import math

def main():
    file = "2021.csv"
    df = pd.read_csv(file)
    while(True):
        hteam = input("Input Home Team: ")
        ateam = input("Input Away Team: ")
        hteamdf = df.loc[df['Team'] == hteam]
        ateamdf = df.loc[df['Team'] == ateam]
        if(hteamdf.empty or ateamdf.empty or ateam==hteam):
            print('Please enter valid NBA teams for Home and Away fields')
        else:
            break
    hteamdf = hteamdf[['ORtg', 'DRtg', 'SRS', 'eFG%', 'TS%']]
    ateamdf = ateamdf[['ORtg', 'DRtg', 'SRS', 'eFG%', 'TS%']]
    cdf = pd.DataFrame()
    cdf.loc[0, 'HOME_ORTG'] = (hteamdf.iloc[0,0])
    cdf.loc[0, 'AWAY_ORTG'] = (ateamdf.iloc[0,0])
    cdf.loc[0, 'HOME_DRTG'] = (hteamdf.iloc[0,1])
    cdf.loc[0, 'AWAY_DRTG'] = (ateamdf.iloc[0,1])
    cdf.loc[0, 'HOME_SRS'] = (hteamdf.iloc[0,2])
    cdf.loc[0, 'AWAY_SRS'] = (ateamdf.iloc[0,2])
    cdf.loc[0, 'HOME_EFG'] = (hteamdf.iloc[0,3])
    cdf.loc[0, 'AWAY_EFG'] = (ateamdf.iloc[0,3])
    cdf.loc[0, 'HOME_TS'] = (hteamdf.iloc[0,4])
    cdf.loc[0, 'AWAY_TS'] = (ateamdf.iloc[0,4])
    values = np.array(cdf)
    cmodel = keras.models.load_model('classification_model.h5')
    rmodel = keras.models.load_model('regression_model.h5')
    home_wins = cmodel.predict(values)
    home_wins = np.argmax(home_wins, axis=-1).astype('int')
    rdf = pd.DataFrame()
    rdf.loc[0, 'HOME_TEAM_WINS'] = home_wins[0]
    rdf = rdf.join(cdf)
    regvalues = np.array(rdf)
    mov = rmodel.predict(regvalues)[0][0]
    print(mov)
    if mov < 1 and mov > 0:
        mov = math.ceil(mov)
    elif mov > -1 and mov < 0:
        mov = math.floor(mov)
    if (home_wins[0] == 0):
        print('The '+ ateam +' will win by '+ str(abs(round(mov))) +' points.')
    elif (home_wins[0] == 1):
        print('The '+ hteam +' will win by '+ str(abs(round(mov))) +' points.')


if __name__ == '__main__':
    main()