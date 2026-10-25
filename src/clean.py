import pandas as pd

#todo convert laptime to seconds
def to_seconds(lap_time):
    lap_time = str(lap_time)

    if ":" in lap_time:
        m, s = lap_time.split(":")
        return int(m) * 60 + float(s)
    
    return float(lap_time)

#* clean dataset
def clean(df, keep_flagged=True, verbose=True):

    n0 = len(df)
    df = df.copy()

    df.columns = df.columns.str.lower()
    
    #? apply conversion to lap time
    df['laptime'] = df['laptime'].map(to_seconds)
    df = df.sort_values(['raceid', 'driver', 'lap']).reset_index(drop=True) #~ sort

    #! flag special laps (1st, pit, out lap, safety car etc)
    df['is_in_lap'] = df['pit_duration'].notna().astype(int)
    df['is_out_lap'] = (df.groupby(['raceid', 'driver'])['stint'].diff().fillna(0) != 0).astype(int)
    df['is_lap1'] = (df['lap'] == 1).astype(int)
    flagged = df[['is_in_lap', 'is_out_lap', 'is_lap1']].sum(axis=1) > 0

    #^ reference pace (from normal laps only)
    race_median = df[~flagged].groupby('raceid')['laptime'].median()
    ref = df['raceid'].map(race_median)

    #& safety car, redflag and traffic outliers
    limit = pd.Series(1.07, index=df.index)
    limit[flagged] = 1.20

    within_limit = df['laptime'] <= limit * ref
    to_keep = within_limit if keep_flagged else (within_limit & ~flagged)

    output_df = df[to_keep].copy()
    if verbose:
        print(f'start {n0} -> kept {len(output_df)} | flagged kept : {int(flagged[to_keep].sum())}')

    return output_df

if __name__ == "__main__":
    raw = pd.read_csv(r'C:\Users\anasq\OneDrive\Desktop\Projects\F1 Lap Time Predictor\data\raw\raw_data.csv')
    
    clean(raw, keep_flagged=False).to_csv(r'data/processed/baseline_pace.csv', index=False)
    clean(raw, keep_flagged=True).to_csv(r'data/processed/race_telemetry.csv', index=False)