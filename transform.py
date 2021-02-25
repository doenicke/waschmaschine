import pandas as pd

import config


def split_into_train_df(df_src, n_features):
    """
    Jeder Waschvorgang wird in n Abschnitte unterteilt.
    Jeder Abschnitt ist dann n Minuten lang.
    Hinzu kommt noch das Feature der Betriebszeit.

    :param df_src:
    :param n_features: Anzahl der Features mit Messwerten
    :return:
    """
    data_list = []
    for row in range(len(df_src) - n_features + 1):
        df_tmp = df_src.iloc[row:row + n_features]
        # print(df)
        row = [list(df_tmp['betrieb'])[-1]] + list(df_tmp['watt']) + [list(df_tmp['rest'])[-1]]
        # print(row)
        data_list.append(row)
    return data_list


def split_into_session_df(df_src):
    """
    DataFrame mit allen Waschvorgängen auftrennen und unrelevante Messwerte verwerfen:
    """
    df_sessions = []
    watt = []
    ende = False
    for index, row in df_src.iterrows():
        value = row['value']
        if value > 1:
            watt.append(value)
            ende = False
        elif value <= 1 and not ende:

            if len(watt) > config.duration_min:  # Nur Vorgänge berücksichtigen, die länger als x Minuten dauern
                watt.append(value)
                watt = [0]*config.n_features + watt
                rest = list(range(len(watt) - 1, -1, -1))
                betrieb = [0] * config.n_features + list(range(0, len(watt)-config.n_features))
                print('Watt:', watt, len(watt))
                print('Betrieb:', betrieb, len(betrieb))
                print('Rest:', rest, len(rest))
                print('')

                session_dict = {'watt': watt, 'betrieb': betrieb, 'rest': rest}
                df_sessions.append(pd.DataFrame(session_dict))

            watt = []
            ende = True

    return df_sessions


def transform_data(df):
    # Create list of washing sessions (list of Dataframes):
    df_sessions = split_into_session_df(df)

    # For prediction we create a wide table (Dataframe) having following features:
    # past minutes and n watt (e.g. 60 watt values per 60 minutes)
    # Label is the resuming time:
    cols_features = ['betrieb'] + list(range(config.n_features))
    cols_label = ['rest']
    df = pd.DataFrame(columns=cols_features+cols_label)

    # Each washing session gets this layout:
    for df_session in df_sessions:
        data = split_into_train_df(df_session, config.n_features)
        df_tmp = pd.DataFrame(data, columns=cols_features+cols_label)
        df = df.append(df_tmp, ignore_index=True)

    print(df)
    print('Count washing sessions:', len(df_sessions))
    return df[cols_features], df[cols_label]