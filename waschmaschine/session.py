from pandas import DataFrame
import waschmaschine.mysql
import logging
import logging_cfg
import config

logging_cfg.init_config()
logger = logging.getLogger(__name__)


def split_into_sessions(df):
    sessions = []
    session = []

    for index, row in df.iterrows():
        if row['value'] > config.standby_watt:
            session.append(row)
        elif len(session) > 0:
            sessions.append(DataFrame(session))
            session = []

    if len(session) > 0:
        sessions.append(DataFrame(session))
    logger.debug(f"Split sessions: {len(sessions)}")
    return sessions


def drop_short_sessions(sessions):
    sessions_new = []
    dropped_sessions = 0
    for session in sessions:
        duration = len(session)
        if duration >= config.duration_min:
            sessions_new.append(session)
        else:
            dropped_sessions += 1
    logger.debug(f"Dropped short sessions: {dropped_sessions}")
    return sessions_new


def print_session_list(sessions):
    for idx, session in enumerate(sessions):
        print(idx, session.iloc[0, 0], len(session))


def mean_duration(sessions):
    m = 0
    for session in sessions:
        m += len(session)
    return m / len(sessions)


def transform_session(df, n_features):
    """
    Jeder Waschvorgang wird in n Abschnitte unterteilt.
    Jeder Abschnitt ist dann n Minuten lang.
    Hinzu kommt noch das Feature der Betriebszeit.

    :param df:
    :param n_features: Anzahl der Features mit Messwerten
    :return:
    """
    data_list = []
    for row in range(len(df) - n_features + 1):
        df_tmp = df.iloc[row:row + n_features]
        print(df_tmp)
        row = [list(df_tmp['betrieb'])[-1]] + list(df_tmp['watt']) + [list(df_tmp['rest'])[-1]]
        # print(row)
        data_list.append(row)
    return data_list


if __name__ == '__main__':
    df = waschmaschine.mysql.load_cache_file(config.db_cache_file)

    sessions = split_into_sessions(df)
    print("Sessions total:", len(sessions))

    sessions_filtered = drop_short_sessions(sessions)
    print("Sessions total:", len(sessions_filtered))

    print_session_list(sessions_filtered)
    print("Mean duration:", mean_duration(sessions_filtered))

    print("Transform last session:")
    session_trans = transform_session(sessions_filtered[-1], config.n_features)
    print(session_trans)