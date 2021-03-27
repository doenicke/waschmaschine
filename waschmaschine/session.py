from pandas import DataFrame
import config


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
    print("Dropped short sessions:", dropped_sessions)
    return sessions_new


def print_session_list(sessions):
    for idx, session in enumerate(sessions):
        print(idx, session.iloc[0, 0], len(session))


def mean_duration(sessions):
    m = 0
    for session in sessions:
        m += len(session)
    return m / len(sessions)


if __name__ == '__main__':
    import waschmaschine.mysql
    df = waschmaschine.mysql.load_cache_file(config.db_cache_file)

    sessions = split_into_sessions(df)
    print("Sessions total:", len(sessions))

    sessions_filtered = drop_short_sessions(sessions)
    print("Sessions total:", len(sessions_filtered))

    print_session_list(sessions_filtered)
    print("Mean duration:", mean_duration(sessions_filtered))
