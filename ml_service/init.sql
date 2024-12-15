CREATE TABLE user_actions(
    user_id integer,
    movie_id integer
);

CREATE TABLE user_recomend(
    user_id integer,
    movie_id integer,
    rank integer
);

CREATE TABLE users(
    user_id SERIAL PRIMARY KEY
);