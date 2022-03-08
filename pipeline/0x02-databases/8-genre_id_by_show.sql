-- List tv shows by title and genre id
SELECT tv_shows.title, tv_genres.id AS genre_id FROM tv_show_genres
INNER JOIN tv_shows ON tv_show_genres.show_id = tv_shows.id
INNER JOIN tv_genres ON tv_show_genres.genre_id = tv_genres.id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;