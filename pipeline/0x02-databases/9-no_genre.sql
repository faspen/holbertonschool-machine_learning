-- Display tv shows with no genre id
SELECT tv_shows.title, tv_genres.id AS genre_id FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
LEFT JOIN tv_genres ON tv_genres.id = tv_show_genres.genre_id
WHERE tv_genres.id IS NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;