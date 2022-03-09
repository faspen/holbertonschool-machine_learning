-- Best genre
SELECT tv_genres.name AS name, SUM(tv_show_ratings.rate) as rating
FROM tv_show_ratings LEFT JOIN tv_shows ON tv_shows.id = tv_show_ratings.show_id
INNER JOIN tv_show_genres ON tv_show_genres.show_id = tv_shows.id
INNER JOIN tv_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY name ORDER BY rating DESC;