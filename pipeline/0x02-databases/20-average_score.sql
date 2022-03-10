-- Procedure for average score
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (user_id INT)
    BEGIN
        SET @avg = (SELECT AVG(corrections.score) FROM corrections WHERE corrections.user_id = user_id);
        UPDATE users SET average_score = @avg WHERE id = user_id;

    END; //

DELIMITER ;