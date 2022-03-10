-- Create a procedure
DELIMITER //

CREATE PROCEDURE AddBonus (user_id INT, project_name VARCHAR(255), score FLOAT)
    BEGIN
        IF NOT EXISTS(SELECT 1 FROM projects WHERE name = project_name) THEN
            INSERT INTO projects (name) VALUES (project_name);
        END IF;

        SET @project_id = (SELECT id FROM projects WHERE projects.name = project_name);

        INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, @project_id, score);
    END; //

DELIMITER ;