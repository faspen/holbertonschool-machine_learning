-- Valid email trigger
DELIMITER //

CREATE TRIGGER reset_validation
BEFORE UPDATE
ON users
FOR EACH ROW
    IF (NEW.email != OLD.email) THEN
        SET NEW.valid_email = 0;
    END IF//

DELIMITER ;