-- Buy buy buy
CREATE TRIGGER deplete_items
AFTER INSERT ON orders 
FOR EACH ROW
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE items.name = NEW.item_name