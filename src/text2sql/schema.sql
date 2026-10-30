CREATE TABLE stores (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, store_id INTEGER REFERENCES stores(id), order_date TEXT, status TEXT);
CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id), qty INTEGER);
-- country: FR US JP DE CA | category: shoes jackets packs socks tents
-- status: shipped returned cancelled pending | order_date: YYYY-MM-DD in 2024
