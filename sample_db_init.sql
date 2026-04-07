-- ─────────────────────────────────────────────────────────────────────────────
-- sample_db_init.sql
--
-- Creates a realistic e-commerce schema with intentionally slow queries
-- so the agent has meaningful work to do from the start.
--
-- Usage:
--   psql $DATABASE_URL -f sample_db_init.sql
--   or loaded automatically by docker-compose via initdb
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable the hint plan extension
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- ── Schema ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS regions (
    region_id   SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    region_id   INT REFERENCES regions(region_id),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    product_id  SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       NUMERIC(10, 2) NOT NULL,
    stock       INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    status      TEXT NOT NULL DEFAULT 'pending',
    total       NUMERIC(10, 2),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id     SERIAL PRIMARY KEY,
    order_id    INT REFERENCES orders(order_id),
    product_id  INT REFERENCES products(product_id),
    quantity    INT NOT NULL,
    unit_price  NUMERIC(10, 2) NOT NULL
);

-- ── Indexes (deliberately sparse so agent can find missing ones) ──────────────

CREATE INDEX IF NOT EXISTS idx_orders_customer_id  ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
-- Intentionally no index on orders.status or customers.region_id

-- ── Seed data ─────────────────────────────────────────────────────────────────

INSERT INTO regions (name, country)
SELECT
    'Region ' || i,
    CASE WHEN i % 3 = 0 THEN 'US'
         WHEN i % 3 = 1 THEN 'EU'
         ELSE 'APAC' END
FROM generate_series(1, 10) i
ON CONFLICT DO NOTHING;

INSERT INTO customers (name, email, region_id)
SELECT
    'Customer ' || i,
    'customer_' || i || '@example.com',
    (i % 10) + 1
FROM generate_series(1, 10000) i
ON CONFLICT DO NOTHING;

INSERT INTO products (name, category, price, stock)
SELECT
    'Product ' || i,
    CASE WHEN i % 4 = 0 THEN 'Electronics'
         WHEN i % 4 = 1 THEN 'Clothing'
         WHEN i % 4 = 2 THEN 'Books'
         ELSE 'Home' END,
    (random() * 500 + 5)::NUMERIC(10,2),
    (random() * 1000)::INT
FROM generate_series(1, 1000) i
ON CONFLICT DO NOTHING;

INSERT INTO orders (customer_id, status, total, created_at)
SELECT
    (random() * 9999 + 1)::INT,
    CASE WHEN random() < 0.6 THEN 'completed'
         WHEN random() < 0.8 THEN 'pending'
         ELSE 'cancelled' END,
    (random() * 2000 + 10)::NUMERIC(10,2),
    NOW() - (random() * INTERVAL '365 days')
FROM generate_series(1, 50000) i
ON CONFLICT DO NOTHING;

INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT
    (random() * 49999 + 1)::INT,
    (random() * 999 + 1)::INT,
    (random() * 5 + 1)::INT,
    (random() * 500 + 5)::NUMERIC(10,2)
FROM generate_series(1, 200000) i
ON CONFLICT DO NOTHING;

-- ── Update stats so EXPLAIN ANALYZE is meaningful ─────────────────────────────
ANALYZE regions, customers, products, orders, order_items;