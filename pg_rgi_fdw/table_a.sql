\echo [session A] writing to the SHARED kv_rgi table (via worker)
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,1000) g;
INSERT INTO kv_rgi VALUES (123456, 42);
