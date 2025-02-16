CREATE TABLE IF NOT EXISTS
    models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        architecture TEXT,
        type TEXT NOT NULL,
        hash TEXT,
        source_url TEXT
    );

CREATE TABLE IF NOT EXISTS
    tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );

CREATE TABLE IF NOT EXISTS
    model_tags (
        model_id INTEGER NOT NULL,
        tag_id INTEGER NOT NULL,
        PRIMARY KEY (model_id, tag_id),
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
    );

INSERT INTO
    tags (name)
VALUES
    ('character'),
    ('style'),
    ('concept'),
    ('clothing'),
    ('poses'),
    ('background'),
    ('vehicle'),
    ('buildings'),
    ('objects'),
    ('animal'),
    ('action');
