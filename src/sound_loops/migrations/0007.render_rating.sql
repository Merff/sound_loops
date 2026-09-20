ALTER TABLE renders ADD COLUMN rating TEXT CHECK (rating IN ('good', 'neutral', 'bad'));
