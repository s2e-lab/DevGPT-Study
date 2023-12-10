# Your existing AbstractStorage, DiskDict, and JSONlStorage here...

class CachedJSONlStorage(JSONlStorage):
    def __init__(self, root_dir='serialized_data', cache_size=1000):
        super().__init__(root_dir)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_dirty = set()  # Keep track of keys that need to be written to disk

    def _check_cache_limit(self):
        if len(self.cache) > self.cache_size:
            oldest_key = list(self.cache.keys())[0]
            if oldest_key in self.cache_dirty:
                super().put(oldest_key, self.cache[oldest_key])
            self.cache.pop(oldest_key)
            self.cache_dirty.discard(oldest_key)

    def get(self, key):
        if key in self.cache:
            return self.cache[key]

        value = super().get(key)
        if value is not None:
            self._check_cache_limit()
            self.cache[key] = value
        return value

    def put(self, key, value):
        # Mark this key as dirty (modified)
        self.cache_dirty.add(key)

        # Update the cache
        if key in self.cache:
            self.cache[key].append(value)
        else:
            existing_values = super().get(key) or []
            existing_values.append(value)
            self._check_cache_limit()
            self.cache[key] = existing_values

    def close(self):
        # Write only the "dirty" keys to disk
        for key in self.cache_dirty:
            super().put(key, self.cache[key])
        super().close()
        self.cache_dirty.clear()
