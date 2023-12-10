from django.core.cache import cache

cache.set('test_key', 'test_value', 300)  # Set a cache value with a 5-minute expiry
value = cache.get('test_key')  # Fetch the cache value
print(value)  # Should print 'test_value'
