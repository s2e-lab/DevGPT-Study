import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> extends LinkedHashMap<K, V> {
    private final int maxSize;

    public LRUCache(int maxSize) {
        // Initial capacity and load factor are default. The third parameter 'true' means that this
        // LinkedHashMap will be in access-order, least recently accessed first.
        super(16, 0.75f, true);
        this.maxSize = maxSize;
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > maxSize;
    }

    public static void main(String[] args) {
        LRUCache<Integer, String> cache = new LRUCache<>(3);

        cache.put(1, "one");
        cache.put(2, "two");
        cache.put(3, "three");
        System.out.println(cache);  // Output will be in access order: {1=one, 2=two, 3=three}

        cache.get(1);  // Access the eldest entry to make it most recently used
        System.out.println(cache);  // Output: {2=two, 3=three, 1=one}

        cache.put(4, "four");  // This will evict the least recently used entry, which is "2=two"
        System.out.println(cache);  // Output: {3=three, 1=one, 4=four}
    }
}
