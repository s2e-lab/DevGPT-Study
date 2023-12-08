import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ImmutableMapExample {
    public static void main(String[] args) {
        // Create a mutable map
        Map<String, Integer> mutableMap = new HashMap<>();
        mutableMap.put("key1", 1);
        mutableMap.put("key2", 2);
        mutableMap.put("key3", 3);

        // Create an immutable map
        Map<String, Integer> immutableMap = Collections.unmodifiableMap(mutableMap);

        // Try to modify the immutable map (will throw an exception)
        try {
            immutableMap.put("key4", 4); // This will throw an UnsupportedOperationException
        } catch (UnsupportedOperationException e) {
            System.out.println("Cannot modify the immutable map!");
        }

        // Original mutable map is still modifiable
        mutableMap.put("key4", 4);
        System.out.println("Mutable map: " + mutableMap);
        System.out.println("Immutable map: " + immutableMap);
    }
}
