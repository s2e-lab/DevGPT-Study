import java.util.*;

public class MapSortingExample {
    public static void main(String[] args) {
        // Create a sample map
        Map<Integer, Integer> map = new HashMap<>();
        map.put(1, 10);
        map.put(2, 5);
        map.put(3, 20);
        map.put(4, 15);
        map.put(5, 8);

        // Sort the map by values
        Map<Integer, Integer> sortedMap = sortByValue(map);

        // Print the sorted map
        for (Map.Entry<Integer, Integer> entry : sortedMap.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    public static Map<Integer, Integer> sortByValue(Map<Integer, Integer> map) {
        List<Map.Entry<Integer, Integer>> entryList = new ArrayList<>(map.entrySet());

        // Sort the list based on values using Comparator
        entryList.sort(Map.Entry.comparingByValue());

        // Create a new LinkedHashMap to preserve the order of the sorted entries
        Map<Integer, Integer> sortedMap = new LinkedHashMap<>();
        for (Map.Entry<Integer, Integer> entry : entryList) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }
}
