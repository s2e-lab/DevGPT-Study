import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KeywordMatcher {
    public static void main(String[] args) {
        // 원본 데이터 배열
        String[] originalData = {"apple", "banana", "cherry", "date", "elderberry"};

        // 대조할 키워드
        String[] keywords = {"banana", "date", "fig"};

        // 대조한 결과를 저장할 리스트
        List<String> A = new ArrayList<>();

        // 원본 데이터에서 각 문자열을 꺼내어 대조
        for (String data : originalData) {
            for (String keyword : keywords) {
                if (data.contains(keyword)) {
                    // 대조한 키워드를 포함하는 경우, 해당 문자열을 A에 추가
                    A.add(data);
                    break; // 이미 찾았으므로 더 이상 검색하지 않음
                }
            }
        }

        // A에 저장된 결과 출력
        System.out.println("A: " + A);
    }
}
