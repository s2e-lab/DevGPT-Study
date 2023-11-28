public class Main {
    public static void main(string [] args) {
        Vector2 gridPos = gridLayoutGroup.GetComponent<RectTransform>().anchoredPosition;
        Vector2 pos = firstSlotRect.anchoredPosition + gridPos;
        shadowObject.GetComponent<RectTransform>().anchoredPosition = pos;

   }
}
