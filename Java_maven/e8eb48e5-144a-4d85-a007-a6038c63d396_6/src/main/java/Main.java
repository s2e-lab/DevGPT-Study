public class Main {
    public static void main(string [] args) {
        List<StickerData> allStickerData = stickerDataDao.getAllStickerData();
        for (StickerData data : allStickerData) {
            // Access the data using getter methods
        }

   }
}
