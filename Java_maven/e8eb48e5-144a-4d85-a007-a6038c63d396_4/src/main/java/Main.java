public class Main {
    public static void main(string [] args) {
        StickerDatabase stickerDatabase = Room.databaseBuilder(getApplicationContext(), StickerDatabase.class, "sticker_db").build();
        StickerDataDao stickerDataDao = stickerDatabase.stickerDataDao();

   }
}
