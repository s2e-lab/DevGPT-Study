import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao

public class Main {
    public interface StickerDataDao {
        @Insert
        void insertStickerData(StickerData stickerData);

        @Query("SELECT * FROM sticker_data")
        List<StickerData> getAllStickerData();
    }
}