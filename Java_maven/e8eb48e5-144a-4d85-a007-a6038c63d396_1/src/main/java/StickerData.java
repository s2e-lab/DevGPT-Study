import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "sticker_data")
public class StickerData {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private String packName;
    private String creatorName;
    private String packIcon;
    private List<Uri> stickerList;

    public StickerData(String packName, String creatorName, String packIcon, List<Uri> stickerList) {
        this.packName = packName;
        this.creatorName = creatorName;
        this.packIcon = packIcon;
        this.stickerList = stickerList;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getPackName() {
        return packName;
    }

    public String getCreatorName() {
        return creatorName;
    }

    public String getPackIcon() {
        return packIcon;
    }

    public List<Uri> getStickerList() {
        return stickerList;
    }
}
