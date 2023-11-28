import android.net.Uri;
import java.io.InputStream;

public class Main {
  public void readFile(String uriString, Promise promise) {
    Uri uri = Uri.parse(uriString);
    try {
      InputStream is = getReactApplicationContext().getContentResolver().openInputStream(uri);
      // Here you can read from the InputStream as needed.
      // Don't forget to close it once you're done!
      is.close();
    } catch (Exception e) {
      promise.reject(e);
    }
  }
}