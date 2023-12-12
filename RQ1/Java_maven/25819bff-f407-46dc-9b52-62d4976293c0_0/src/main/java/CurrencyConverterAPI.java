import android.os.AsyncTask;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class CurrencyConverterAPI extends AsyncTask<String, Void, JSONObject> {
    private CurrencyConversionListener listener;

    public CurrencyConverterAPI(CurrencyConversionListener listener) {
        this.listener = listener;
    }

    @Override
    protected JSONObject doInBackground(String... strings) {
        JSONObject response = null;
        try {
            URL url = new URL(strings[0]);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setRequestProperty("X-Api-Key", "YOUR_API_KEY");

            InputStream inputStream = connection.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder stringBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line);
            }
            response = new JSONObject(stringBuilder.toString());

            reader.close();
            inputStream.close();
            connection.disconnect();
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }
        return response;
    }

    @Override
    protected void onPostExecute(JSONObject jsonObject) {
        listener.onConversionComplete(jsonObject);
    }

    public interface CurrencyConversionListener {
        void onConversionComplete(JSONObject response);
    }
}
