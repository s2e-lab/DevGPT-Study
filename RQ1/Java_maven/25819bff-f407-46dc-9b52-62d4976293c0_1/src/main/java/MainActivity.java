public class MainActivity extends AppCompatActivity implements CurrencyConverterAPI.CurrencyConversionListener {
    // Declare your UI elements and other variables

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize your UI elements and set up event listeners
    }

    public void onConvertButtonClicked(View view) {
        // Get the selected currencies and the amount from the UI elements
        String haveCurrency = ...; // Get the selected "have" currency
        String wantCurrency = ...; // Get the selected "want" currency
        double amount = ...; // Get the amount from the UI element

        String apiUrl = "https://api.api-ninjas.com/v1/convertcurrency?have=" +
                haveCurrency + "&want=" + wantCurrency + "&amount=" + amount;

        CurrencyConverterAPI converterAPI = new CurrencyConverterAPI(this);
        converterAPI.execute(apiUrl);
    }

    @Override
    public void onConversionComplete(JSONObject response) {
        try {
            double newAmount = response.getDouble("new_amount");
            // Process the converted amount as desired (e.g., display it in a TextView)
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
