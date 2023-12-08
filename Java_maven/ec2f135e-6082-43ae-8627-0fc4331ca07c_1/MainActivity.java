package org.example.rpsarena;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private TextView playerScoreTextView;
    private TextView opponentScoreTextView;
    private TextView movesTextView;

    private Game game;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        playerScoreTextView = findViewById(R.id.playerScoreTextView);
        opponentScoreTextView = findViewById(R.id.opponentScoreTextView);
        movesTextView = findViewById(R.id.movesTextView);

        game = new Game(this);
    }

    public void updateScores(int playerScore, int opponentScore) {
        playerScoreTextView.setText("Your Score: " + playerScore);
        opponentScoreTextView.setText("Opponent Score: " + opponentScore);
    }

    public void updateMovesText(String move) {
        movesTextView.setText("Your Move: " + move);
    }

    public void onMoveButtonClicked(View view) {
        Button button = (Button) view;
        String move = button.getText().toString();
        game.onMoveSelected(move);
    }

    public void onExitButtonClicked(View view) {
        game.exitGame();
        finish();
    }
}
