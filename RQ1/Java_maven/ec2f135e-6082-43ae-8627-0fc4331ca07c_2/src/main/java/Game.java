package org.example.rpsarena;

import java.util.Random;

public class Game {
    private MainActivity activity;
    private Player player;
    private Player opponent;
    private GameLogic gameLogic;

    public Game(MainActivity activity) {
        this.activity = activity;
        this.gameLogic = new GameLogic();
        setupGame();
    }

    private void setupGame() {
        player = new Player("Player 1");
        opponent = new Player("Player 2");
        player.setOpponent(opponent);
        opponent.setOpponent(player);
        updateScores();
    }

    public void onMoveSelected(String move) {
        if (move.equalsIgnoreCase("exit")) {
            exitGame();
        } else {
            Moves playerMove = convertToMove(move);
            if (playerMove == null) {
                activity.updateMovesText("Invalid move. Please try again.");
            } else {
                Moves opponentMove = generateOpponentMove();
                String result = gameLogic.determineWinner(playerMove, opponentMove);
                updateScores(result);
                activity.updateMovesText("Your Move: " + move + "\nOpponent Move: " + opponentMove);
            }
        }
    }

    private Moves convertToMove(String input) {
        try {
            return Moves.valueOf(input.toUpperCase());
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private Moves generateOpponentMove() {
        Moves[] moves = Moves.values();
        Random random = new Random();
        return moves[random.nextInt(moves.length)];
    }

    private void updateScores(String result) {
        if (result.equals("WIN")) {
            player.incrementPoints();
            activity.updateScores(player.getPlayerPoints(), opponent.getPlayerPoints());
        } else if (result.equals("LOSS")) {
            opponent.incrementPoints();
            activity.updateScores(player.getPlayerPoints(), opponent.getPlayerPoints());
        }
    }

    private void updateScores() {
        activity.updateScores(player.getPlayerPoints(), opponent.getPlayerPoints());
    }

    public void exitGame() {
        // Any cleanup or additional actions required on game exit
    }
}
