import tensorflow as tf
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from flask import Flask, request, jsonify


class RPSModel:
    def __init__(self, history_length, learning_rate=0.001):
        self.history_length = history_length
        self.learning_rate = learning_rate
        self.num_actions = 3  # rock, paper, and scissors :)
        self.model = self.build_model()


    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.history_length, 2))
        lstm_layer = tf.keras.layers.LSTM(128)(input_layer)
        dense_layer = tf.keras.layers.Dense(64, activation='relu')(lstm_layer)
        output_layer = tf.keras.layers.Dense(self.num_actions, activation='softmax')(dense_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model


    def predict(self, history):
        history = np.array(history[-self.history_length:])
        history = np.expand_dims(history, axis=0)
        prediction = self.model.predict(history)
        return np.argmax(prediction)

    def train(self, history, action):
        history = np.expand_dims(
            np.array(history[-self.history_length:]), axis=0)
        target = np.zeros((1, self.num_actions))
        target[0, action] = 1
        self.model.train_on_batch(history, target)

    def move_to_int(self, move):
        if move == "rock":
            return 0
        elif move == "paper":
            return 1
        elif move == "scissors":
            return 2

    def int_to_move(self, move):
        if move == 0:
            return "rock"
        elif move == 1:
            return "paper"
        elif move == 2:
            return "scissors"


max_history_len = 5
history = np.random.randint(3, size=(max_history_len, 2))
app = Flask(__name__)
model = RPSModel(max_history_len)


@app.route("/", methods=["GET"])
def index():
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rock Paper Scissors</title>
        </head>
        <body>
            <h1>Rock Paper Scissors</h1>
            <p>Choose your move:</p>
            <button id="rock">Rock</button>
            <button id="paper">Paper</button>
            <button id="scissors">Scissors</button>
            <p id="ai-move"></p>
        </body>
        <script>
            const rockButton = document.getElementById("rock");
            const paperButton = document.getElementById("paper");
            const scissorsButton = document.getElementById("scissors");
            const aiMove = document.getElementById("ai-move");

            // Send the user's move to the server and display the AI's move
            function sendMove(move) {
                fetch("/play", {
                    method: "POST",
                    body: JSON.stringify({ move: move }),
                    headers: {
                        "Content-Type": "application/json"
                    }
                })
                .then(response => response.json())
                .then(data => {
                    const aimove = data.move;
                    aiMove.textContent = "AI chose " + aimove;
                    
                    const usermove = move;
                    const outcome = determineOutcome(usermove, aimove);
                    
                    const result = document.createElement("p");
                    result.textContent = "You chose " + usermove + ". " + outcome;
                    aiMove.parentNode.insertBefore(result, aiMove.nextSibling);
                });
            }

            function determineOutcome(userChoice, aiChoice) {
                if (userChoice === aiChoice) {
                    return "It's a tie!";
                } else if (
                    (userChoice === "rock" && aiChoice === "scissors") ||
                    (userChoice === "paper" && aiChoice === "rock") ||
                    (userChoice === "scissors" && aiChoice === "paper")
                ) {
                    return "You win!";
                } else {
                    return "You lose!";
                }
            }

            rockButton.addEventListener("click", () => {
                sendMove("rock");
            });
            paperButton.addEventListener("click", () => {
                sendMove("paper");
            });
            scissorsButton.addEventListener("click", () => {
                sendMove("scissors");
            });
        </script>
        </html>
        """


@app.route("/play", methods=["POST"])
def play():
    move = request.json["move"]
    ai_move = process_move(move)
    response_data = {"move": ai_move}
    return jsonify(response_data)

def process_move(user_move):
    global history
    global max_history_len

    prediction = model.predict(history)
    ai_move = (prediction + 2) % 3

    # Update the history with the user's move and the AI's move
    history = np.append(history, [[model.move_to_int(user_move), ai_move]], axis=0)

    # Train the model on the updated history
    model.train(history[:-1], history[-1:])

    return model.int_to_move(ai_move)


if __name__ == "__main__":
    app.run()  # http://localhost:5000