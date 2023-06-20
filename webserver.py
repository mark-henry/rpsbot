import tensorflow as tf
import numpy as np

# Define the data schema for the history of turns
history_schema = tf.TensorSpec(shape=(None, 2), dtype=tf.int32)

# Define the function names for the AI model's inference and training
inference_fn_name = "inference"
training_fn_name = "train"

# Define the web server
class WebServer:
    def __init__(self, model, history_length):
        self.model = model
        self.history_length = history_length

    def run(self):
        html = """
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
            // Define DOM element IDs for the user input and AI output display
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
                    aiMove.textContent = "AI chose " + data.move;
                });
            }

            // Add event listeners to the buttons
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

        # Define the web server routes
        @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
        def serve_html(request_body):
            return html

        @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
        def play(request_body):
            # Parse the user's move from the request body
            data = tf.io.decode_json_example(request_body)
            user_move = data["move"]

            # Get the AI's move
            ai_move = self.model.predict(np.array([self.history]))

            # Update the history with the user's move and the AI's move
            self.history = np.append(self.history, [[self.move_to_int(user_move), ai_move]], axis=0)
            self.history = self.history[-self.history_length:]

            # Train the model on the updated history
            self.model.train_on_batch(self.history[:-1], self.history[1:])

            # Return the AI's move
            return {"move": self.int_to_move(ai_move)}

        # Define the web server
        server = tf.saved_model.experimental.WebModel(
            serve=serve_html,
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)],
            output_signature=tf.TensorSpec(shape=None, dtype=tf.string)
        )
        server.compile(optimizer=tf.keras.optimizers.Adam())

        # Initialize the history with random moves
        self.history = np.random.randint(0, size=(self.history_length, 2))

        # Start the web server
        server.run('localhost:8080')

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


WebServer(RPSModel(10), 10)