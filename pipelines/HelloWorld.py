from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        pass  # Pas de config pour ce test

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {"id": "hello-world", "name": "Hello World Agent"}
        ]

    def pipe(self, body: dict):
        model = body.get("model", "unknown")
        return f"Bonjour depuis le Pipe ! Modèle sélectionné: {model} [web:9][web:20]"
