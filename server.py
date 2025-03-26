from flask import Flask , request , jsonify
from flask_cors import CORS
app = Flask(__name__)

default_story = "The corridors of Hogwarts were eerily quiet at midnight, the torches flickering dimly as Harry crept past the portrait of the Fat Lady. He wasn't supposed to be out, but he had overheard something strange in the Great Hall earlier—a whispering voice calling his name from the shadows. Ron and Hermione had dismissed it. “Probably Peeves messing around,” Ron had said, but Harry knew better. There was something in the castle tonight, and it wanted him to find it."

@app.route("/llama",methods = ['POST'])
def generate_story1():
  data = request.get_json()
  print(data)
  return jsonify(default_story)

@app.route("/gpt",methods = ['POST'])
def generate_story2():
  data = request.get_json()
  print(data)
  return jsonify(default_story)

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)