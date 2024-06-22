from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count_cows')
def count_cows():
    from count_cows import run_yolo
    output_path, cow_count = run_yolo()
    return render_template('result_cows.html', output_path=output_path, counter=cow_count)


@app.route('/count_chickens')
def count_chickens():
    from count_chickens import run_yolo
    output_path, chicken_count = run_yolo()
    return render_template('result_chickens.html', output_path=output_path, counter=chicken_count)

@app.route('/count_sheeps')
def count_sheeps():
    from count_sheeps import run_yolo
    output_path, sheep_coun = run_yolo()
    return render_template('result_sheeps.html', output_path=output_path, counter=sheep_coun)

if __name__ == '__main__':
    app.run(debug=True)
