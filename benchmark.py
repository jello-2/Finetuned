import json
import ollama


contest_instructions = "This is a 25-question, multiple choice test. Each question is followed by answers marked A, B, C, D and E. Only one of these is correct. You will receive 6 points for a correct answer, and 0 points for each incorrect answer, for a maximum of 150 points. No aids are permitted other than scratch paper, graph paper, ruler, compass, protractor and erasers (and calculators that are accepted for use on the test if before 2006. No problems on the test will <i>require</i> the use of a calculator). Avoid unecessary brute forcing. Figures are not necessarily drawn to scale."
answer_instructions = "You are to reply your answer with a box, with the label \\boxed{} of the letter corresponding to the answer choice you are most confident with."


def load():
    with open("key_pair.json", "r", encoding = "utf-8") as f:
        return json.load(f)
        


# testing loop

current_score = 0



response = ollama.chat(model = 'deepseek-r1:14b',messages = [{
    'role':'user',
    'content':f'{contest_instructions} {answer_instructions} The question is given as follows: This is just a test'
}])

print(response['message']['content'])