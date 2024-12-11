import torch
import random
import json
import os

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    config = json.load(f)

def prompt_tokenizer(tokenizer, system_prompt, question, answer):

    bos_token = tokenizer.additional_special_tokens[0]
    eos_token = tokenizer.additional_special_tokens[1]

    prompt_wo_loss = f"{bos_token}system\n{system_prompt}{eos_token}\n{bos_token}user\n{question}{eos_token}\n{bos_token}assistant"
    prompt_w_loss = f"{answer}{eos_token}"

    prompt_wo_loss_id = tokenizer(prompt_wo_loss, return_tensors="pt").input_ids
    prompt_w_loss_id = tokenizer(prompt_w_loss, return_tensors="pt").input_ids
    
    loss_idx = prompt_wo_loss_id.shape[1]

    prompt_id = torch.cat((prompt_wo_loss_id, prompt_w_loss_id), dim=1)

    return prompt_id, loss_idx

def generate(model, tokenizer, prompt, num_return_sequences, temperature=1.0):
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_input = tokenizer(text, return_tensors="pt")
    input_ids = model_input.input_ids.to(model.device)
    attention_mask = model_input.attention_mask.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=config["max_length"], num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)[:, input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs

def generate_question(model, tokenizer, question, num_return_sequences=1):
    prompt = config["generate_question_prompt"].replace("{question}", question)
    generated_questions = generate(model, tokenizer, prompt, num_return_sequences, temperature=config["question_temperature"])
    return generated_questions

def generate_correct_answer(model, tokenizer, generated_questions):
    return [generate(model, tokenizer, generated_question+config["generate_correct_answer_prompt"], num_return_sequences=1, temperature=config["correct_answer_temperature"])[0] for generated_question in generated_questions]


def generate_incorrect_answer(model, tokenizer, question, answer, num_return_sequences=3):
    prompt = config["generate_incorrect_answer_prompt"].replace("{question}", question).replace("{answer}", answer)
    generated_incorrect_answers = generate(model, tokenizer, prompt, num_return_sequences=num_return_sequences, temperature=config["incorrect_answer_temperature"])
    return generated_incorrect_answers

def generate_incorrect_answers(model, tokenizer, generated_questions, generated_answers, num_incorrect_answers=3):
    return [generate_incorrect_answer(model, tokenizer, generated_question, generated_answer, num_return_sequences=num_incorrect_answers) for generated_question, generated_answer in zip(generated_questions, generated_answers)]

def generate_free_text_batch(model, tokenizer, question, num_return_sequences):
    free_text_batch = []
    generated_questions = generate_question(model, tokenizer, question, num_return_sequences)
    generated_answers = generate_correct_answer(model, tokenizer, generated_questions)
    for question, answer in zip(generated_questions, generated_answers):
        free_text_batch.append({"question": question, "answer": answer})
    return free_text_batch

def generate_choice(answer, incorrect_answers):
    choice_list = [chr(ord('A')+i)+'. ' for i in range(len(incorrect_answers)+1)]
    answers = incorrect_answers
    right_choice = random.randint(0, len(incorrect_answers))
    answers.insert(right_choice, answer)
    choice_list = [choice_list[i]+answers[i] for i in range(len(answers))]
    return choice_list, chr(ord('A')+right_choice)

def generate_choice_batch(model, tokenizer, question, num_return_sequences, num_choices=4):
    choice_batch = []
    generated_questions = generate_question(model, tokenizer, question, num_return_sequences)
    generated_answers = generate_correct_answer(model, tokenizer, generated_questions)
    generated_incorrect_answers = generate_incorrect_answers(model, tokenizer, generated_questions, generated_answers, num_choices-1)
    for question, answer, incorrect_answers in zip(generated_questions, generated_answers, generated_incorrect_answers):
        choice_list, right_choice = generate_choice(answer, incorrect_answers)
        choice_batch.append({"question": question, "choice_list": choice_list, "right_choice": right_choice})
    return choice_batch

def test_time_train(model, tokenizer, question, mode, num_return_sequences=3, num_choices=2, show_generated=False):
    assert mode in ["free_text", "choice", "full_choice"]
    system_prompt = config["system_prompt"]
    cal_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    model = model.eval()

    if mode == "free_text":
        generated_batch = generate_free_text_batch(model, tokenizer, question, num_return_sequences)
        model = model.train()
        for generated in generated_batch:
            prompt_id, loss_idx = prompt_tokenizer(tokenizer, system_prompt, generated["question"], generated["answer"])
            prompt_id = prompt_id.to(model.device)
            output = model(prompt_id)

            loss = cal_loss(output.logits[:, loss_idx+1:].squeeze(), prompt_id[:, loss_idx:-1].squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if show_generated:
                print("generated_question:\t", generated["question"])
                print("generated_answer:\t", generated["answer"])

        if show_generated:
            print()

    elif mode == "choice":
        generated_batch = generate_choice_batch(model, tokenizer, question, num_return_sequences, num_choices)
        model = model.train()
        for generated in generated_batch:
            generated_question_with_choice = generated["question"]+'\n'+'\n'.join(generated["choice_list"])+config["choice_prompt"]
            prompt_id, loss_idx = prompt_tokenizer(tokenizer, system_prompt, generated_question_with_choice, generated["right_choice"])
            prompt_id = prompt_id.to(model.device)
            output = model(prompt_id)

            loss = cal_loss(output.logits[:, loss_idx+1:].squeeze(), prompt_id[:, loss_idx:-1].squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if show_generated:
                print("generated_question:\t", generated["question"])
                print("choice_list:\t", generated["choice_list"])
                print("right_choice:\t", generated["right_choice"])
                print("generated_question_with_choice:\t", generated_question_with_choice)

        if show_generated:
            print()
            
    elif mode == "full_choice":
        generated_batch = generate_choice_batch(model, tokenizer, question, num_return_sequences, num_choices)
        model = model.train()
        for generated in generated_batch:
            generated_question_with_choice = generated["question"]+'\n'+'\n'.join(generated["choice_list"])+config["choice_prompt"]
            right_full_choice = generated["choice_list"][ord(generated["right_choice"])-ord('A')]
            prompt_id, loss_idx = prompt_tokenizer(tokenizer, system_prompt, generated_question_with_choice, right_full_choice)
            prompt_id = prompt_id.to(model.device)
            output = model(prompt_id)

            loss = cal_loss(output.logits[:, loss_idx+1:].squeeze(), prompt_id[:, loss_idx:-1].squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if show_generated:
                print("generated_question:\t", generated["question"])
                print("choice_list:\t", generated["choice_list"])
                print("right_choice:\t", generated["right_choice"])
                print("generated_question_with_choice:\t", generated_question_with_choice)
                print("right_full_choice:\t", right_full_choice)

        if show_generated:
            print()
    
    model = model.eval()