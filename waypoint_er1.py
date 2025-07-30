import requests
import base64
import cv2
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from gradio_client import Client, handle_file

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_waypoints(image_rgb, task_instruction="Locate the green object.", prompt_type=None):
    # import IPython; IPython.embed()
    # assert image_rgb.shape == (480, 640, 3) # may not be 480x640 eg google robot 512x640
    
    if "Select grasp point" in task_instruction:
        prompt_type = "pick"
    elif "Select place point" in task_instruction:
        prompt_type = "place"

    assert prompt_type is not None, f"Unknown task instruction: {task_instruction}"

    if prompt_type == "pick":
        prompt = f"Provide one or more points coordinate of objects region this sentence describes: {task_instruction}. The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. The answer consists only of several coordinate points, with the overall format being: <think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    elif prompt_type == "place":
        question = f"You are currently a robot performing robotic manipulation tasks. The task instruction is: {task_instruction}. Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved."
        # question = f"Provide one or more points coordinate of objects region this sentence describes: {task_instruction}. The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>."
        instruction_following = (
            r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
            r'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. '
            r'The answer consists only of several coordinate points, with the overall format being: '
            r'<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>'
        )
        prompt = question + "\n" + instruction_following
    else:
        assert False
        
    cv2.imwrite("upload.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    client = Client("http://127.0.0.1:7860/")
    
    client.predict(
        file=handle_file('upload.png'),
        api_name="/add_file"
    )
    client.predict(
        text=prompt,
        api_name="/add_text"
    )
    result = client.predict(
        _chatbot=[[{"file":handle_file('upload.png'),"alt_text":None},None],[prompt,None]],
        api_name="/predict"
    )
    print(result[-1][-1])

    match = re.search(r"<[Aa]nswer>.*?<[Pp]oint>(.*?)</[Pp]oint>.*?</[Aa]nswer>", result[-1][-1], re.DOTALL)
    if not match:
        raise Exception("No answer found")
    answer = match.group(1)
    points = np.array(json.loads(answer), dtype=np.float32)
    
    if prompt_type == "pick":
        center_point = points[0]
    elif prompt_type == "place":
        center_point = points[-1]
    else:
        assert False

    return center_point, points

if __name__ == "__main__":
    image = cv2.imread("20250514_023729 stack the green block on the yellow block.png")
    image = cv2.resize(image, (640, 480))
    center_point, points = get_waypoints(image, "stack the green block on the yellow block", "place")
    
    # plot points
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    cv2.circle(image, (int(center_point[0]), int(center_point[1])), 5, (0, 255, 0), -1)
    cv2.imwrite("image_with_points.png", image)
    
    print(points)
    print(center_point)