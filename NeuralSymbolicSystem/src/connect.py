import requests
import json
from urllib.request import urlopen, Request

import src.logic

# http://207.154.220.61:10099/api/getNN


def post(url: str, d: dict) -> dict:
    return json.loads(requests.post(url, json=d).text)


def post_json_file(url: str, path: str) -> dict:
    # As an example
    with open(path, 'r') as json_file:
        return post(url, d=json.load(json_file))


def post_json_file_and_save_to_file(url: str, path_from: str, path_to: str):
    # As an example

    received_json = post_json_file(url, path_from)

    with open(path_to, 'w') as to_json_file:
        json.dump(received_json, to_json_file)


#def get(f, phrase, url='http://207.154.220.61:10099/api/'):
def get(f, phrase, url='http://64.225.103.216:10100/api/'):
#def get(f, phrase, url='http://127.0.0.1:10100/api/'):
    """
    Opens url using Request library

    :param f: to which function you want to connect (Str)
    :param phrase: request phrase (Str)
    :param url: url of server (Str)
    :return: response (Str)

    """
    request = Request(url+f, phrase.encode("utf-8"))
    # print(request.get_full_url())
    response = urlopen(request)
    html = response.read()
    response.close()
    return html.decode("utf-8")


def get_nn_recipe(logic_program: src.logic.LogicProgram,
                  abductive_goal: src.logic.Clause,
                  factors: src.logic.Factors) -> dict:
    """
    Get a Neural Network Recipe from API.

    :param logic_program: logic program (src.logic.LogicProgram)
    :param abductive_goal: abductive goal for abductive process (src.logic.Clause)
    :param factors: factors for neural network (src.logic.Factors)
    :return: recipe for neural network (dict)

    """
    request_dict = {"lp": logic_program.to_dict(),
                    "abductive_goal": abductive_goal.to_dict(),
                    "factors": factors.to_dict()}

    request_json = json.dumps(request_dict)
    return json.loads(get('lp2nn', request_json))


def get_lp_from_nn(order_inp: [str], order_out: [str], amin: float, io_pairs: [tuple]) -> dict:

    request_dict = {"orderInp": order_inp,
                    "orderOut": order_out,
                    "amin": amin,
                    "ioPairs": io_pairs}
    request_json = json.dumps(request_dict)#.replace('"', r'\"')
    response = get('nn2lp', request_json)
    return json.loads(response)
